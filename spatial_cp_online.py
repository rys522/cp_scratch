import os
# (optional) avoid BLAS oversubscription when we parallelize over time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from numpy.linalg import det
from joblib import Parallel, delayed



# ============================================================
# Parameters
# ============================================================
rng = np.random.default_rng(2023)

# space/time
H, W   = 64, 64
Tsteps = 120
box    = 100.0
xs = np.linspace(0, box, W)
ys = np.linspace(0, box, H)
Xg, Yg = np.meshgrid(xs, ys)

# prediction parameters

time_horizon = 3  # one-step ahead

# corpus size
N = 1000

# motion scales
dt, sigma_robot, sigma_obst = 1.0, 4, 4
x0_robot, x0_obst = [70.0, 30.0], [30.0, 70.0]
safe_threshold = 8.0

# PCA / GMM / CP hyperparams
p_base = 3
K      = 4
alpha  = 0.05 # miscoverage level

n_workers = os.cpu_count()-2

# ============================================================
# Brownian motion + distance field utilities
# ============================================================


def reflect_to_box(pos, box):
    """Reflect a 2D point back into [0, box]^2 if it crosses the boundary."""
    for i in range(2):
        if pos[i] < 0:
            pos[i] = -pos[i]
        if pos[i] > box:
            pos[i] = 2 * box - pos[i]
        while pos[i] < 0 or pos[i] > box:
            if pos[i] < 0:
                pos[i] = -pos[i]
            if pos[i] > box:
                pos[i] = 2 * box - pos[i]
    return pos

def simulate_brownian_2d(n_steps, dt, sigma, x0, box, rng):
    """Simulate a 2D Brownian trajectory with reflecting boundaries."""
    traj = np.zeros((n_steps, 2), dtype=float)
    x = np.array(x0, dtype=float)
    traj[0] = reflect_to_box(x.copy(), box)
    for t in range(1, n_steps):
        step = rng.normal(0.0, sigma * np.sqrt(dt), size=2)
        x = x + step
        x = reflect_to_box(x, box)
        traj[t] = x
    return traj  # (T, 2)

def distance_field_single_obstacle(point, grid_x, grid_y):
    """Distance from each grid point to a single obstacle location."""
    px, py = point
    d2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
    return np.sqrt(d2)  # (H, W)



# ============================================================
# Build TRAINING CORPUS: N episodes of fields f_i(x,y,t)
# ============================================================
fields_train = np.zeros((N, Tsteps, H, W), dtype=np.float32)
robot_trajs_train = np.zeros((N, Tsteps, 2), dtype=np.float32)
obst_trajs_train  = np.zeros((N, Tsteps, 2), dtype=np.float32)

for i in range(N):
    r_traj = simulate_brownian_2d(Tsteps, dt, sigma_robot, x0_robot, box, rng).astype(np.float32)
    o_traj = simulate_brownian_2d(Tsteps, dt, sigma_obst,  x0_obst,  box, rng).astype(np.float32)
    robot_trajs_train[i] = r_traj
    obst_trajs_train[i]  = o_traj
    for t in range(Tsteps):
        fields_train[i, t] = distance_field_single_obstacle(o_traj[t], Xg, Yg).astype(np.float32)

# ============================================================
# Build a TEST episode (held out) for visualization only
# ============================================================
robot_test = simulate_brownian_2d(Tsteps, dt, sigma_robot, x0_robot, box, rng)
obst_test  = simulate_brownian_2d(Tsteps, dt, sigma_obst,  x0_obst,  box, rng)

# ============================
# 1) Simple obstacle predictor
# ============================
def predict_h_step_cv(y_t, y_tminus1, h, box):
    """
    h-step constant-velocity prediction with per-step reflection.

    y_{t+1|t} = reflect(y_t + (y_t - y_{t-1}))
    y_{t+2|t} = reflect(y_{t+1|t} + (y_t - y_{t-1}))
    ...
    y_{t+h|t} repeats the same velocity and reflects at each step.
    """
    v = y_t - y_tminus1
    y_hat = np.array(y_t, dtype=float)
    for _ in range(h):
        y_hat = y_hat + v
        y_hat = reflect_to_box(y_hat, box)
    return y_hat

# ==========================================
# 2) Build TRAIN residual corpus for f_h(x)
#     f_h(x) = d(x, Y_{t+h}) - d(x, Ỹ_{t+h|t})
# ==========================================
h = int(time_horizon)
assert h >= 1, "time_horizon must be >= 1"

# Residuals are defined for t = 1..Tsteps-h-1 so that (t-1, t, t+h) exist.
T_res = Tsteps - h - 1
if T_res <= 0:
    raise ValueError(f"time_horizon={h} is too large for Tsteps={Tsteps}")

residuals_train = np.zeros((N, T_res, H, W), dtype=np.float32)

for i in range(N):
    o_traj = obst_trajs_train[i]  # (Tsteps, 2)
    # valid t: 1 .. Tsteps - h - 1  (inclusive)
    for t in range(1, Tsteps - h):
        y_tm1, y_t, y_tph = o_traj[t-1], o_traj[t], o_traj[t+h]
        y_hat = predict_h_step_cv(y_t, y_tm1, h, box)
        F_true = distance_field_single_obstacle(y_tph, Xg, Yg).astype(np.float32)
        F_pred = distance_field_single_obstacle(y_hat,  Xg, Yg).astype(np.float32)
        residuals_train[i, t-1] = F_pred - F_true  # index shift: (t-1) ∈ [0 .. T_res-1], at time t+h

# ============================================================
# 3) Functional CP on residual field at a given time t
#    Returns upper envelope \underline f_t(x) on the residuals
# ============================================================
def cp_residual_upper_at_t(t_idx: int) -> np.ndarray:
    D = H * W
    Yw = residuals_train[:, t_idx].reshape(N, D).astype(np.float32)

    p_eff = int(min(p_base, N, D))
    pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(Yw)
    A = pca.components_.astype(np.float64)   # (p_eff, D)
    mean_vec = pca.mean_.astype(np.float64)  # (D,)

    Xi_train, Xi_cal = train_test_split(scores, test_size=0.3, random_state=0)
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0).fit(Xi_train)

    logf_cal = gmm.score_samples(Xi_cal)
    q_log = np.quantile(logf_cal, 1 - alpha)
    lam = np.exp(q_log)

    pis, mus, Sigmas = gmm.weights_, gmm.means_, gmm.covariances_

    def rk_sq(k: int) -> float:
        tau = lam / (K * pis[k])
        Sk = Sigmas[k]
        det_Sk = det(Sk)
        if det_Sk <= 0:
            Sk = Sk + 1e-8 * np.eye(Sk.shape[0])
            det_Sk = det(Sk)
        val = tau * ((2*np.pi)**(p_eff/2)) * np.sqrt(det_Sk)
        if not np.isfinite(val) or val <= 0.0:
            return 0.0
        return max(0.0, -2.0*np.log(val))

    rks = np.array([np.sqrt(rk_sq(k)) for k in range(K)], dtype=np.float64)

    AT = A.T
    upper = np.full((K, D), np.nan, dtype=np.float64)
    for k in range(K):
        mu_k = mus[k]
        rk = rks[k]
        Sk = Sigmas[k]
        center = AT @ mu_k
        AS = AT @ Sk
        quad_diag = np.einsum("dp,dp->d", AS, AT)
        quad_diag = np.clip(quad_diag, 0.0, np.inf)
        rad = rk * np.sqrt(quad_diag)
        upper[k] = center + rad        # <-- upper envelope in PCA-projected space

    g_upper_vec = mean_vec + np.nanmax(upper, axis=0)   # pointwise max over components
    return g_upper_vec.reshape(H, W).astype(np.float32)

# ============================================================
# 4) Precompute residual CP upper envelopes for all time indices
#    and build the h-step forecasted distance lower bound
# ============================================================

# Precompute g_upper for all residual time indices
g_upper_all_t = np.asarray(
    Parallel(n_jobs=n_workers)(
        delayed(cp_residual_upper_at_t)(t_idx) for t_idx in range(T_res)
    ),
    dtype=np.float32
)

# Build lower bound for true distance at t+h:
# We'll store at index (t+h), consistent with the "true field at t+h".
lower_forecast_at_tplusH = np.zeros((Tsteps, H, W), dtype=np.float32)

# For test episode (held out)
for t in range(1, Tsteps - h):
    y_tm1, y_t = obst_test[t-1], obst_test[t]
    y_hat = predict_h_step_cv(y_t, y_tm1, h, box)
    d_pred = distance_field_single_obstacle(y_hat, Xg, Yg).astype(np.float32)
    g_up   = g_upper_all_t[t-1]   # align residual index t-1
    lower_forecast_at_tplusH[t + h] = np.maximum(d_pred - g_up, 0.0)

# ============================================================
# 6) Evaluate coverage: true-unsafe pixels missed by prediction (h-step ahead)
# ============================================================

def evaluate_test_with_seed(seed, g_upper_all_t, h):
    """
    Measures mean coverage and mean missed pixels for the h-step forecast.
    For each valid t, we compare:
      True unsafe at (t+h): d(x, Y_{t+h}) < safe_threshold
      Forecast unsafe set:  d_pred(x; Ỹ_{t+h|t}) - g_upper_t(x) < safe_threshold
      (equivalently, d_pred < safe_threshold + g_upper)
    """
    rng = np.random.default_rng(seed)
    obst_test = simulate_brownian_2d(Tsteps, dt, sigma_obst, x0_obst, box, rng)

    missed, total = [], []

    for t in range(1, Tsteps - h):
        F_true = distance_field_single_obstacle(obst_test[t + h], Xg, Yg)

        y_tm1, y_t = obst_test[t - 1], obst_test[t]
        y_hat = predict_h_step_cv(y_t, y_tm1, h, box)
        d_pred = distance_field_single_obstacle(y_hat, Xg, Yg)

        g_up = g_upper_all_t[t - 1]

        mask_true = (F_true < safe_threshold)
        # Forecasted unsafe if predicted lower bound falls below threshold:
        # lower_bound = max(d_pred - g_up, 0) < threshold  =>  d_pred < threshold + g_up
        mask_pred = (d_pred < safe_threshold + g_up)

        missed.append(np.sum(mask_true & ~mask_pred))
        total.append(np.sum(mask_true))

    coverage = 1 - np.array(missed) / (np.array(total) + 1e-8)
    return coverage.mean(), np.mean(missed)

seeds = np.arange(100, 110)
results = Parallel(n_jobs=-1)(
    delayed(evaluate_test_with_seed)(s, g_upper_all_t, h) for s in seeds
)

coverages, missed_means = zip(*results)
print(f"[h={h}] Mean coverage: {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
print(f"[h={h}] Avg missed pixels: {np.mean(missed_means):.1f} ± {np.std(missed_means):.1f}")

# ============================================================
# Visualization: true vs. forecasted CP lower fields
# ============================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
titles = [
    "True distance field at t+1",
    "Predicted + CP residual lower → Lower bound at t+1",
    f"Unsafe mask (true d < {safe_threshold})",
    "Unsafe mask: Forecast (red) + True overlay (blue)"
]

def update(frame):
    """
    Animation frame is 't' (decision time). We visualize at time (t+h).
    Valid frames: t ∈ [1 .. Tsteps - h - 1].
    """
    if frame < 1 or frame >= Tsteps - h:
        return

    for ax in axs.flat:
        ax.cla()

    tpH = frame + h  # time where we compare truth vs forecast

    # --- (1,1) True field at t+h
    F_true = distance_field_single_obstacle(obst_test[tpH], Xg, Yg)
    axs[0, 0].imshow(F_true, extent=[0, box, 0, box], origin="lower")
    axs[0, 0].plot(robot_test[:tpH + 1, 0], robot_test[:tpH + 1, 1], 'b-', lw=1)
    axs[0, 0].plot(obst_test[:tpH + 1, 0],  obst_test[:tpH + 1, 1],  'r-', lw=1)
    axs[0, 0].plot(robot_test[tpH, 0], robot_test[tpH, 1], 'bo', ms=5)
    axs[0, 0].plot(obst_test[tpH, 0],  obst_test[tpH, 1],  'rx', ms=7, mew=2)
    axs[0, 0].set_title(f"True distance field at t+h (t={frame}, h={h} → {tpH})")
    axs[0, 0].set_xlim([0, box]); axs[0, 0].set_ylim([0, box]); axs[0, 0].set_aspect('equal')

    # --- (1,2) Forecasted CP lower bound field at t+h
    lower_field = lower_forecast_at_tplusH[tpH]
    axs[0, 1].imshow(lower_field, extent=[0, box, 0, box], origin="lower")
    axs[0, 1].plot(robot_test[:tpH + 1, 0], robot_test[:tpH + 1, 1], 'b-', lw=1)
    axs[0, 1].plot(obst_test[:tpH + 1, 0],  obst_test[:tpH + 1, 1],  'r-', lw=1)
    axs[0, 1].plot(robot_test[tpH, 0], robot_test[tpH, 1], 'bo', ms=5)
    axs[0, 1].plot(obst_test[tpH, 0],  obst_test[tpH, 1],  'rx', ms=7, mew=2)
    axs[0, 1].set_title(f"Forecasted CP lower bound at t+h (t={frame}, h={h} → {tpH})")
    axs[0, 1].set_xlim([0, box]); axs[0, 1].set_ylim([0, box]); axs[0, 1].set_aspect('equal')

    # --- (2,1) True unsafe mask at t+h
    cmap = plt.cm.colors.ListedColormap(['white', 'red'])
    mask_true = (F_true < safe_threshold).astype(float)
    axs[1, 0].imshow(mask_true, extent=[0, box, 0, box], origin="lower", cmap=cmap, vmin=0, vmax=1)
    axs[1, 0].set_title(f"True unsafe mask at t+h (t={frame}, h={h} → {tpH})")
    axs[1, 0].set_xlim([0, box]); axs[1, 0].set_ylim([0, box]); axs[1, 0].set_aspect('equal')

    # --- (2,2) Forecast vs True unsafe overlay at t+h
    mask_forecast = (lower_field < safe_threshold).astype(float)

    axs[1, 1].imshow(mask_forecast, extent=[0, box, 0, box],
                     origin="lower", cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axs[1, 1].imshow(mask_true, extent=[0, box, 0, box],
                     origin="lower", cmap="Blues", alpha=0.5, vmin=0, vmax=1)
    axs[1, 1].set_title(f"Unsafe forecast (red) vs True (blue) at t+h (t={frame}, h={h} → {tpH})")
    axs[1, 1].set_xlim([0, box]); axs[1, 1].set_ylim([0, box]); axs[1, 1].set_aspect('equal')

# Run animation
ani = FuncAnimation(fig, update, frames=range(1, Tsteps - 1), blit=False, interval=150)
plt.show()

