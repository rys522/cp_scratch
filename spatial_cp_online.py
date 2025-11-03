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
def predict_next_cv(y_t, y_tminus1, box):
    """Constant-velocity predictor: y_{t+1|t} = y_t + (y_t - y_{t-1}) with reflection."""
    y_pred = y_t + (y_t - y_tminus1)
    return reflect_to_box(y_pred.copy(), box)

# ==========================================
# 2) Build TRAIN residual corpus for f(x)
#    f(x) = d(x, Y_{t+1}) - d(x, Yhat_{t+1|t})
# ==========================================
# Reuse obst_trajs_train already built above.
# Residuals are defined for t = 1..Tsteps-2 (so that t-1, t, t+1 exist).
T_res = Tsteps - 2
residuals_train = np.zeros((N, T_res, H, W), dtype=np.float32)

for i in range(N):
    o_traj = obst_trajs_train[i]  # (Tsteps, 2)
    for t in range(1, Tsteps-1):
        y_tm1, y_t, y_tp1 = o_traj[t-1], o_traj[t], o_traj[t+1]
        y_hat = predict_next_cv(y_t, y_tm1, box)
        F_true = distance_field_single_obstacle(y_tp1, Xg, Yg).astype(np.float32)
        F_pred = distance_field_single_obstacle(y_hat, Xg, Yg).astype(np.float32)
        residuals_train[i, t-1] = F_pred - F_true  # index shift: t-1 ∈ [0..Tsteps-3]

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
# 4) Precompute residual CP lower bounds for all time indices
#    and build the one-step forecasted distance lower bound
# ============================================================

# Precompute g_upper for all t
g_upper_all_t = np.asarray(
    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(cp_residual_upper_at_t)(t_idx) for t_idx in range(T_res)
    ),
    dtype=np.float32
)

# Build lower bound for true distance at t+1:
lower_forecast_at_tplus1 = np.zeros((Tsteps, H, W), dtype=np.float32)
for t in range(1, Tsteps-1):
    y_tm1, y_t = obst_test[t-1], obst_test[t]
    y_hat = predict_next_cv(y_t, y_tm1, box)
    d_pred = distance_field_single_obstacle(y_hat, Xg, Yg).astype(np.float32)
    g_up   = g_upper_all_t[t-1]                       # align index
    lower_forecast_at_tplus1[t+1] = np.maximum(d_pred - g_up, 0.0)

# ============================================================
# 6) Evaluate coverage: true-unsafe pixels missed by prediction
# ============================================================

def evaluate_test_with_seed(seed, g_upper_all_t):
    rng = np.random.default_rng(seed)
    obst_test = simulate_brownian_2d(Tsteps, dt, sigma_obst, x0_obst, box, rng)
    missed, total = [], []

    for t in range(1, Tsteps - 1):
        F_true = distance_field_single_obstacle(obst_test[t + 1], Xg, Yg)
        y_tm1, y_t = obst_test[t - 1], obst_test[t]
        y_hat = predict_next_cv(y_t, y_tm1, box)
        d_pred = distance_field_single_obstacle(y_hat, Xg, Yg)
        g_up = g_upper_all_t[t - 1]
        mask_true = (F_true < safe_threshold)
        mask_pred = (d_pred < safe_threshold + g_up)
        missed.append(np.sum(mask_true & ~mask_pred))
        total.append(np.sum(mask_true))

    coverage = 1 - np.array(missed) / (np.array(total) + 1e-8)
    return coverage.mean(), np.mean(missed)

seeds = np.arange(100, 110)
results = Parallel(n_jobs=-1)(
    delayed(evaluate_test_with_seed)(s, g_upper_all_t) for s in seeds
)

coverages, missed_means = zip(*results)
print(f"Mean coverage: {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
print(f"Avg missed pixels: {np.mean(missed_means):.1f} ± {np.std(missed_means):.1f}")

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
    # Valid frames: t ∈ [1..Tsteps-2]
    if frame < 1 or frame >= Tsteps - 1:
        return

    for ax in axs.flat:
        ax.cla()

    # --- (1,1) True field at t+1
    F_true = distance_field_single_obstacle(obst_test[frame + 1], Xg, Yg)
    axs[0, 0].imshow(F_true, extent=[0, box, 0, box], origin="lower")
    axs[0, 0].plot(robot_test[:frame + 2, 0], robot_test[:frame + 2, 1], 'b-', lw=1)
    axs[0, 0].plot(obst_test[:frame + 2, 0],  obst_test[:frame + 2, 1],  'r-', lw=1)
    axs[0, 0].plot(robot_test[frame + 1, 0], robot_test[frame + 1, 1], 'bo', ms=5)
    axs[0, 0].plot(obst_test[frame + 1, 0],  obst_test[frame + 1, 1],  'rx', ms=7, mew=2)
    axs[0, 0].set_title(f"{titles[0]} (t={frame} → t+1={frame + 1})")
    axs[0, 0].set_xlim([0, box])
    axs[0, 0].set_ylim([0, box])
    axs[0, 0].set_aspect('equal')

    # --- (1,2) Forecasted CP lower bound field
    lower_field = lower_forecast_at_tplus1[frame + 1]
    axs[0, 1].imshow(lower_field, extent=[0, box, 0, box], origin="lower")
    axs[0, 1].plot(robot_test[:frame + 2, 0], robot_test[:frame + 2, 1], 'b-', lw=1)
    axs[0, 1].plot(obst_test[:frame + 2, 0],  obst_test[:frame + 2, 1],  'r-', lw=1)
    axs[0, 1].plot(robot_test[frame + 1, 0], robot_test[frame + 1, 1], 'bo', ms=5)
    axs[0, 1].plot(obst_test[frame + 1, 0],  obst_test[frame + 1, 1],  'rx', ms=7, mew=2)
    axs[0, 1].set_title(f"{titles[1]} (t={frame} → t+1={frame + 1})")
    axs[0, 1].set_xlim([0, box])
    axs[0, 1].set_ylim([0, box])
    axs[0, 1].set_aspect('equal')

    # --- (2,1) True unsafe mask
    cmap = plt.cm.colors.ListedColormap(['white', 'red'])
    mask_true = (F_true < safe_threshold).astype(float)
    axs[1, 0].imshow(mask_true, extent=[0, box, 0, box], origin="lower", cmap=cmap, vmin=0, vmax=1)
    axs[1, 0].set_title(f"{titles[2]} (t={frame + 1})")
    axs[1, 0].set_xlim([0, box])
    axs[1, 0].set_ylim([0, box])
    axs[1, 0].set_aspect('equal')

    # --- (2,2) Forecasted unsafe mask + True mask overlay
    mask_forecast = (lower_field < safe_threshold).astype(float)
    mask_true = (F_true < safe_threshold).astype(float)

    # Base: forecasted unsafe (red)
    axs[1, 1].imshow(mask_forecast, extent=[0, box, 0, box],
                     origin="lower", cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    # Overlay: true unsafe (blue)
    axs[1, 1].imshow(mask_true, extent=[0, box, 0, box],
                     origin="lower", cmap="Blues", alpha=0.5, vmin=0, vmax=1)

    axs[1, 1].set_title(f"{titles[3]} (t={frame + 1})")
    axs[1, 1].set_xlim([0, box])
    axs[1, 1].set_ylim([0, box])
    axs[1, 1].set_aspect('equal')

# Run animation
ani = FuncAnimation(fig, update, frames=range(1, Tsteps - 1), blit=False, interval=150)
plt.show()

