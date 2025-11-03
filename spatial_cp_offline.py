import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from numpy.linalg import det

# ============================================================
# Brownian motion
# ============================================================
def reflect_to_box(pos, box):
    for i in range(2):
        if pos[i] < 0:
            pos[i] = -pos[i]
        if pos[i] > box:
            pos[i] = 2*box - pos[i]
        while pos[i] < 0 or pos[i] > box:
            if pos[i] < 0:
                pos[i] = -pos[i]
            if pos[i] > box:
                pos[i] = 2*box - pos[i]
    return pos

def simulate_brownian_2d(n_steps, dt, sigma, x0, box, rng):
    traj = np.zeros((n_steps, 2), dtype=float)
    x = np.array(x0, dtype=float)
    traj[0] = reflect_to_box(x.copy(), box)
    for t in range(1, n_steps):
        step = rng.normal(0.0, sigma*np.sqrt(dt), size=2)
        x = x + step
        x = reflect_to_box(x, box)
        traj[t] = x
    return traj

def distance_field_single_obstacle(point, grid_x, grid_y):
    px, py = point
    d2 = (grid_x - px)**2 + (grid_y - py)**2
    return np.sqrt(d2)

# ============================================================
# Parameters
# ============================================================
rng = np.random.default_rng(2025)

H, W   = 40, 40
Tsteps = 60
box    = 10.0
xs = np.linspace(0, box, W)
ys = np.linspace(0, box, H)
Xg, Yg = np.meshgrid(xs, ys)

N = 200
dt          = 1.0
sigma_robot = 0.35
sigma_obst  = 0.45
x0_robot    = [7.0, 3.0]
x0_obst     = [3.0, 7.0]

p     = 12
K     = 3
alpha = 0.1

# ============================================================
# Training data (full field)
# ============================================================
fields = np.zeros((N, Tsteps, H, W), dtype=float)
robot_trajs = np.zeros((N, Tsteps, 2), dtype=float)
obst_trajs  = np.zeros((N, Tsteps, 2), dtype=float)

for i in range(N):
    r_traj = simulate_brownian_2d(Tsteps, dt, sigma_robot, x0_robot, box, rng)
    o_traj = simulate_brownian_2d(Tsteps, dt, sigma_obst,  x0_obst,  box, rng)
    robot_trajs[i] = r_traj
    obst_trajs[i]  = o_traj
    for t in range(Tsteps):
        fields[i, t] = distance_field_single_obstacle(o_traj[t], Xg, Yg)

D = Tsteps * H * W
Y = fields.reshape(N, D)

# ============================================================
# PCA + GMM + CP
# ============================================================
pca = PCA(n_components=min(p, D), svd_solver="randomized", random_state=0)
scores = pca.fit_transform(Y)
A = pca.components_
mean_field = pca.mean_.reshape(Tsteps, H, W)
p_eff = A.shape[0]

Xi_train, Xi_cal = train_test_split(scores, test_size=0.3, random_state=0)
gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0).fit(Xi_train)

logf_cal = gmm.score_samples(Xi_cal)
q_log = np.quantile(logf_cal, 1 - alpha)
lam   = np.exp(q_log)

pis, mus, Sigmas = gmm.weights_, gmm.means_, gmm.covariances_

def rk_squared(k, lam):
    tau = lam / (K * pis[k])
    Sigma_k = Sigmas[k]
    det_Sk = det(Sigma_k)
    if det_Sk <= 0:
        Sigma_k = Sigma_k + 1e-8*np.eye(Sigma_k.shape[0])
        det_Sk = det(Sigma_k)
    val = tau * ((2*np.pi)**(p_eff/2)) * np.sqrt(det_Sk)
    if not np.isfinite(val) or val <= 0.0:
        return 0.0
    return max(0.0, -2.0*np.log(val))

rks = np.array([np.sqrt(rk_squared(k, lam)) for k in range(K)])

def band_fields(rks):
    lower = np.full((K, D), np.nan)
    upper = np.full((K, D), np.nan)
    for k in range(K):
        mu_k = mus[k]
        rk   = rks[k]
        center = (A.T @ mu_k)
        Sk = Sigmas[k]
        AS = (A.T @ Sk)
        quad_diag = np.einsum("dp,dp->d", AS, A.T)
        quad_diag = np.clip(quad_diag, 0.0, np.inf)
        rad = rk * np.sqrt(quad_diag)
        lower[k] = center - rad
        upper[k] = center + rad
    lower_env = np.nanmin(lower, axis=0).reshape(Tsteps, H, W)
    upper_env = np.nanmax(upper, axis=0).reshape(Tsteps, H, W)
    return lower_env, upper_env

lower_env, _ = band_fields(rks)
lower_field = np.maximum(mean_field + lower_env, 0.0)

# ============================================================
# Animation: actual field vs CP lower bound
# ============================================================
i_show = 0
rpath, opath = robot_trajs[i_show], obst_trajs[i_show]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
titles = ["Actual distance field", "CP lower bound"]

def update(frame):
    axs[0].cla()
    axs[1].cla()

    # Actual field
    F_true = distance_field_single_obstacle(opath[frame], Xg, Yg)
    im0 = axs[0].imshow(F_true, extent=[0,box,0,box], origin="lower")
    axs[0].plot(rpath[:frame+1,0], rpath[:frame+1,1], 'b-', lw=1)
    axs[0].plot(opath[:frame+1,0], opath[:frame+1,1], 'r-', lw=1)
    axs[0].plot(rpath[frame,0], rpath[frame,1], 'bo')
    axs[0].plot(opath[frame,0], opath[frame,1], 'rx', mew=2)
    axs[0].set_title(f"{titles[0]} (t={frame})")
    axs[0].set_xlim([0,box])
    axs[0].set_ylim([0,box])

    # CP lower bound
    im1 = axs[1].imshow(lower_field[frame], extent=[0,box,0,box], origin="lower")
    axs[1].plot(rpath[:frame+1,0], rpath[:frame+1,1], 'b-', lw=1)
    axs[1].plot(opath[:frame+1,0], opath[:frame+1,1], 'r-', lw=1)
    axs[1].plot(rpath[frame,0], rpath[frame,1], 'bo')
    axs[1].plot(opath[frame,0], opath[frame,1], 'rx', mew=2)
    axs[1].set_title(f"{titles[1]} (t={frame})")
    axs[1].set_xlim([0,box])
    axs[1].set_ylim([0,box])

    return [im0, im1]

ani = FuncAnimation(fig, update, frames=Tsteps, blit=False, interval=200)
plt.show()