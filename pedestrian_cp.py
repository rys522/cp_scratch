from __future__ import annotations
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D
from numpy.linalg import det
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle

# ============================================================
# 0) Global parameters (EGO-CENTRIC VERSION)
# ============================================================

# Data
DATA_PATH = "./data/seq_eth"   # expects key 'episodes' with shape (E,T,2) or (E,T,M,2)
USE_MULTI_OBST = False           # if True, truth field uses min-dist to all agents at each t
TARGET_ID = 0                    # used when USE_MULTI_OBST=False and episodes are (E,T,M,2)

# Grid / domain (world is used only for loading & reflection; all fields are EGO)
H, W = 64, 64
BOX = 100.0                      # world normalization box (for data scaling & reflection)
SCALING = None                   # (sx, sy) world->meters; if None, computed from data bounds

# Ego grid (everything is computed in this frame)
EGO_BOX = 60.0                   # extent in meters around the agent ([-EGO_BOX/2, EGO_BOX/2]^2)

# Corpus / horizon / seeds
TSTEPS = 120
N_TRAIN = 1000
TIME_HORIZON = 6
SEED = 2023
EVAL_SEEDS = np.arange(SEED + 100, SEED + 110)

# Safety threshold (meters in normalized space)
SAFE_THRESHOLD = 8.0

# CP hyperparams
P_BASE = 3
K_MIX = 4
ALPHA = 0.05
TEST_SIZE = 0.30
RANDOM_STATE = 0
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
BACKEND = "loky"

# Agent trajectory options (in WORLD coords, per time t)
AGENT_SOURCE = "random_walk"    # "fixed" | "copy_obst" | "random_walk" | "from_array"
AGENT_FIXED_XY = (BOX/2, BOX/2)
EGO_ALIGN_HEADING = False        # if you have heading, you can rotate the ego frame

# ============================================================
# 1) Utils: grids, transforms, distance fields (EGO)
# ============================================================

def build_grid(ego_box: float, H: int, W: int):
    xs = np.linspace(-ego_box/2, ego_box/2, W)
    ys = np.linspace(-ego_box/2, ego_box/2, H)
    Xg, Yg = np.meshgrid(xs, ys)
    return xs, ys, Xg, Yg


def reflect_to_box(pos: np.ndarray, box: float) -> np.ndarray:
    """Reflect a 2D point back into [0, box]^2 (for CV rollouts at boundaries)."""
    pos = np.asarray(pos, dtype=float).copy()
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


def world_to(points: np.ndarray, agent_xy: np.ndarray, align_heading=False, heading_rad=0.0):
    """Translate (and optionally rotate) world coords to ego coords centered at agent_xy.
    points: (...,2); agent_xy: (2,)
    returns (...,2)
    """
    rel = points - agent_xy[None, ...] if points.ndim == 2 else points - agent_xy
    if align_heading:
        c, s = np.cos(-heading_rad), np.sin(-heading_rad)
        R = np.array([[c, -s],[s, c]], dtype=np.float32)
        if rel.ndim == 2:
            rel = rel @ R.T
        else:
            rel = rel @ R.T
    return rel


def distance_field_points(points_rel: np.ndarray, Xg_rel: np.ndarray, Yg_rel: np.ndarray) -> np.ndarray:
    """Min Euclidean distance to one or multiple points, in EGO frame."""
    if points_rel.ndim == 1:  # (2,)
        px, py = points_rel
        d2 = (Xg_rel - px) ** 2 + (Yg_rel - py) ** 2
        return np.sqrt(d2).astype(np.float32)
    # (M,2)
    dmin = None
    for p in points_rel:
        px, py = p
        d2 = (Xg_rel - px) ** 2 + (Yg_rel - py) ** 2
        d = np.sqrt(d2)
        dmin = d if dmin is None else np.minimum(dmin, d)
    return dmin.astype(np.float32)


# ============================================================
# 2) Data loading + normalization (WORLD)
# ============================================================

def _apply_homography_if_available(xy: np.ndarray, H_path: str | None) -> np.ndarray:
    if not H_path or not os.path.isfile(H_path):
        return xy
    try:
        H = np.loadtxt(H_path)
        if H.shape != (3, 3):
            return xy
        ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
        homo = np.concatenate([xy, ones], axis=1)
        trans = (H @ homo.T).T
        w = np.clip(trans[:, 2], 1e-8, None)
        xy_w = trans[:, :2] / w[:, None]
        return xy_w.astype(np.float32)
    except Exception:
        return xy


def _load_eth_folder(folder: str) -> list[np.ndarray]:
    obsmat_path = os.path.join(folder, "obsmat.txt")
    if not os.path.isfile(obsmat_path):
        raise FileNotFoundError(f"obsmat.txt not found in {folder}")
    M = np.loadtxt(obsmat_path)
    if M.ndim != 2 or M.shape[1] < 4:
        raise ValueError("obsmat.txt must be a 2D table with >= 4 columns")

    frame = M[:, 0].astype(int)
    pid   = M[:, 1].astype(int)

    cand_pairs = [(2, 3), (3, 2), (4, 5)]
    xy = None
    for cx, cy in cand_pairs:
        if max(cx, cy) < M.shape[1]:
            tmp = M[:, [cx, cy]].astype(np.float32)
            if np.isfinite(tmp).all() and (tmp.std(axis=0) > 1e-6).all():
                xy = tmp
                break
    if xy is None:
        xy = M[:, [2, 3]].astype(np.float32)

    H_path = os.path.join(folder, "H.txt")
    xy = _apply_homography_if_available(xy, H_path)

    episodes: list[np.ndarray] = []
    for p in np.unique(pid):
        sel = pid == p
        fr  = frame[sel]
        pts = xy[sel]
        order = np.argsort(fr)
        traj = pts[order]
        if len(traj) >= 5 and np.isfinite(traj).all():
            episodes.append(traj.astype(np.float32))
    if len(episodes) == 0:
        raise ValueError("No valid trajectories extracted from ETH folder.")
    return episodes


def load_episodes(data_path: str) -> np.ndarray | list[np.ndarray]:
    if os.path.isdir(data_path):
        return _load_eth_folder(data_path)
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npz":
        data = np.load(data_path, allow_pickle=True)
        return data["episodes"]
    elif ext in (".pkl", ".pickle"):
        with open(data_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "episodes" in obj:
            return obj["episodes"]
        return obj
    else:
        raise ValueError(f"Unsupported data path: {data_path}")


def normalize_world(arr, box: float, scaling=None):
    if isinstance(arr, list):
        all_xy = np.concatenate([ep.reshape(-1, 2) for ep in arr], axis=0)
    else:
        all_xy = arr.reshape(-1, arr.shape[-1])
    x_min, y_min = all_xy[:, 0].min(), all_xy[:, 1].min()
    x_max, y_max = all_xy[:, 0].max(), all_xy[:, 1].max()

    if scaling is None:
        span = max(x_max - x_min, y_max - y_min, 1e-6)
        sx = sy = box / span
    else:
        sx, sy = scaling

    def _norm_ep(ep):
        out = ep.astype(np.float32).copy()
        out[..., 0] = (out[..., 0] - x_min) * sx
        out[..., 1] = (out[..., 1] - y_min) * sy
        return out

    if isinstance(arr, list):
        return [_norm_ep(ep) for ep in arr]
    else:
        return _norm_ep(arr)


def split_train_test(E: int, split_ratio: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_train = int(split_ratio * E)
    return perm[:n_train], perm[n_train:]


def select_obstacle_traj(ep: np.ndarray, use_multi: bool, target_id: int):
    if ep.ndim == 2:
        return ep
    if use_multi:
        return ep  # (T,M,2)
    tid = int(target_id)
    if tid >= ep.shape[1]:
        tid = 0
    return ep[:, tid, :]

# ============================================================
# 3) Predictor (WORLD)
# ============================================================

def predict_h_step_cv(y_t: np.ndarray, y_tm1: np.ndarray, h: int, box: float) -> np.ndarray:
    v = y_t - y_tm1
    y_hat = np.array(y_t, dtype=float)
    for _ in range(h):
        y_hat = y_hat + v
        y_hat = reflect_to_box(y_hat, box)
    return y_hat

# ============================================================
# 4) Agent trajectory (WORLD)
# ============================================================

def build_agent_traj(T: int, source: str, box: float, obst_ref: np.ndarray | None = None):
    if source == "fixed":
        return np.tile(np.asarray(AGENT_FIXED_XY, dtype=np.float32), (T,1))
    if source == "copy_obst" and obst_ref is not None:
        return obst_ref.copy()
    if source == "random_walk":
        rng = np.random.default_rng(42)
        xy = np.array(AGENT_FIXED_XY, dtype=np.float32)
        traj = [xy.copy()]
        for _ in range(T-1):
            step = rng.normal(scale=1.0, size=2).astype(np.float32)
            xy = reflect_to_box(xy + step, box)
            traj.append(xy.copy())
        return np.stack(traj, axis=0)
    raise ValueError(f"Unsupported AGENT_SOURCE={source}. Use 'fixed'|'copy_obst'|'random_walk'.")

# ============================================================
# 5) Training corpus: residual fields (EGO)
# ============================================================

def build_training_residuals(
    obst_trajs: np.ndarray,     # (N,T,2) representative single traj per sample
    agent_trajs: np.ndarray,    # (N,T,2) agent in WORLD coords
    full_eps: np.ndarray | None,
    use_multi: bool,
    Xg_rel: np.ndarray, Yg_rel: np.ndarray,
    h: int, box: float
) -> np.ndarray:
    """Residual tensor in EGO frame.
    residual[i, t-1] = d_pred^ego(x; y_hat_{t+h|t}) - d_true^ego(x; y_{t+h})
    anchor for both terms is agent at time t.
    """
    N, T, _ = obst_trajs.shape
    H, W = Xg_rel.shape
    T_res = T - h - 1
    if T_res <= 0:
        raise ValueError(f"h={h} too large for T={T}")
    residuals = np.zeros((N, T_res, H, W), dtype=np.float32)

    for i in range(N):
        single = obst_trajs[i]           # (T,2)
        agent  = agent_trajs[i]          # (T,2)
        multi  = full_eps[i] if (use_multi and full_eps is not None) else None  # (T,M,2) or None

        for t in range(1, T - h):
            y_tm1, y_t, y_tph = single[t-1], single[t], single[t+h]
            y_hat = predict_h_step_cv(y_t, y_tm1, h, box)

            # anchor at agent(t)
            a_t = agent[t]
            y_hat_rel = world_to(y_hat, a_t, align_heading=EGO_ALIGN_HEADING)

            if use_multi and (multi is not None) and (multi.ndim == 3):
                y_true_rel = world_to(multi[t+h], a_t, align_heading=EGO_ALIGN_HEADING)
                F_true = distance_field_points(y_true_rel, Xg_rel, Yg_rel)
            else:
                y_true_rel = world_to(y_tph, a_t, align_heading=EGO_ALIGN_HEADING)
                F_true = distance_field_points(y_true_rel, Xg_rel, Yg_rel)

            F_pred = distance_field_points(y_hat_rel, Xg_rel, Yg_rel)
            residuals[i, t-1] = F_pred - F_true
    return residuals

# ============================================================
# 6) Functional CP (PCA + GMM) on residuals (unchanged)
# ============================================================

def cp_upper_for_index(
    residuals_train: np.ndarray, t_idx: int,
    p_base: int, K: int, alpha: float, test_size: float, random_state: int
) -> np.ndarray:
    N, _, H, W = residuals_train.shape
    D = H * W
    Yw = residuals_train[:, t_idx].reshape(N, D).astype(np.float32)

    p_eff = int(min(p_base, N, D))
    if p_eff <= 0:
        return np.zeros((H, W), dtype=np.float32)

    pca = PCA(n_components=p_eff, svd_solver="randomized", random_state=random_state)
    scores = pca.fit_transform(Yw)
    A = pca.components_.astype(np.float64)     # (p_eff, D)
    mean_vec = pca.mean_.astype(np.float64)    # (D,)

    if N >= 4:
        Xi_train, Xi_cal = train_test_split(scores, test_size=test_size, random_state=random_state)
        if len(Xi_train) < K + 1:
            Xi_train, Xi_cal = scores, scores
    else:
        Xi_train, Xi_cal = scores, scores

    K_eff = min(K, max(1, len(Xi_train)))
    gmm = GaussianMixture(n_components=K_eff, covariance_type="full", random_state=random_state).fit(Xi_train)

    logf_cal = gmm.score_samples(Xi_cal)
    q_log = np.quantile(logf_cal, 1.0 - alpha)
    lam = float(np.exp(q_log))

    pis = gmm.weights_
    mus = gmm.means_
    Sigmas = gmm.covariances_

    def _rk_sq(k: int) -> float:
        tau = lam / (K_eff * pis[k])
        Sk = Sigmas[k]
        _det = det(Sk)
        if _det <= 0 or not np.isfinite(_det):
            Sk = Sk + 1e-8 * np.eye(Sk.shape[0])
            _det = det(Sk)
        val = tau * ((2.0 * np.pi) ** (p_eff / 2.0)) * math.sqrt(max(_det, 0.0))
        if (not np.isfinite(val)) or val <= 0.0:
            return 0.0
        return max(0.0, -2.0 * math.log(val))

    rks = np.array([math.sqrt(_rk_sq(k)) for k in range(K_eff)], dtype=np.float64)

    AT = A.T
    upper = np.full((K_eff, D), np.nan, dtype=np.float64)
    for k in range(K_eff):
        mu_k = mus[k]
        Sk = Sigmas[k]
        rk = rks[k]
        center = AT @ mu_k
        AS = AT @ Sk
        quad_diag = np.einsum("dp,dp->d", AS, AT)
        quad_diag = np.clip(quad_diag, 0.0, np.inf)
        rad = rk * np.sqrt(quad_diag)
        upper[k] = center + rad

    g_upper_vec = mean_vec + np.nanmax(upper, axis=0)
    return g_upper_vec.reshape(H, W).astype(np.float32)


def compute_cp_upper_envelopes(
    residuals_train: np.ndarray,
    p_base: int, K: int, alpha: float, test_size: float,
    random_state: int, n_jobs: int, backend: str
) -> np.ndarray:
    _, T_res, H, W = residuals_train.shape
    outs = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(cp_upper_for_index)(
            residuals_train, t_idx, p_base, K, alpha, test_size, random_state
        )
        for t_idx in range(T_res)
    )
    return np.asarray(outs, dtype=np.float32).reshape(T_res, H, W)

# ============================================================
# 7) Build CP-based lower-bound fields for a held-out episode (EGO)
# ============================================================

def build_lower_forecast_fields(
    obst_traj_test: np.ndarray, agent_traj_test: np.ndarray,
    Xg_rel: np.ndarray, Yg_rel: np.ndarray,
    g_upper_all_t: np.ndarray, h: int, box: float
) -> np.ndarray:
    T = obst_traj_test.shape[0]
    H, W = Xg_rel.shape
    lower = np.zeros((T, H, W), dtype=np.float32)

    for t in range(1, T - h):
        y_tm1, y_t = obst_traj_test[t-1], obst_traj_test[t]
        y_hat = predict_h_step_cv(y_t, y_tm1, h, box)

        a_t = agent_traj_test[t]
        y_hat_rel = world_to(y_hat, a_t, align_heading=EGO_ALIGN_HEADING)

        d_pred = distance_field_points(y_hat_rel, Xg_rel, Yg_rel)
        g_up = g_upper_all_t[t - 1]
        lower[t + h] = np.maximum(d_pred - g_up, 0.0)
    return lower

# ============================================================
# 8) Coverage evaluation (EGO)
# ============================================================

def evaluate_coverage(
    seeds: np.ndarray,
    test_pool_obst: np.ndarray,   # (E_test, T, 2)
    test_pool_agent: np.ndarray,  # (E_test, T, 2)
    Xg_rel: np.ndarray, Yg_rel: np.ndarray,
    g_upper_all_t: np.ndarray, h: int,
    safe_threshold: float, box: float
):
    results = []
    rng = np.random.default_rng(1234)
    E_test = test_pool_obst.shape[0]

    for _seed in seeds:
        ep_idx = int(rng.integers(E_test))
        obst = test_pool_obst[ep_idx]
        agent = test_pool_agent[ep_idx]

        misses, totals = [], []
        for t in range(1, obst.shape[0] - h):
            y_tm1, y_t = obst[t-1], obst[t]
            y_hat = predict_h_step_cv(y_t, y_tm1, h, box)

            a_t = agent[t]
            y_true_rel = world_to(obst[t+h], a_t, align_heading=EGO_ALIGN_HEADING)
            y_hat_rel  = world_to(y_hat,      a_t, align_heading=EGO_ALIGN_HEADING)

            F_true = distance_field_points(y_true_rel, Xg_rel, Yg_rel)
            d_pred = distance_field_points(y_hat_rel,  Xg_rel, Yg_rel)
            g_up = g_upper_all_t[t - 1]

            mask_true = (F_true < safe_threshold)
            mask_pred = (d_pred < safe_threshold + g_up)

            missed = np.sum(mask_true & ~mask_pred)
            total  = np.sum(mask_true)
            misses.append(missed); totals.append(total)

        misses = np.array(misses, dtype=float)
        totals = np.array(totals, dtype=float) + 1e-8
        coverage = 1.0 - (misses / totals)
        results.append((coverage.mean(), misses.mean()))

    covs, miss_means = zip(*results)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(miss_means)), float(np.std(miss_means))

# ============================================================
# 9) Visualization (EGO)
# ============================================================

def animate_cp_comparison(
    agent_traj: np.ndarray,            # (T,2) robot in WORLD
    obst_traj:  np.ndarray,            # (T,2) obstacle in WORLD
    true_fields_ego: np.ndarray,       # (T,hH,hW) TRUE distance field on EGO grid at time t
    lower_fields_ego: np.ndarray,      # (T,hH,hW) CP lower-bound field on EGO grid at time t
    box: float,                        # WORLD domain [0,box]^2 (for axes limits)
    ego_box: float,                    # EGO window size [-L/2,L/2]^2 placed at robot pose
    safe_threshold: float,
    h: int,
    headings: np.ndarray = None,       # (T,) yaw [rad]; if None, no rotation
    interval: int = 150
):
    """
    Visualize ego-centric TRUE/CP rasters by painting them into the WORLD axes.
    Frame = t (decision time), we compare/display at tph = t + h.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = np.array(axs)

    T = min(agent_traj.shape[0], obst_traj.shape[0],
            true_fields_ego.shape[0], lower_fields_ego.shape[0])
    if T <= h + 1:
        raise ValueError(f"Not enough time steps for horizon h={h}. Got T={T}.")

    world_extent = [0, box, 0, box]
    cmap_warn = plt.cm.colors.ListedColormap(['white', 'red'])

    def draw_ego_raster(ax, raster_ego, x, y, heading_rad=0.0, alpha=1.0,
                        cmap=None, vmin=None, vmax=None):
        L = float(ego_box)
        ego_extent = [-L/2, L/2, -L/2, L/2]  # ego coords
        im = ax.imshow(raster_ego, extent=ego_extent, origin="lower",
                       cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        trans = Affine2D().rotate(heading_rad).translate(x, y) + ax.transData
        im.set_transform(trans)
        im.set_clip_on(True)
        return im

    def update(t):
        if t < 1 or t >= T - h:
            return
        for ax in axs.flat:
            ax.cla()

        tph = t + h
        x, y = agent_traj[tph]
        yaw = float(headings[tph]) if headings is not None else 0.0

        # ---------- (1,1) TRUE distance field (EGO) placed in WORLD ----------
        F_true_ego = true_fields_ego[tph]
        draw_ego_raster(axs[0, 0], F_true_ego, x, y, heading_rad=yaw, alpha=1.0)
        axs[0, 0].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[0, 0].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[0, 0].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[0, 0].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 0].set_title(f"TRUE distance (ego → world) at t+h (t={t} → {tph})")
        axs[0, 0].set_xlim([0, box]); axs[0, 0].set_ylim([0, box]); axs[0, 0].set_aspect('equal')

        # ---------- (1,2) CP lower field (EGO) placed in WORLD ----------
        lower_ego = lower_fields_ego[tph]
        draw_ego_raster(axs[0, 1], lower_ego, x, y, heading_rad=yaw, alpha=1.0)
        axs[0, 1].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[0, 1].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[0, 1].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[0, 1].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 1].set_title(f"CP lower (ego → world) at t+h (t={t} → {tph})")
        axs[0, 1].set_xlim([0, box]); axs[0, 1].set_ylim([0, box]); axs[0, 1].set_aspect('equal')

        # ---------- (2,1) TRUE unsafe mask (EGO) placed in WORLD ----------
        mask_true = (F_true_ego < safe_threshold).astype(float)
        draw_ego_raster(axs[1, 0], mask_true, x, y, heading_rad=yaw,
                        alpha=1.0, cmap=cmap_warn, vmin=0, vmax=1)
        axs[1, 0].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[1, 0].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[1, 0].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[1, 0].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[1, 0].set_title("TRUE unsafe mask (ego → world)")
        axs[1, 0].set_xlim([0, box]); axs[1, 0].set_ylim([0, box]); axs[1, 0].set_aspect('equal')

        # ---------- (2,2) Overlay: CP unsafe (ego) vs TRUE unsafe (ego), both in WORLD ----------
        axs[1, 1].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[1, 1].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[1, 1].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[1, 1].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        # TRUE unsafe (blue)
        draw_ego_raster(axs[1, 1], mask_true, x, y, heading_rad=yaw,
                        alpha=0.5, cmap="Blues", vmin=0, vmax=1)
        # CP unsafe (red)
        mask_forecast = (lower_ego < safe_threshold).astype(float)
        draw_ego_raster(axs[1, 1], mask_forecast, x, y, heading_rad=yaw,
                        alpha=0.5, cmap="Reds", vmin=0, vmax=1)
        axs[1, 1].set_title("Unsafe: CP (red, ego→world) vs TRUE (blue, ego→world)")
        axs[1, 1].set_xlim([0, box]); axs[1, 1].set_ylim([0, box]); axs[1, 1].set_aspect('equal')

    ani = FuncAnimation(fig, update, frames=range(1, T - h), blit=False, interval=interval)
    return ani

# ============================================================
# 10) Main experiment (fully ego-centric)
# ============================================================

def main():
    rng = np.random.default_rng(SEED)

    # Load + normalize + split
    episodes_raw = load_episodes(DATA_PATH)                 # list[(T_i,2)] or ndarray
    episodes = normalize_world(episodes_raw, BOX, SCALING)

    if isinstance(episodes, list):
        E = len(episodes)
        episodes_arr = np.empty(E, dtype=object)
        for i, ep in enumerate(episodes):
            episodes_arr[i] = ep
    else:
        episodes_arr = episodes
        E = episodes_arr.shape[0]

    train_idx, test_idx = split_train_test(E, split_ratio=0.8, seed=SEED)
    train_eps = episodes_arr[train_idx]
    test_eps  = episodes_arr[test_idx] if len(test_idx) > 0 else episodes_arr[train_idx]

    # Ego grid (only grid we use for fields)
    _, _, Xg_rel, Yg_rel = build_grid(EGO_BOX, H, W)

    # Sample N_TRAIN episodes and align to (N_TRAIN, TSTEPS, 2)
    obst_trajs = np.zeros((N_TRAIN, TSTEPS, 2), dtype=np.float32)
    full_eps   = []  # keep original for multi-obstacle truth if needed
    for i in range(N_TRAIN):
        ep = train_eps[rng.integers(len(train_eps))]
        if ep.shape[0] < TSTEPS:
            pad = np.repeat(ep[-1][None, ...], TSTEPS - ep.shape[0], axis=0)
            ep_use = np.concatenate([ep, pad], axis=0)
        else:
            ep_use = ep[:TSTEPS]

        rep = select_obstacle_traj(ep_use, use_multi=False, target_id=TARGET_ID)
        obst_trajs[i] = rep.astype(np.float32)
        full_eps.append(select_obstacle_traj(ep_use, use_multi=USE_MULTI_OBST, target_id=TARGET_ID))
    full_eps = np.array(full_eps, dtype=object if USE_MULTI_OBST else np.float32)

    # Agent trajectories for training set
    agent_trajs = np.zeros_like(obst_trajs)
    for i in range(N_TRAIN):
        agent_trajs[i] = build_agent_traj(TSTEPS, AGENT_SOURCE, BOX, obst_ref=obst_trajs[i])

    # Residual tensor (EGO)
    residuals_train = build_training_residuals(
        obst_trajs=obst_trajs,
        agent_trajs=agent_trajs,
        full_eps=full_eps if USE_MULTI_OBST else None,
        use_multi=USE_MULTI_OBST,
        Xg_rel=Xg_rel, Yg_rel=Yg_rel,
        h=TIME_HORIZON, box=BOX
    )
    N, T_res, Hh, Ww = residuals_train.shape
    print(f"[Info] residuals_train (EGO): N={N}, T_res={T_res}, H={Hh}, W={Ww}")

    # CP: g_upper(t) for all residual indices
    g_upper_all_t = compute_cp_upper_envelopes(
        residuals_train=residuals_train,
        p_base=P_BASE, K=K_MIX, alpha=ALPHA, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, n_jobs=N_JOBS, backend=BACKEND
    ).astype(np.float32)
    print(f"[Info] g_upper_all_t: shape={g_upper_all_t.shape}")

    # Held-out test episode (single representative)
    ep_test = test_eps[rng.integers(len(test_eps))]
    if ep_test.shape[0] < TSTEPS:
        pad = np.repeat(ep_test[-1][None, ...], TSTEPS - ep_test.shape[0], axis=0)
        ep_test = np.concatenate([ep_test, pad], axis=0)
    else:
        ep_test = ep_test[:TSTEPS]

    obst_test = select_obstacle_traj(ep_test, use_multi=False, target_id=TARGET_ID).astype(np.float32)
    agent_test = build_agent_traj(TSTEPS, AGENT_SOURCE, BOX, obst_ref=obst_test)

    # Lower-bound fields (EGO)
    lower_forecast = build_lower_forecast_fields(
        obst_traj_test=obst_test, agent_traj_test=agent_test,
        Xg_rel=Xg_rel, Yg_rel=Yg_rel,
        g_upper_all_t=g_upper_all_t, h=TIME_HORIZON, box=BOX
    )

    # Coverage on random test episodes (EGO)
    test_pool_obst = []
    test_pool_agent = []
    for ep in test_eps:
        rep = select_obstacle_traj(ep, use_multi=False, target_id=TARGET_ID)
        if rep.shape[0] < TSTEPS:
            pad = np.repeat(rep[-1][None, :], TSTEPS - rep.shape[0], axis=0)
            rep = np.concatenate([rep, pad], axis=0)
        else:
            rep = rep[:TSTEPS]
        rep = rep.astype(np.float32)
        test_pool_obst.append(rep)
        test_pool_agent.append(build_agent_traj(TSTEPS, AGENT_SOURCE, BOX, obst_ref=rep))

    test_pool_obst = np.asarray(test_pool_obst, dtype=np.float32)
    test_pool_agent = np.asarray(test_pool_agent, dtype=np.float32)

    mean_cov, std_cov, mean_miss, std_miss = evaluate_coverage(
        seeds=EVAL_SEEDS,
        test_pool_obst=test_pool_obst,
        test_pool_agent=test_pool_agent,
        Xg_rel=Xg_rel, Yg_rel=Yg_rel,
        g_upper_all_t=g_upper_all_t, h=TIME_HORIZON,
        safe_threshold=SAFE_THRESHOLD, box=BOX
    )
    print(f"[h={TIME_HORIZON}] Mean coverage (ego): {mean_cov*100:.2f}% ± {std_cov*100:.2f}%")
    print(f"[h={TIME_HORIZON}] Avg missed pixels (ego): {mean_miss:.1f} ± {std_miss:.1f}")

    # Visualization (EGO): build true distance fields w.r.t. agent(t)
    T = obst_test.shape[0]
    true_fields_ego = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        a_t = agent_test[t]
        y_true_rel = world_to(obst_test[t], a_t, align_heading=EGO_ALIGN_HEADING)
        true_fields_ego[t] = distance_field_points(y_true_rel, Xg_rel, Yg_rel)

    ani = animate_cp_comparison(
        agent_traj=agent_test,
        obst_traj=obst_test,
        true_fields_ego=true_fields_ego,      
        lower_fields_ego=lower_forecast,      # <-- ego 필드
        box=BOX,                              # <-- WORLD 축 범위
        ego_box=EGO_BOX,                      # <-- ego 윈도우 크기
        safe_threshold=SAFE_THRESHOLD,
        h=TIME_HORIZON,
        headings=None,                        # 혹은 headings=robot_heading_rad
        interval=150
    )
    plt.show()


if __name__ == "__main__":
    main()
