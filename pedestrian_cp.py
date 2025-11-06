# pedestrian_cp_single.py
from __future__ import annotations
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import det
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import pickle

# ============================================================
# 0) Global parameters (tweak here)
# ============================================================

# Data
DATA_PATH = "./data/seq_eth"  # expects key 'episodes' with shape (E,T,2) or (E,T,M,2)
USE_MULTI_OBST = False                # if True, distance field uses min-dist to all agents at each time
TARGET_ID = 0                         # used when USE_MULTI_OBST=False and episodes are (E,T,M,2)

# Grid / domain
H, W = 64, 64
BOX = 100.0                           # normalize world coords into [0, BOX]^2
SCALING = None                        # (sx, sy) world->meters; if None, computed from data bounds

# Corpus / horizon / seeds
TSTEPS = 120
N_TRAIN = 1000                        # #episodes sampled w/ replacement for training corpus
TIME_HORIZON = 3
SEED = 2023
EVAL_SEEDS = np.arange(SEED + 100, SEED + 110)  # coverage evaluation

# Safety threshold (distance units == meters in normalized space)
SAFE_THRESHOLD = 8.0

# CP hyperparams
P_BASE = 3
K_MIX = 4
ALPHA = 0.05
TEST_SIZE = 0.30
RANDOM_STATE = 0
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
BACKEND = "loky"

# ============================================================
# 1) Utils: grid + distance fields
# ============================================================

def build_grid(box: float, H: int, W: int):
    xs = np.linspace(0, box, W)
    ys = np.linspace(0, box, H)
    Xg, Yg = np.meshgrid(xs, ys)
    return xs, ys, Xg, Yg

def distance_field_single_obstacle(point: np.ndarray, Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """Euclidean distance from each grid cell to one obstacle."""
    px, py = point
    d2 = (Xg - px) ** 2 + (Yg - py) ** 2
    return np.sqrt(d2).astype(np.float32)

def distance_field_multi_obstacles(points: np.ndarray, Xg: np.ndarray, Yg: np.ndarray) -> np.ndarray:
    """Min distance over multiple obstacles at a single time step.
       points: (M,2) or (2,) – if (2,), it degenerates to single obstacle.
    """
    if points.ndim == 1:
        return distance_field_single_obstacle(points, Xg, Yg)
    dmin = None
    for p in points:
        d = distance_field_single_obstacle(p, Xg, Yg)
        dmin = d if dmin is None else np.minimum(dmin, d)
    return dmin.astype(np.float32)

# ============================================================
# 2) Data loading + normalization
# ============================================================

def _apply_homography_if_available(xy: np.ndarray, H_path: str | None) -> np.ndarray:
    """Apply projective transform if H.txt exists (pixel->world)."""
    if not H_path or not os.path.isfile(H_path):
        return xy
    try:
        H = np.loadtxt(H_path)
        if H.shape != (3, 3):
            return xy
        ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
        homo = np.concatenate([xy, ones], axis=1)          # (N,3)
        trans = (H @ homo.T).T                              # (N,3)
        w = np.clip(trans[:, 2], 1e-8, None)
        xy_w = trans[:, :2] / w[:, None]
        return xy_w.astype(np.float32)
    except Exception:
        return xy

def _load_eth_folder(folder: str) -> list[np.ndarray]:
    """
    Read ETH/UCY-style folder:
      - obsmat.txt  (frame, id, x, y, ... or sometimes frame, id, px, py, ...)
      - H.txt       (optional homography to meters)
    Returns: list of episodes, each is (T_i, 2) in world coords.
    """
    obsmat_path = os.path.join(folder, "obsmat.txt")
    if not os.path.isfile(obsmat_path):
        raise FileNotFoundError(f"obsmat.txt not found in {folder}")

    # load plain whitespace table
    M = np.loadtxt(obsmat_path)
    if M.ndim != 2 or M.shape[1] < 4:
        raise ValueError("obsmat.txt must be a 2D table with >= 4 columns")

    # columns: 0=frame, 1=id, and pick plausible (x,y) columns
    frame = M[:, 0].astype(int)
    pid   = M[:, 1].astype(int)

    # heuristic: try (2,3) then (3,2) then (4,5)
    cand_pairs = [(2, 3), (3, 2), (4, 5)]
    xy = None
    for cx, cy in cand_pairs:
        if max(cx, cy) < M.shape[1]:
            tmp = M[:, [cx, cy]].astype(np.float32)
            if np.isfinite(tmp).all() and (tmp.std(axis=0) > 1e-6).all():
                xy = tmp
                break
    if xy is None:
        # fallback: just take next two columns after id
        xy = M[:, [2, 3]].astype(np.float32)

    # optional homography (pixel->meters)
    H_path = os.path.join(folder, "H.txt")
    xy = _apply_homography_if_available(xy, H_path)

    # group by person id -> trajectory sorted by frame
    episodes: list[np.ndarray] = []
    for p in np.unique(pid):
        sel = pid == p
        fr  = frame[sel]
        pts = xy[sel]
        order = np.argsort(fr)
        traj = pts[order]                     # (T_i, 2)

        # drop degenerate/very short tracks
        if len(traj) >= 5 and np.isfinite(traj).all():
            episodes.append(traj.astype(np.float32))

    if len(episodes) == 0:
        raise ValueError("No valid trajectories extracted from ETH folder.")
    return episodes

def load_episodes(data_path: str) -> np.ndarray | list[np.ndarray]:
    """
    Accepts:
      - Directory with ETH files (obsmat.txt, optional H.txt)
      - .npz with key 'episodes' (E,T,2) or (E,T,M,2)
      - .pkl/.pickle: dict with 'episodes' or direct ndarray/list
    Returns: list of (T_i,2) or ndarray with shape (E,T,2)/(E,T,M,2).
    """
    if os.path.isdir(data_path):
        return _load_eth_folder(data_path)

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npz":
        data = np.load(data_path, allow_pickle=True)
        episodes = data["episodes"]
        return episodes
    elif ext in (".pkl", ".pickle"):
        with open(data_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "episodes" in obj:
            return obj["episodes"]
        return obj
    else:
        raise ValueError(f"Unsupported data path: {data_path}")

def normalize_world(arr, box: float, scaling=None):
    """
    Works for:
      - ndarray (E,T,2) or (E,T,M,2)
      - list of (T_i,2)
    Translate+scale into ~[0,box]^2 using global bounds.
    """
    # gather all points to get bounds
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
    """Episode-level train/test split indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    n_train = int(split_ratio * E)
    return perm[:n_train], perm[n_train:]

def select_obstacle_traj(ep: np.ndarray, use_multi: bool, target_id: int):
    """Return per-episode obstacle trajectory for the CP pipeline.
       - If ep is (T,2): return (T,2).
       - If ep is (T,M,2) and use_multi=True: return (T,M,2) (for min-distance fields).
       - Else: return (T,2) of the target agent.
    """
    if ep.ndim == 2:
        return ep
    if use_multi:
        return ep  # (T,M,2)
    tid = int(target_id)
    if tid >= ep.shape[1]:
        tid = 0  # safe fallback
    return ep[:, tid, :]

# ============================================================
# 3) Predictor: constant-velocity h-step with reflection to box
# ============================================================

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

def predict_h_step_cv(y_t: np.ndarray, y_tm1: np.ndarray, h: int, box: float) -> np.ndarray:
    """h-step constant-velocity prediction with step-wise reflection."""
    v = y_t - y_tm1
    y_hat = np.array(y_t, dtype=float)
    for _ in range(h):
        y_hat = y_hat + v
        y_hat = reflect_to_box(y_hat, box)
    return y_hat

# ============================================================
# 4) Training corpus: residual fields
# ============================================================

def build_training_residuals(
    obst_trajs: np.ndarray,     # (N, T, 2)  (if multi, pass a representative single traj here)
    full_eps: np.ndarray,       # original episodes aligned per sample, for multi-obst mode if needed
    use_multi: bool,
    Xg: np.ndarray, Yg: np.ndarray,
    h: int, box: float
) -> np.ndarray:
    """Residual tensor over all training episodes.
       residual[i, t-1] = d_pred(x; y_hat_{t+h|t}) - d_true(x; y_{t+h})
       valid t in [1 .. T-h-1].
       If use_multi=True and full_eps has (N,T,M,2), we build truth fields via min-distance over M agents.
    """
    N, T, _ = obst_trajs.shape
    H, W = Xg.shape
    T_res = T - h - 1
    if T_res <= 0:
        raise ValueError(f"h={h} too large for T={T}")
    residuals = np.zeros((N, T_res, H, W), dtype=np.float32)

    for i in range(N):
        single = obst_trajs[i]              # (T,2) representative trajectory
        multi  = full_eps[i] if full_eps is not None else None  # (T,M,2) or (T,2)

        for t in range(1, T - h):
            y_tm1, y_t, y_tph = single[t - 1], single[t], single[t + h]
            y_hat = predict_h_step_cv(y_t, y_tm1, h, box)
            F_pred = distance_field_single_obstacle(y_hat, Xg, Yg)

            # build true field
            if use_multi and (multi is not None) and (multi.ndim == 3):
                F_true = distance_field_multi_obstacles(multi[t + h], Xg, Yg)
            else:
                F_true = distance_field_single_obstacle(y_tph, Xg, Yg)
            residuals[i, t - 1] = F_pred - F_true
    return residuals

# ============================================================
# 5) Functional CP (PCA + GMM) on residuals
# ============================================================

def cp_upper_for_index(
    residuals_train: np.ndarray, t_idx: int,
    p_base: int, K: int, alpha: float, test_size: float, random_state: int
) -> np.ndarray:
    """Compute g_upper(x) for a single residual time index t_idx."""
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
        if len(Xi_train) < K + 1:  # tiny split fallback
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

    # per-component radii in score space
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
        mu_k = mus[k]                          # (p_eff,)
        Sk = Sigmas[k]                         # (p_eff,p_eff)
        rk = rks[k]

        center = AT @ mu_k                     # (D,)
        AS = AT @ Sk                           # (D,p_eff)
        quad_diag = np.einsum("dp,dp->d", AS, AT)
        quad_diag = np.clip(quad_diag, 0.0, np.inf)
        rad = rk * np.sqrt(quad_diag)          # (D,)
        upper[k] = center + rad

    g_upper_vec = mean_vec + np.nanmax(upper, axis=0)
    return g_upper_vec.reshape(H, W).astype(np.float32)

def compute_cp_upper_envelopes(
    residuals_train: np.ndarray,
    p_base: int, K: int, alpha: float, test_size: float,
    random_state: int, n_jobs: int, backend: str
) -> np.ndarray:
    """Compute g_upper(x) for all residual indices in parallel.
       Returns (T_res, H, W).
    """
    _, T_res, H, W = residuals_train.shape
    outs = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(cp_upper_for_index)(
            residuals_train, t_idx, p_base, K, alpha, test_size, random_state
        )
        for t_idx in range(T_res)
    )
    return np.asarray(outs, dtype=np.float32).reshape(T_res, H, W)

# ============================================================
# 6) Build CP-based lower-bound fields for a held-out episode
# ============================================================

def build_lower_forecast_fields(
    obst_traj_test: np.ndarray, Xg: np.ndarray, Yg: np.ndarray,
    g_upper_all_t: np.ndarray, h: int, box: float
) -> np.ndarray:
    """lower[t+h] = max( d_pred(y_hat_{t+h|t}) - g_upper_t , 0 )"""
    T = obst_traj_test.shape[0]
    H, W = Xg.shape
    lower = np.zeros((T, H, W), dtype=np.float32)

    for t in range(1, T - h):
        y_tm1, y_t = obst_traj_test[t - 1], obst_traj_test[t]
        y_hat = predict_h_step_cv(y_t, y_tm1, h, box)
        d_pred = distance_field_single_obstacle(y_hat, Xg, Yg)
        g_up = g_upper_all_t[t - 1]
        lower[t + h] = np.maximum(d_pred - g_up, 0.0)
    return lower

# ============================================================
# 7) Coverage evaluation
# ============================================================

def evaluate_coverage(
    seeds: np.ndarray,
    test_pool: np.ndarray,          # (E_test, T, 2)  (representative single traj per episode)
    Xg: np.ndarray, Yg: np.ndarray,
    g_upper_all_t: np.ndarray, h: int, box: float,
    safe_threshold: float
):
    """Mean coverage / missed-unsafe-pixels over random test episodes."""
    results = []
    rng = np.random.default_rng(1234)

    E_test = test_pool.shape[0]
    for _seed in seeds:
        ep_idx = int(rng.integers(E_test))
        obst = test_pool[ep_idx]  # (T,2)

        misses, totals = [], []
        for t in range(1, obst.shape[0] - h):
            y_tm1, y_t = obst[t - 1], obst[t]
            y_hat = predict_h_step_cv(y_t, y_tm1, h, box)

            F_true = distance_field_single_obstacle(obst[t + h], Xg, Yg)
            d_pred = distance_field_single_obstacle(y_hat, Xg, Yg)
            g_up = g_upper_all_t[t - 1]

            mask_true = (F_true < safe_threshold)
            # lower < tau  <=>  d_pred - g_up < tau  <=>  d_pred < tau + g_up
            mask_pred = (d_pred < safe_threshold + g_up)

            missed = np.sum(mask_true & ~mask_pred)
            total = np.sum(mask_true)
            misses.append(missed); totals.append(total)

        misses = np.array(misses, dtype=float)
        totals = np.array(totals, dtype=float) + 1e-8
        coverage = 1.0 - (misses / totals)
        results.append((coverage.mean(), misses.mean()))

    covs, miss_means = zip(*results)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(miss_means)), float(np.std(miss_means))

# ============================================================
# 8) Visualization (optional)
# ============================================================

def animate_cp_comparison(
    robot_traj: np.ndarray, obst_traj: np.ndarray,
    true_fields: np.ndarray, lower_fields: np.ndarray,
    box: float, safe_threshold: float, h: int, interval: int = 150
):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    def update(frame):
        if frame < 1 or frame >= true_fields.shape[0] - h:
            return
        for ax in axs.flat:
            ax.cla()
        tph = frame + h

        F_true = true_fields[tph]
        axs[0, 0].imshow(F_true, extent=[0, box, 0, box], origin="lower")
        axs[0, 0].plot(robot_traj[:tph + 1, 0], robot_traj[:tph + 1, 1], 'b-', lw=1)
        axs[0, 0].plot(obst_traj[:tph + 1, 0],  obst_traj[:tph + 1, 1],  'r-', lw=1)
        axs[0, 0].plot(robot_traj[tph, 0], robot_traj[tph, 1], 'bo', ms=5)
        axs[0, 0].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 0].set_title(f"True distance field at t+h (t={frame}, h={h} → {tph})")
        axs[0, 0].set_xlim([0, box]); axs[0, 0].set_ylim([0, box]); axs[0, 0].set_aspect('equal')

        lower = lower_fields[tph]
        axs[0, 1].imshow(lower, extent=[0, box, 0, box], origin="lower")
        axs[0, 1].plot(robot_traj[:tph + 1, 0], robot_traj[:tph + 1, 1], 'b-', lw=1)
        axs[0, 1].plot(obst_traj[:tph + 1, 0],  obst_traj[:tph + 1, 1],  'r-', lw=1)
        axs[0, 1].plot(robot_traj[tph, 0], robot_traj[tph, 1], 'bo', ms=5)
        axs[0, 1].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 1].set_title(f"Forecasted CP lower bound at t+h (t={frame}, h={h} → {tph})")
        axs[0, 1].set_xlim([0, box]); axs[0, 1].set_ylim([0, box]); axs[0, 1].set_aspect('equal')

        cmap = plt.cm.colors.ListedColormap(['white', 'red'])
        mask_true = (F_true < safe_threshold).astype(float)
        axs[1, 0].imshow(mask_true, extent=[0, box, 0, box], origin="lower", cmap=cmap, vmin=0, vmax=1)
        axs[1, 0].set_title(f"True unsafe mask at t+h (t={frame}, h={h} → {tph})")
        axs[1, 0].set_xlim([0, box]); axs[1, 0].set_ylim([0, box]); axs[1, 0].set_aspect('equal')

        mask_forecast = (lower < safe_threshold).astype(float)
        axs[1, 1].imshow(mask_forecast, extent=[0, box, 0, box], origin="lower",
                         cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        axs[1, 1].imshow(mask_true, extent=[0, box, 0, box], origin="lower",
                         cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        axs[1, 1].set_title(f"Unsafe forecast (red) vs True (blue) at t+h")
        axs[1, 1].set_xlim([0, box]); axs[1, 1].set_ylim([0, box]); axs[1, 1].set_aspect('equal')

    ani = FuncAnimation(fig, update, frames=range(1, true_fields.shape[0] - 1), blit=False, interval=interval)
    return ani

# ============================================================
# 9) Main experiment
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

    # Build grid
    _, _, Xg, Yg = build_grid(BOX, H, W)

    # Sample N_TRAIN episodes (with replacement) and align shapes to (N, T, 2)
    obst_trajs = np.zeros((N_TRAIN, TSTEPS, 2), dtype=np.float32)  # representative single traj
    full_eps   = []  # keep original per-sample (T,2) or (T,M,2) for multi-obstacle residuals
    for i in range(N_TRAIN):
        ep = train_eps[rng.integers(len(train_eps))]
        Tcur = ep.shape[0] if not isinstance(ep, np.ndarray) or ep.ndim==2 else ep.shape[0]
        # pad or crop to TSTEPS
        if ep.shape[0] < TSTEPS:
            pad = np.repeat(ep[-1][None, ...], TSTEPS - ep.shape[0], axis=0)
            ep_use = np.concatenate([ep, pad], axis=0)
        else:
            ep_use = ep[:TSTEPS]

        # representative single trajectory
        rep = select_obstacle_traj(ep_use, use_multi=False, target_id=TARGET_ID)  # (T,2)
        obst_trajs[i] = rep.astype(np.float32)

        # store full for multi-obstacle truth fields if needed
        full_eps.append(select_obstacle_traj(ep_use, use_multi=USE_MULTI_OBST, target_id=TARGET_ID))
    full_eps = np.array(full_eps, dtype=object if USE_MULTI_OBST else np.float32)
    # If USE_MULTI_OBST=True and shapes vary in M, we keep dtype=object; residual builder handles it.

    # Residual tensor (N, T_res, H, W)
    residuals_train = build_training_residuals(
        obst_trajs=obst_trajs,
        full_eps=full_eps if USE_MULTI_OBST else None,
        use_multi=USE_MULTI_OBST,
        Xg=Xg, Yg=Yg,
        h=TIME_HORIZON, box=BOX
    )
    N, T_res, Hh, Ww = residuals_train.shape
    print(f"[Info] residuals_train: N={N}, T_res={T_res}, H={Hh}, W={Ww}")

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

    obst_test = select_obstacle_traj(ep_test, use_multi=False, target_id=TARGET_ID).astype(np.float32)  # (T,2)
    robot_test = obst_test.copy()

    # Lower-bound fields
    lower_forecast = build_lower_forecast_fields(
        obst_traj_test=obst_test, Xg=Xg, Yg=Yg,
        g_upper_all_t=g_upper_all_t, h=TIME_HORIZON, box=BOX
    )

    # Coverage on random test episodes (simple, representative single-agent coverage)
    # If you want multi-obstacle coverage, adapt here with distance_field_multi_obstacles( test_ep[t+h] ).
    test_pool_single = []
    for ep in test_eps:
        rep = select_obstacle_traj(ep, use_multi=False, target_id=TARGET_ID)
        # pad/crop to TSTEPS
        if rep.shape[0] < TSTEPS:
            pad = np.repeat(rep[-1][None, :], TSTEPS - rep.shape[0], axis=0)
            rep = np.concatenate([rep, pad], axis=0)
        else:
            rep = rep[:TSTEPS]
        test_pool_single.append(rep.astype(np.float32))
    test_pool_single = np.asarray(test_pool_single, dtype=np.float32)

    mean_cov, std_cov, mean_miss, std_miss = evaluate_coverage(
        seeds=EVAL_SEEDS,
        test_pool=test_pool_single,
        Xg=Xg, Yg=Yg,
        g_upper_all_t=g_upper_all_t, h=TIME_HORIZON, box=BOX,
        safe_threshold=SAFE_THRESHOLD
    )
    print(f"[h={TIME_HORIZON}] Mean coverage: {mean_cov*100:.2f}% ± {std_cov*100:.2f}%")
    print(f"[h={TIME_HORIZON}] Avg missed pixels: {mean_miss:.1f} ± {std_miss:.1f}")

    # Visualization (optional): true vs CP-lower for the chosen test episode
    # Build true distance fields for visualization (single obstacle)
    T = obst_test.shape[0]
    true_fields = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        true_fields[t] = distance_field_single_obstacle(obst_test[t], Xg, Yg)

    fig, _ = plt.subplots(2, 2, figsize=(12, 10))
    ani = animate_cp_comparison(
        robot_traj=robot_test,
        obst_traj=obst_test,
        true_fields=true_fields,
        lower_fields=lower_forecast,
        box=BOX,
        safe_threshold=SAFE_THRESHOLD,
        h=TIME_HORIZON,
        interval=150
    )
    plt.show()


if __name__ == "__main__":
    main()