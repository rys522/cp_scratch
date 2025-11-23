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
from preprocess import *
from joblib import Parallel, delayed

from cp.functional_cp import compute_cp_upper_envelopes
from .utils import build_grid, reflect_to_box, world_to_ego, animate_cp_comparison, distance_field_points

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
P_BASE = 3              # basis for PCA
K_MIX = 4               # GMM components
ALPHA = 0.05            # miscoverage level
TEST_SIZE = 0.30        # calibration size for CP
RANDOM_STATE = 0
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
BACKEND = "loky"

# Agent trajectory options (in WORLD coords, per time t)
AGENT_SOURCE = "random_walk"    # "fixed" | "copy_obst" | "random_walk" | "from_array"
AGENT_FIXED_XY = (BOX/2, BOX/2)
EGO_ALIGN_HEADING = False        # if you have heading, you can rotate the ego frame

# ============================================================
# 1) Utils: grids, transforms, distance fields 
# ============================================================





# ============================================================
# 2) Data loading + normalization
# ============================================================



# ============================================================
# 3) Predictor (World)
# ============================================================

def predict_h_step_cv(y_t: np.ndarray, y_tm1: np.ndarray, h: int, box: float) -> np.ndarray:
    v = y_t - y_tm1
    y_hat = np.array(y_t, dtype=float)
    for _ in range(h):
        y_hat = y_hat + v
        y_hat = reflect_to_box(y_hat, box)
    return y_hat

# ============================================================
# 4) Agent trajectory 
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
#   Training corpus: residual fields 
#     f_h(x) = d(x, Y_{t+h}) - d(x, Ỹ_{t+h|t})
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
            y_hat_rel = world_to_ego(y_hat, a_t, align_heading=EGO_ALIGN_HEADING)

            if use_multi and (multi is not None) and (multi.ndim == 3):
                y_true_rel = world_to_ego(multi[t+h], a_t, align_heading=EGO_ALIGN_HEADING)
                F_true = distance_field_points(y_true_rel, Xg_rel, Yg_rel)
            else:
                y_true_rel = world_to_ego(y_tph, a_t, align_heading=EGO_ALIGN_HEADING)
                F_true = distance_field_points(y_true_rel, Xg_rel, Yg_rel)

            F_pred = distance_field_points(y_hat_rel, Xg_rel, Yg_rel)
            residuals[i, t-1] = F_pred - F_true
    return residuals



# ============================================================
# Build CP-based lower-bound fields for a held-out episode (EGO)
# ============================================================

def build_lower_forecast_fields(
    obst_traj_test: np.ndarray, agent_traj_test: np.ndarray,
    Xg_rel: np.ndarray, Yg_rel: np.ndarray,
    g_upper_all_t: np.ndarray, h: int, box: float
) -> np.ndarray:
    """
    Build the *lower forecast distance fields* (conservative safety maps).

    ---------------------------------------------------------------------
        ℓ(x, i, t) = d^ego(x, ŷ_{t+h|t}) − d^ego(x, y_{t+h})
        P( ℓ(x, i, t) ≤ u(1−α) ) ≥ 1−α                (Conformal bound)

    Hence, with probability ≥ (1−α):

        d^ego(x, y_{t+h}) ≥ r_safe
        ⇐ d^ego(x, ŷ_{t+h|t}) ≥ r_safe + u(1−α)

    Implementation equivalence:
        lower[t+h](x) = max( d_pred(x) − g_upper(x), 0 )

    →  d_pred : predicted distance field from ego to forecasted obstacle
       g_upper : spatial uncertainty upper bound (e.g., conformal quantile)
       lower   : conservative distance field ensuring safety at level (1−α)
    ---------------------------------------------------------------------
    Output:
        lower[t] ∈ [0, ∞) — ego-centric conservative distance field over time.
    """
    T = obst_traj_test.shape[0]
    H, W = Xg_rel.shape
    lower = np.zeros((T, H, W), dtype=np.float32)

    for t in range(1, T - h):
        y_tm1, y_t = obst_traj_test[t-1], obst_traj_test[t]
        y_hat = predict_h_step_cv(y_t, y_tm1, h, box)

        a_t = agent_traj_test[t]
        y_hat_rel = world_to_ego(y_hat, a_t, align_heading=EGO_ALIGN_HEADING)

        d_pred = distance_field_points(y_hat_rel, Xg_rel, Yg_rel) # d_pred^ego
        g_up = g_upper_all_t[t - 1]
        lower[t + h] = np.maximum(d_pred - g_up, 0.0)
    return lower

# ============================================================
# Coverage evaluation (EGO)
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
            y_true_rel = world_to_ego(obst[t+h], a_t, align_heading=EGO_ALIGN_HEADING)
            y_hat_rel  = world_to_ego(y_hat,      a_t, align_heading=EGO_ALIGN_HEADING)

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


def main():
    rng = np.random.default_rng(SEED)
# ============================================================
#  offline training on training episodes
# ============================================================

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

    # Residual tensor 
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

# ============================================================
#   online evaluation on test episodes
# ============================================================

    # Held-out test episode (single representative)
    ep_test = test_eps[rng.integers(len(test_eps))]
    if ep_test.shape[0] < TSTEPS:
        pad = np.repeat(ep_test[-1][None, ...], TSTEPS - ep_test.shape[0], axis=0)
        ep_test = np.concatenate([ep_test, pad], axis=0)
    else:
        ep_test = ep_test[:TSTEPS]

    obst_test = select_obstacle_traj(ep_test, use_multi=False, target_id=TARGET_ID).astype(np.float32)
    agent_test = build_agent_traj(TSTEPS, AGENT_SOURCE, BOX, obst_ref=obst_test)

    # Lower-bound fields 
    lower_forecast = build_lower_forecast_fields(
        obst_traj_test=obst_test, agent_traj_test=agent_test,
        Xg_rel=Xg_rel, Yg_rel=Yg_rel,
        g_upper_all_t=g_upper_all_t, h=TIME_HORIZON, box=BOX
    )

# ============================================================
#   Coverage evaluation + Visualization
# ============================================================

    # Coverage on random test episodes 
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
    print(f"[h={TIME_HORIZON}] Mean coverage : {mean_cov*100:.2f}% ± {std_cov*100:.2f}%")
    print(f"[h={TIME_HORIZON}] Avg missed pixels : {mean_miss:.1f} ± {std_miss:.1f}")

    # Visualization (EGO): build true distance fields w.r.t. agent(t)
    T = obst_test.shape[0]
    true_fields_ego = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        a_t = agent_test[t]
        y_true_rel = world_to_ego(obst_test[t], a_t, align_heading=EGO_ALIGN_HEADING)
        true_fields_ego[t] = distance_field_points(y_true_rel, Xg_rel, Yg_rel)

    ani = animate_cp_comparison(
        agent_traj=agent_test,
        obst_traj=obst_test,
        true_fields_ego=true_fields_ego,      
        lower_fields_ego=lower_forecast,      
        box=BOX,                              
        ego_box=EGO_BOX,                      
        safe_threshold=SAFE_THRESHOLD,
        h=TIME_HORIZON,
        headings=None,                        
        interval=150
    )
    plt.show()


if __name__ == "__main__":
    main()