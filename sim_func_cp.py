from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local module imports
from preprocess import load_eth_pickle_preprocessed
from cp.functional_cp import compute_cp_upper_envelopes
from controller import CPFieldMPCController
from utils import build_grid, reflect_vectorized, distance_field_points

# ==============================================================================
# 1. Configuration & Global Constants
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "univ"

H = W = 128
BOX = 100.0
WORLD_CENTER = np.array([BOX / 2.0, BOX / 2.0], dtype=np.float32)

TSTEPS = 20
TIME_HORIZON = 20
MPC_DT = 0.1

SEED = 2021
N_TRAIN = 1000
NUM_PEDS = 20
P_BASE = 3
K_MIX = 5
ALPHA = 0.05  # 95% coverage guarantee
TEST_SIZE = 0.30
RANDOM_STATE = 0
N_JOBS = max(1, (os.cpu_count() or 4) - 2)
BACKEND = "loky"

N_SKIP = 4
N_PATHS = 1200
ROBOT_RAD = 0.4
OBSTACLE_RAD = 1.0 / np.sqrt(2.0)
HARD_CP_CONSTRAINT = True
W_CP_VIOLATION = 200.0

GOAL_REACH_THRESHOLD = 2.0 

# ==============================================================================
# 2. Physics & Utility Functions
# ==============================================================================

def reflect_to_box_xy(xy: np.ndarray, box: float) -> np.ndarray:
    xy = xy.astype(np.float32, copy=True)
    for i in range(2):
        while xy[i] < 0 or xy[i] > box:
            if xy[i] < 0: xy[i] = -xy[i]
            if xy[i] > box: xy[i] = 2 * box - xy[i]
    return xy

def unicycle_step(xy: np.ndarray, th: float, v: float, w: float, dt: float):
    x = xy[0] + dt * v * np.cos(th)
    y = xy[1] + dt * v * np.sin(th)
    th = th + dt * w
    return np.array([x, y], dtype=np.float32), float(th)

def align_mask_dims(mask, target_T, target_M):
    if mask.ndim == 3 and mask.shape[1] == target_M and mask.shape[2] == target_T:
        return np.transpose(mask, (0, 2, 1))
    if mask.ndim == 3 and mask.shape[1] == target_T and mask.shape[2] == target_M:
        return mask
    if mask.ndim == 2 and mask.shape[1] == target_M:
        return np.repeat(mask[:, None, :], target_T, axis=1)
    if mask.ndim == 2 and mask.shape[1] == target_T:
        return np.repeat(mask[:, :, None], target_M, axis=2)
    raise ValueError(f"Unsupported mask shape: {mask.shape}")

def build_cv_predictions(trajs_true, box, horizon):
    trajs_pred = trajs_true.copy().astype(np.float32)
    N, T = trajs_true.shape[:2]
    if T <= horizon + 1: return trajs_pred
    for t in range(horizon + 1, T):
        base = t - horizon
        v = trajs_true[:, base] - trajs_true[:, base - 1]
        curr = trajs_true[:, base].copy()
        for _ in range(horizon):
            curr = reflect_vectorized(curr + v, box)
        trajs_pred[:, t] = curr
    return trajs_pred

def build_training_residuals_valid_only(obst_true, obst_pred, masks, Xg, Yg, horizon):
    if obst_true.ndim == 3:
        obst_true, obst_pred, masks = obst_true[:,:,None,:], obst_pred[:,:,None,:], masks[:,:,None]
    N, T, M, _ = obst_true.shape
    t_offset = horizon + 1
    T_eff = T - t_offset
    residuals = []
    for n in range(N):
        for m in range(M):
            if not np.all(masks[n, t_offset:, m]): continue
            res = np.zeros((T_eff, Xg.shape[0], Xg.shape[1]), dtype=np.float32)
            for t in range(T_eff):
                y_true = obst_true[n, t + t_offset, m] - WORLD_CENTER
                y_pred = obst_pred[n, t + t_offset, m] - WORLD_CENTER
                res[t] = (distance_field_points(y_pred, Xg, Yg) - distance_field_points(y_true, Xg, Yg))
            residuals.append(res)
    return np.stack(residuals, axis=0), t_offset

def make_cv_prediction_from_two_points(y_tm1, y_t, n_steps, box):
    v = (y_t - y_tm1).astype(np.float32)
    pred = np.zeros((n_steps, y_t.shape[0], 2), dtype=np.float32)
    curr = y_t.copy().astype(np.float32)
    for t in range(n_steps):
        curr = reflect_vectorized(curr + v, box)
        pred[t] = curr
    return pred

def compute_lower_field_single_step(obst_pred_step, obst_mask_step, Xg, Yg, g_up, world_center):
    lower = np.full(Xg.shape, np.inf, dtype=np.float32)
    for m in range(obst_pred_step.shape[0]):
        if not bool(obst_mask_step[m]): continue
        y_hat = obst_pred_step[m] - world_center
        d_pred = distance_field_points(y_hat, Xg, Yg)
        lower_m = np.maximum(d_pred - g_up, 0.0)
        lower = np.minimum(lower, lower_m)
    return lower if np.isfinite(lower).any() else np.full(Xg.shape, 20.0, dtype=np.float32)

def min_dist_robot_to_peds(robot_xy, peds_xy):
    if peds_xy.size == 0: return float("inf")
    d = peds_xy - robot_xy[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())

# ==============================================================================
# 3. Main Simulation & Animation
# ==============================================================================

def run_eth_cp_mpc_visual():
    rng = np.random.default_rng(SEED)

    # --- Data Loading ---
    train_data = load_eth_pickle_preprocessed(dataset=DATASET, box=BOX, T=TSTEPS, split_ratio=0.8, seed=SEED, base_dir=BASE_DIR, num_peds=NUM_PEDS)
    train_true_raw, _, train_mask, test_true_raw, _, test_mask = train_data
    
    train_true = np.transpose(train_true_raw, (0, 2, 1, 3))
    test_true = np.transpose(test_true_raw, (0, 2, 1, 3))
    train_mask = align_mask_dims(train_mask, TSTEPS, NUM_PEDS)
    test_mask = align_mask_dims(test_mask, TSTEPS, NUM_PEDS)

    xs, ys, Xg, Yg = build_grid(BOX, H, W)

    # --- Calibration ---
    print("Calibrating CP Envelopes...")
    idx = rng.choice(train_true.shape[0], size=min(N_TRAIN, train_true.shape[0]), replace=False)
    train_pred_sel = build_cv_predictions(train_true[idx], BOX, horizon=1)
    residuals, _ = build_training_residuals_valid_only(train_true[idx], train_pred_sel, train_mask[idx], Xg, Yg, horizon=1)

    g_upper_ds = compute_cp_upper_envelopes(residuals_train=residuals, p_base=P_BASE, K=K_MIX, alpha=ALPHA, 
                                            test_size=TEST_SIZE, random_state=RANDOM_STATE, n_jobs=N_JOBS, backend=BACKEND)

    if g_upper_ds.shape[0] >= TIME_HORIZON:
        g_upper = g_upper_ds[:TIME_HORIZON]
    else:
        pad = np.repeat(g_upper_ds[-1][None], TIME_HORIZON - g_upper_ds.shape[0], axis=0)
        g_upper = np.concatenate([g_upper_ds, pad], axis=0)
    g_upper = g_upper.astype(np.float32)

    # --- Initialization ---
    ep = int(rng.integers(test_true.shape[0]))
    obst_ep, mask_ep = test_true[ep], test_mask[ep]

    max_tracking_error = 0.05 
    planned_next_xy = None

    ctrl = CPFieldMPCController(
        box=BOX, world_center=WORLD_CENTER, g_upper=g_upper, grid_H=H, grid_W=W,
        n_steps=TIME_HORIZON, dt=MPC_DT, n_skip=N_SKIP, n_paths=N_PATHS, seed=0,
        robot_rad=ROBOT_RAD, obstacle_rad=OBSTACLE_RAD + max_tracking_error,
        hard_cp_constraint=HARD_CP_CONSTRAINT, w_cp_violation=W_CP_VIOLATION,
    )

    robot_xy, robot_th = np.array([10.0, 10.0], dtype=np.float32), 0.0
    goal = np.array([90.0, 90.0], dtype=np.float32)
    robot_traj = [robot_xy.copy()]

    # --- Visualization ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    extent = [xs[0] + WORLD_CENTER[0], xs[-1] + WORLD_CENTER[0], ys[0] + WORLD_CENTER[1], ys[-1] + WORLD_CENTER[1]]

    im_heat = axs[0].imshow(np.zeros((H, W)), extent=extent, origin="lower", cmap="viridis_r", vmin=0, vmax=10)
    plt.colorbar(im_heat, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("CP Lower Bound Distance Field")

    axs[1].set_title(f"Safety (Alpha={ALPHA}): True (Blue) vs CP (Red)")
    im_true = axs[1].imshow(np.zeros((H, W)), extent=extent, origin="lower", cmap="Blues", vmin=0, vmax=1, alpha=0.4)
    im_cp = axs[1].imshow(np.zeros((H, W)), extent=extent, origin="lower", cmap="Reds", vmin=0, vmax=1, alpha=0.5)

    robot_dots, traj_lines, plan_lines, peds_scatters = [], [], [], []
    for ax in axs:
        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_aspect("equal")
        ax.plot(goal[0], goal[1], "gx", ms=12, mew=2, label="Goal")
        dot, = ax.plot([], [], "bo", ms=7, zorder=5); line, = ax.plot([], [], "b-", lw=1, alpha=0.5)
        plan_p, = ax.plot([], [], "c--", lw=1.5, zorder=4); peds = ax.scatter([], [], c="red", s=25, edgecolors="black", zorder=6)
        robot_dots.append(dot); traj_lines.append(line); plan_lines.append(plan_p); peds_scatters.append(peds)

    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10, fontweight="bold")
    collision_count = 0
    infeasible_count = 0

    def update(k):
        nonlocal robot_xy, robot_th, collision_count, infeasible_count, max_tracking_error, planned_next_xy
        t = k + 1
        
        # 1. Update Tracking Error & Tighten Constraints
        if planned_next_xy is not None:
            curr_err = np.linalg.norm(robot_xy - planned_next_xy)
            if curr_err > max_tracking_error:
                max_tracking_error = curr_err
                ctrl.obstacle_rad = OBSTACLE_RAD + max_tracking_error

        current_total_safe_thresh = ROBOT_RAD + OBSTACLE_RAD + max_tracking_error

        # Goal Check
        dist_to_goal = np.linalg.norm(robot_xy - goal)
        if dist_to_goal < GOAL_REACH_THRESHOLD:
            status_text.set_text(f"GOAL REACHED! Distance: {dist_to_goal:.2f} | Infeasible: {infeasible_count}")
            status_text.set_color("green")
            anim.event_source.stop(); return []

        # 2. Perception & Field Computation
        p_t, vis = obst_ep[t], mask_ep[t].astype(bool)
        pred = make_cv_prediction_from_two_points(obst_ep[t-1], p_t, TIME_HORIZON, BOX)
        obst_mask = np.repeat(vis[None], TIME_HORIZON, axis=0)

        lower_field = compute_lower_field_single_step(pred[0], obst_mask[0], Xg, Yg, g_upper[0], WORLD_CENTER)
        y_true_rel = p_t[vis] - WORLD_CENTER
        true_dist = distance_field_points(y_true_rel, Xg, Yg) if y_true_rel.size > 0 else np.full((H,W), 20.0)
        
        im_true.set_data((true_dist < current_total_safe_thresh).astype(float))
        im_cp.set_data((lower_field < current_total_safe_thresh).astype(float))

        # 3. MPC Step & Infeasibility Logic
        act, info = ctrl(pos_x=float(robot_xy[0]), pos_y=float(robot_xy[1]), orientation_z=float(robot_th),
                         boxes=None, obst_pred_traj=pred, obst_mask=obst_mask, goal=goal)
        
        feasible = info.get("feasible", False)
        if not feasible:
            infeasible_count += 1
            v, w = 0.0, 0.0 # Emergency stop or hold on infeasibility
        else:
            v, w = float(act[0]), float(act[1])
        
        plan = info.get("final_path")
        planned_next_xy = plan[1].copy() if (plan is not None and len(plan) > 1) else robot_xy.copy()

        # 4. Physics & Collisions
        robot_xy, robot_th = unicycle_step(robot_xy, robot_th, v, w, MPC_DT)
        robot_xy = reflect_to_box_xy(robot_xy, BOX)
        robot_traj.append(robot_xy.copy())

        dmin = min_dist_robot_to_peds(robot_xy, p_t[vis])
        is_coll = (dmin < (ROBOT_RAD + OBSTACLE_RAD))
        if is_coll: collision_count += 1

        # 5. Visualization
        im_heat.set_data(lower_field)
        for i in range(2):
            robot_dots[i].set_data([robot_xy[0]], [robot_xy[1]])
            robot_dots[i].set_color("red" if is_coll else "blue")
            curr_traj = np.stack(robot_traj)
            traj_lines[i].set_data(curr_traj[:, 0], curr_traj[:, 1])
            if plan is not None: plan_lines[i].set_data(plan[:, 0], plan[:, 1])
            peds_scatters[i].set_offsets(p_t[vis] if p_t[vis].size else np.zeros((0, 2)))

        status_text.set_text(
            f"Step: {t} | Collisions: {collision_count} | Infeasible: {infeasible_count} | "
            f"Max Tracking Err: {max_tracking_error:.3f} | Safe Threshold: {current_total_safe_thresh:.2f}"
        )
        status_text.set_color("red" if (is_coll or not feasible) else "black")
        return []

    anim = FuncAnimation(fig, update, frames=obst_ep.shape[0] - 1, interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    run_eth_cp_mpc_visual()