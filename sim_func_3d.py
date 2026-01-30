from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import time

# ---- 3D env/utils ----
from quad_env import QuadWorldEnv3D
from quad_env import (
    build_grid_3d,
    distance_field_points_3d,
)

from cp.functional_cp import get_envelopes_value_and_function
from controllers.func_3d_mpc import FunctionalCPMPC3D

DT = 0.4
ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2

MAX_LINEAR_VEL = 3.0 
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7

MAX_VZ = 0.7

def build_training_residuals_from_env_3d(env, n_samples, X, Y, Z, time_horizon, episode_len=60, group_by_episode=True,v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL), 
    
    yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z), vz_lim=(-MAX_VZ, MAX_VZ)):
    residuals = []
    groups = []
    
    ep_idx = 0
    obs = env.reset()
    
    while len(residuals) < n_samples:
        p_dict = obs.get("prediction", {})
        f_dict = obs.get("future", {})
        
        pids = [pid for pid in p_dict.keys() if pid in f_dict]
        
        if len(pids) > 0:
            res_t = np.zeros((time_horizon, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
            valid = True
            
            for h in range(time_horizon):
                pts_pred, pts_true = [], []
                for pid in pids:
                    pr = p_dict[pid]
                    gt = f_dict[pid]
                    if h < len(pr) and h < len(gt):
                        pts_pred.append(pr[h])
                        pts_true.append(gt[h])
                
                if not pts_pred:
                    valid = False
                    break
                    
                pts_pred = np.array(pts_pred)
                pts_true = np.array(pts_true)
                
                sdf_pred = distance_field_points_3d(pts_pred, X, Y, Z)
                sdf_true = distance_field_points_3d(pts_true, X, Y, Z)
                res_t[h] = sdf_pred - sdf_true # Residual: Pred - True
            
            if valid:
                residuals.append(res_t)
                groups.append(ep_idx if group_by_episode else (len(residuals)-1))

        # 2. Random Step

        robot_xyz = obs["robot_xyz"]

        vx = np.random.uniform(v_lim[0], v_lim[1])
        vy = np.random.uniform(v_lim[0], v_lim[1])    
        w = np.random.uniform(yaw_rate_lim[0], yaw_rate_lim[1])
        vz = np.random.uniform(vz_lim[0], vz_lim[1])

        target_pos = robot_xyz + np.array([vx, vy, vz], dtype=np.float32) * float(env.dt)
        target_pos = np.array([
            np.clip(target_pos[0], env.xlim[0], env.xlim[1]),
            np.clip(target_pos[1], env.ylim[0], env.ylim[1]),
            np.clip(target_pos[2], env.zlim[0], env.zlim[1]),
        ], dtype=np.float32)

        cmd = (vx, vy, vz, w)     # w = yaw_rate
        obs = env.step(target_pos, cmd)

        
        if env.step_counter % episode_len == 0:
            ep_idx += 1
            obs = env.reset()
            
    return np.stack(residuals), np.array(groups)


def stack_pred3d_from_p_dict(p_dict: Dict[int, np.ndarray], horizon: int):
    pids = list(p_dict.keys())
    M = len(pids)
    Hh = int(horizon)

    pred = np.zeros((Hh, M, 3), dtype=np.float32)
    mask = np.zeros((Hh, M), dtype=bool)

    for j, pid in enumerate(pids):
        arr = np.asarray(p_dict[pid], dtype=np.float32)
        take = min(Hh, arr.shape[0])
        if take > 0:
            pred[:take, j, :] = arr[:take, :]
            mask[:take, j] = True
    return pred, mask, pids

def _deepcopy_obs_dict(obs: Dict) -> Dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, dict):
            out[k] = _deepcopy_obs_dict(v)
        elif isinstance(v, list):
            out[k] = [vv.copy() if isinstance(vv, np.ndarray) else vv for vv in v]
        else:
            out[k] = v
    return out


# ----------------------------
# Rendering Helpers
# ----------------------------
def _remove_poly3d(ax, poly: Optional[Poly3DCollection]) -> None:
    if poly is None: return
    try: poly.remove()
    except Exception: pass
    try: 
        if poly in ax.collections: ax.collections.remove(poly)
    except Exception: pass

def _clear_contours(cont_list: List) -> None:
    """cont_list holds QuadContourSet; remove robustly."""
    for cs in cont_list:
        if cs is None:
            continue
        try:
            cs.remove()
            continue
        except Exception:
            pass
        try:
            for coll in getattr(cs, "collections", []):
                try:
                    coll.remove()
                except Exception:
                    pass
        except Exception:
            pass
    cont_list.clear()

def _draw_isosurface_from_bool(ax, vol_bool_zyx, xs, ys, zs, *, alpha, facecolor):
    if vol_bool_zyx is None or vol_bool_zyx.ndim != 3 or not np.any(vol_bool_zyx):
        return None
    
    vol = vol_bool_zyx.astype(np.float32)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    except Exception:
        return None

    vz_idx, vy_idx, vx_idx = verts[:, 0], verts[:, 1], verts[:, 2]
    vx = np.interp(vx_idx, np.arange(xs.size), xs)
    vy = np.interp(vy_idx, np.arange(ys.size), ys)
    vz = np.interp(vz_idx, np.arange(zs.size), zs)
    V = np.stack([vx, vy, vz], axis=1).astype(np.float32)

    mesh = Poly3DCollection(V[faces], alpha=alpha)
    mesh.set_facecolor((*facecolor, alpha))
    mesh.set_edgecolor((0,0,0,0))
    ax.add_collection3d(mesh)
    return mesh

def _set_equal_aspect_3d(ax, xlim, ylim, zlim):
    x0, x1 = xlim; y0, y1 = ylim; z0, z1 = zlim
    cx, cy, cz = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    r = max(x1-x0, y1-y0, z1-z0)/2
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def _get_obs_positions_from_history(obs_dict) -> np.ndarray:
    """obs['history']에서 현재(가장 최근) 장애물 위치들을 추출"""
    h_dict = obs_dict.get("history", {})
    if not h_dict:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)

def _min_dist_robot_to_points(robot_xyz: np.ndarray, pts_xyz: np.ndarray) -> float:

    if pts_xyz.size == 0:
        return float("inf")
    d = np.linalg.norm(pts_xyz - robot_xyz[None, :], axis=1)
    return float(np.min(d))


def run_one_episode_visual_3d(
    env: QuadWorldEnv3D,
    *,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    time_horizon: int = 12,
    alpha: float = 0.10,
    p_base: int = 4,
    k_mix: int = 5,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    n_paths: int = 600,
    max_steps: int = 2500, 
    i_view: int = 3, # 미래 몇 스텝을 시각화할지 (0 ~ time_horizon-1)
    n_calib_samples: int = 120,
    goal_finish_dist: float = 0.8,
    mc_stride: int = 2,
    view_elev: float = 22.0, 
    view_azim: float = -55.0,
    dpi: int = 140,
    CP: bool = False,
    save: bool = False
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    # ----------------------------
    # 1) Grid & CP Setup
    # ----------------------------
    xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
    xs, ys, zs, X, Y, Z = build_grid_3d(xlim, ylim, zlim, nx, ny, nz)
    Xxy, Yxy = np.meshgrid(xs, ys, indexing="xy")
    Xxz, Zxz = np.meshgrid(xs, zs, indexing="xy")
    Yyz, Zyz = np.meshgrid(ys, zs, indexing="xy")

    margin = 5.0 # 벽 두께
    b_min = 0.0
    b_max = 20.0
    
    # 전체 공간을 넉넉하게 커버하는 범위 (-50 ~ 50)
    cov_min, cov_max = -50.0, 50.0

    x_min, x_max = xlim
    y_min, y_max = ylim
    z_min, z_max = zlim

    wall_boxes = [
        # 1. X-axis walls (Left / Right)
        [x_min - margin, x_min, cov_min, cov_max, cov_min, cov_max],  # X < x_min
        [x_max, x_max + margin, cov_min, cov_max, cov_min, cov_max],  # X > x_max

        # 2. Y-axis walls (Front / Back)
        [cov_min, cov_max, y_min - margin, y_min, cov_min, cov_max],  # Y < y_min
        [cov_min, cov_max, y_max, y_max + margin, cov_min, cov_max],  # Y > y_max

        # 3. Z-axis walls (Floor / Ceiling)
        [cov_min, cov_max, cov_min, cov_max, z_min - margin, z_min],  # Z < z_min
        [cov_min, cov_max, cov_min, cov_max, z_max, z_max + margin],  # Z > z_max
    ]

    residuals, _ = build_training_residuals_from_env_3d(
        env, n_samples=n_calib_samples, X=X, Y=Y, Z=Z,
        time_horizon=time_horizon, episode_len=1000, group_by_episode=False, 
        v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL), 
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z), vz_lim=(-MAX_VZ, MAX_VZ)
    )
    
    g_upper_grid, cp_params = get_envelopes_value_and_function(
        residuals_train=residuals, p_base=p_base, K=k_mix, alpha=alpha,
        test_size=test_size, random_state=random_state, n_jobs=n_jobs, backend=backend
    )
    g_upper_grid = g_upper_grid.astype(np.float32)

    # ----------------------------
    # 2) Controller Setup
    # ----------------------------
    ctrl = FunctionalCPMPC3D(
        cp_params=cp_params, xs=xs, ys=ys, zs=zs, n_steps=time_horizon,
        dt=env.dt, n_skip=n_skip, robot_rad=ROBOT_RAD, obstacle_rad=OBSTACLE_RAD,
        v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL), 
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        vz_lim=(-MAX_VZ, MAX_VZ),
        n_paths=n_paths, seed=0, CP=CP,
    )

    # ----------------------------
    # 3) Record Phase (Run Simulation First)
    # ----------------------------
    print(f"Running simulation for max {max_steps} steps to record Ground Truth...")
    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)
    
    episode_history = []
    timing = {
        "ctrl_ms": [],
        "step_ms": [],
        "loop_ms": [],
    }
    vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0


    for k in range(max_steps):
        t_loop0 = time.perf_counter()

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])

        # Check Goal
        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            print(f"Goal reached at step {k}")
            break

        # Prediction for Controller
        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)

        # ---- controller timing ----
        t0 = time.perf_counter()
        act, info = ctrl(
            robot_xyz=robot, robot_yaw=yaw, goal_xyz=goal,
            pred_xyz=pred, pred_mask=pred_mask, boxes_3d=wall_boxes,
            robot_vel=(vx_global, vy_global, vz_global)
        )
        t1 = time.perf_counter()

        if act is None:
            # Emergency fallback (should be rare)
            target = robot.copy()
            vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0
            act_to_store = None
        else:
            pos, vel = act
            target = np.asarray(pos, dtype=np.float32).reshape(3,)
            vel = np.asarray(vel, dtype=np.float32).reshape(4,)
            vx_global, vy_global, vz_global, yaw_rate = map(float, vel)
            act_to_store = (target.copy(), vel.copy())

        cmd = (vx_global, vy_global, vz_global, yaw_rate)

        is_feasible = bool(info.get("feasible", False))

        # ---- env.step timing ----
        t2 = time.perf_counter()
        obs = env.step(target, cmd)
        t3 = time.perf_counter()

        ctrl_ms = (t1 - t0) * 1000.0
        step_ms = (t3 - t2) * 1000.0
        loop_ms = (t3 - t_loop0) * 1000.0

        timing["ctrl_ms"].append(ctrl_ms)
        timing["step_ms"].append(step_ms)
        timing["loop_ms"].append(loop_ms)

        episode_history.append({
            "step": k,
            "obs": _deepcopy_obs_dict(obs),
            "robot": robot,
            "yaw": yaw,
            "act": act_to_store,          # None or (pos, vel)
            "feasible": bool(is_feasible),
            "pred": pred,
            "pred_mask": pred_mask,
            "timing": {"ctrl_ms": ctrl_ms, "step_ms": step_ms, "loop_ms": loop_ms},
        })

    total_frames = len(episode_history)
    def _summ(x: List[float], name: str):
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            print(f"[timing] {name}: no data")
            return
        p50, p90, p99 = np.percentile(arr, [50, 90, 99])
        print(
            f"[timing] {name}: mean={arr.mean():.3f} ms | "
            f"p50={p50:.3f} | p90={p90:.3f} | p99={p99:.3f} | max={arr.max():.3f}"
        )

    print("\n==== Online compute timing (ms) ====")
    _summ(timing["ctrl_ms"], "controller (FunctionalCPMPC3D)")
    _summ(timing["step_ms"], "env.step (physics)")
    _summ(timing["loop_ms"], "total loop")
    print("===================================\n")
    print(f"Simulation finished. Total frames: {total_frames}. Starting Visualization...")

    # ----------------------------
    # 4) Visualization Phase (Replay)  [UI MATCHED to 2nd script]
    # ----------------------------
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, 2]); ax_txt.axis("off")

    # ---- axes formatting (same as 2nd script) ----
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=view_elev, azim=view_azim)
    _set_equal_aspect_3d(ax3d, xlim, ylim, zlim)
    ax3d.grid(False)

    ax3d.scatter([goal[0]], [goal[1]], [goal[2]],
                 marker="*", s=140, edgecolors="k", linewidths=0.4)

    for ax in (ax_xy, ax_xz, ax_yz):
        ax.grid(True, alpha=0.20, linewidth=0.6)
        ax.tick_params(labelsize=9)

    ax_xy.set_title("XY (min over z): boundary ϕ=0", fontsize=10)
    ax_xy.set_xlim(xlim); ax_xy.set_ylim(ylim)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.plot([goal[0]], [goal[1]], marker="*", markersize=10, markeredgewidth=0.6)

    ax_xz.set_title("XZ (min over y): boundary ϕ=0", fontsize=10)
    ax_xz.set_xlim(xlim); ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.plot([goal[0]], [goal[2]], marker="*", markersize=10, markeredgewidth=0.6)

    ax_yz.set_title("YZ (min over x): boundary ϕ=0", fontsize=10)
    ax_yz.set_xlim(ylim); ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")
    ax_yz.plot([goal[1]], [goal[2]], marker="*", markersize=10, markeredgewidth=0.6)

    # ---- artists (same as 2nd script) ----
    robot_dot, = ax3d.plot(
        [], [], [],
        marker="o",
        markersize=10,         
        linewidth=0,
        markerfacecolor=(1.0, 0.85, 0.0),
        markeredgecolor="k",        
        markeredgewidth=1.2,
        zorder=10
    )

    robot_line, = ax3d.plot(
        [], [], [],
        linewidth=3.0,        
        alpha=0.95,
        color=(1.0, 0.85, 0.0),     
        zorder=9
    )

    obs_sc  = ax3d.scatter([], [], [], s=18, alpha=0.80)              # ● obst(now)
    pred_sc = ax3d.scatter([], [], [], s=32, marker="x", alpha=0.95)  # × pred(i_view)
    gt_sc   = ax3d.scatter([], [], [], s=30, marker="D", alpha=0.90)  # ◆ GT(i_view)

    xy_robot_dot, = ax_xy.plot(
        [], [], "o",
        ms=8,
        mfc=(1.0, 0.85, 0.0),
        mec="k",
        mew=1.0,
        zorder=10
    )
    xy_robot_line, = ax_xy.plot(
        [], [], "-",
        lw=2.8,
        alpha=0.95,
        color=(1.0, 0.85, 0.0),
        zorder=9
    )
    xy_obs_sc  = ax_xy.scatter([], [], s=14, alpha=0.80)
    xy_pred_sc = ax_xy.scatter([], [], s=22, marker="x", alpha=0.95)
    xy_gt_sc   = ax_xy.scatter([], [], s=20, marker="D", alpha=0.90)

    xz_robot_dot, = ax_xz.plot(
        [], [], "o",
        ms=8,
        mfc=(1.0, 0.85, 0.0),
        mec="k",
        mew=1.0,
        zorder=10
    )
    xz_robot_line, = ax_xz.plot(
        [], [], "-",
        lw=2.8,
        alpha=0.95,
        color=(1.0, 0.85, 0.0),
        zorder=9
    )
    xz_obs_sc  = ax_xz.scatter([], [], s=14, alpha=0.80)
    xz_pred_sc = ax_xz.scatter([], [], s=22, marker="x", alpha=0.95)
    xz_gt_sc   = ax_xz.scatter([], [], s=20, marker="D", alpha=0.90)

    yz_robot_dot, = ax_yz.plot(
        [], [], "o",
        ms=8,
        mfc=(1.0, 0.85, 0.0),
        mec="k",
        mew=1.0,
        zorder=10
    )
    yz_robot_line, = ax_yz.plot(
        [], [], "-",
        lw=2.8,
        alpha=0.95,
        color=(1.0, 0.85, 0.0),
        zorder=9
    )
    yz_obs_sc  = ax_yz.scatter([], [], s=14, alpha=0.80)
    yz_pred_sc = ax_yz.scatter([], [], s=22, marker="x", alpha=0.95)
    yz_gt_sc   = ax_yz.scatter([], [], s=20, marker="D", alpha=0.90)

    # ---- containers ----
    true_mesh: Optional[Poly3DCollection] = None
    cp_mesh: Optional[Poly3DCollection] = None
    xy_contours: List = []
    xz_contours: List = []
    yz_contours: List = []

    # replay trajectory buffers
    robot_traj = []

    # stats (same style)
    collision_count = 0
    contain_ok_steps = 0
    contain_eval_steps = 0
    voxel_recall_sum = 0.0
    infeasible_count = 0

    def upd(k: int):
        nonlocal true_mesh, cp_mesh
        nonlocal collision_count, contain_ok_steps, contain_eval_steps, voxel_recall_sum, infeasible_count

        if k >= total_frames:
            return []

        data = episode_history[k]
        obs_k = data["obs"]
        robot = np.asarray(data["robot"], dtype=np.float32).reshape(3,)
        yaw = float(data["yaw"])
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj, dtype=np.float32)

        # goal distance (for text only)
        dist_goal = float(np.linalg.norm(robot - goal))

        # current obstacles (now)
        h_dict = obs_k.get("history", {})
        obs_now = (
            np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)
            if len(h_dict) else np.zeros((0, 3), dtype=np.float32)
        )

        # prediction (i_view)
        pred_full = np.asarray(data["pred"], dtype=np.float32)          # (H,M,3)
        pred_mask_full = np.asarray(data["pred_mask"], dtype=bool)      # (H,M)

        i_view_clamped = int(np.clip(i_view, 0, pred_full.shape[0] - 1)) if pred_full.size else 0
        pred_i = pred_full[i_view_clamped]
        mask_i = pred_mask_full[i_view_clamped]
        pred_i_vis = pred_i[mask_i] if pred_i.size else np.zeros((0, 3), dtype=np.float32)

        # GT future (replay): k + (i_view+1)
        future_idx = k + (i_view_clamped + 1)
        gt_i_vis = np.zeros((0, 3), dtype=np.float32)
        if future_idx < total_frames:
            future_obs = episode_history[future_idx]["obs"]
            gt_i_vis = _get_obs_positions_from_history(future_obs)

        # collision now
        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            collision_count += 1
        is_feasible = bool(data["feasible"])
        if not is_feasible:
            infeasible_count += 1

        # distance fields
        D_pred = (
            distance_field_points_3d(pred_i_vis, X, Y, Z)
            if pred_i_vis.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
        )

        U_i = g_upper_grid[i_view_clamped] if CP else 0.0
        U_i = np.maximum(U_i, 0.0)
        D_lower = np.maximum(D_pred - U_i, 0.0)
        cp_unsafe = (D_lower < safe_rad)

        D_true = None
        true_unsafe = None
        if gt_i_vis.size:
            D_true = distance_field_points_3d(gt_i_vis, X, Y, Z)
            true_unsafe = (D_true < safe_rad)

        # containment metrics
        containment_ok = None
        voxel_recall = None
        if true_unsafe is not None:
            containment_ok = (not np.any(true_unsafe & (~cp_unsafe)))
            contain_eval_steps += 1
            contain_ok_steps += int(containment_ok)

            denom = float(np.sum(true_unsafe))
            if denom > 0.0:
                voxel_recall = float(np.sum(true_unsafe & cp_unsafe)) / denom
                voxel_recall_sum += voxel_recall
            else:
                voxel_recall = 1.0

        cov_running = (contain_ok_steps / max(1, contain_eval_steps))
        mean_recall = (voxel_recall_sum / max(1, contain_eval_steps)) if contain_eval_steps > 0 else 0.0

        # ---- update 3D artists ----
        robot_dot.set_data([robot[0]], [robot[1]])
        robot_dot.set_3d_properties([robot[2]])
        robot_line.set_data(tr[:, 0], tr[:, 1])
        robot_line.set_3d_properties(tr[:, 2])

        obs_sc._offsets3d = (obs_now[:, 0], obs_now[:, 1], obs_now[:, 2]) if obs_now.size else ([], [], [])
        pred_sc._offsets3d = (pred_i_vis[:, 0], pred_i_vis[:, 1], pred_i_vis[:, 2]) if pred_i_vis.size else ([], [], [])
        gt_sc._offsets3d = (gt_i_vis[:, 0], gt_i_vis[:, 1], gt_i_vis[:, 2]) if gt_i_vis.size else ([], [], [])

        # ---- update 2D ----
        xy_robot_dot.set_data([robot[0]], [robot[1]])
        xy_robot_line.set_data(tr[:, 0], tr[:, 1])
        xz_robot_dot.set_data([robot[0]], [robot[2]])
        xz_robot_line.set_data(tr[:, 0], tr[:, 2])
        yz_robot_dot.set_data([robot[1]], [robot[2]])
        yz_robot_line.set_data(tr[:, 1], tr[:, 2])

        xy_obs_sc.set_offsets(obs_now[:, [0, 1]] if obs_now.size else np.zeros((0, 2)))
        xz_obs_sc.set_offsets(obs_now[:, [0, 2]] if obs_now.size else np.zeros((0, 2)))
        yz_obs_sc.set_offsets(obs_now[:, [1, 2]] if obs_now.size else np.zeros((0, 2)))

        xy_pred_sc.set_offsets(pred_i_vis[:, [0, 1]] if pred_i_vis.size else np.zeros((0, 2)))
        xz_pred_sc.set_offsets(pred_i_vis[:, [0, 2]] if pred_i_vis.size else np.zeros((0, 2)))
        yz_pred_sc.set_offsets(pred_i_vis[:, [1, 2]] if pred_i_vis.size else np.zeros((0, 2)))

        xy_gt_sc.set_offsets(gt_i_vis[:, [0, 1]] if gt_i_vis.size else np.zeros((0, 2)))
        xz_gt_sc.set_offsets(gt_i_vis[:, [0, 2]] if gt_i_vis.size else np.zeros((0, 2)))
        yz_gt_sc.set_offsets(gt_i_vis[:, [1, 2]] if gt_i_vis.size else np.zeros((0, 2)))

        # ---- update iso-surfaces ----
        _remove_poly3d(ax3d, true_mesh)
        _remove_poly3d(ax3d, cp_mesh)
        true_mesh = None
        cp_mesh = None

        s = int(max(1, mc_stride))
        cp_ds = cp_unsafe[::s, ::s, ::s]
        xs_ds = xs[np.arange(0, xs.size, s)]
        ys_ds = ys[np.arange(0, ys.size, s)]
        zs_ds = zs[np.arange(0, zs.size, s)]

        cp_mesh = _draw_isosurface_from_bool(
            ax3d, cp_ds, xs_ds, ys_ds, zs_ds,
            alpha=0.16, facecolor=(0.85, 0.20, 0.20)
        )
        if true_unsafe is not None:
            true_ds = true_unsafe[::s, ::s, ::s]
            true_mesh = _draw_isosurface_from_bool(
                ax3d, true_ds, xs_ds, ys_ds, zs_ds,
                alpha=0.22, facecolor=(0.20, 0.35, 0.95)
            )

        # ---- update contours ----
        _clear_contours(xy_contours)
        _clear_contours(xz_contours)
        _clear_contours(yz_contours)

        phi_cp = D_lower - safe_rad
        phi_true = (D_true - safe_rad) if D_true is not None else None

        phi_cp_xy = np.min(phi_cp, axis=0)  # (ny,nx)
        phi_cp_xz = np.min(phi_cp, axis=1)  # (nz,nx)
        phi_cp_yz = np.min(phi_cp, axis=2)  # (nz,ny)

        xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_cp_xy, levels=[0.0], linewidths=1.8))
        xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_cp_xz, levels=[0.0], linewidths=1.8))
        yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_cp_yz, levels=[0.0], linewidths=1.8))

        if phi_true is not None:
            phi_true_xy = np.min(phi_true, axis=0)
            phi_true_xz = np.min(phi_true, axis=1)
            phi_true_yz = np.min(phi_true, axis=2)
            xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_true_xy, levels=[0.0], linewidths=1.8, linestyles="--"))
            xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_true_xz, levels=[0.0], linewidths=1.8, linestyles="--"))
            yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_true_yz, levels=[0.0], linewidths=1.8, linestyles="--"))

        # ---- titles + text (same style) ----
        ok_str = "-" if containment_ok is None else ("OK" if containment_ok else "FAIL")
        ax3d.set_title(
            f"3D unsafe iso-surfaces | step={k} | i_view={i_view_clamped} | feasible={data['feasible']} | contain={ok_str}",
            fontsize=11
        )

        act = data.get("act", None)

        if act is None:
            pos_cmd = obs_k["robot_xyz"]
            vel_cmd = np.zeros((4,), dtype=np.float32)
        else:
            pos_cmd, vel_cmd = act
            pos_cmd = np.asarray(pos_cmd, dtype=np.float32).reshape(3,)
            vel_cmd = np.asarray(vel_cmd, dtype=np.float32).reshape(4,)
        robot_pos = obs_k["robot_xyz"]
        robot_vel = obs_k["robot_vel"]


        ax_txt.clear(); 
        ax_txt.axis("off")
        ax_txt.text(
            0.02, 0.98,
            "\n".join([
                f"step = {k}/{total_frames-1}",
                f"i_view = {i_view_clamped}",
                f"dist_goal = {dist_goal:.3f} (finish<= {goal_finish_dist})",
                "",
                f"feasible = {bool(data['feasible'])}",
                f"pos_cmd = (x={pos_cmd[0]:.3f}, y={pos_cmd[1]:.3f}, z={pos_cmd[2]:.3f})",
                f"vel_cmd = (vx={vel_cmd[0]:.3f}, vy={vel_cmd[1]:.3f}, vz={vel_cmd[2]:.3f}, yaw={vel_cmd[3]:.3f})",
                f"robot_pos = (x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, z={robot_pos[2]:.3f})",
                f"robot_vel = (vx={robot_vel[0]:.3f}, vy={robot_vel[1]:.3f}, vz={robot_vel[2]:.3f}",
                "",
                f"collisions = {collision_count}  (dmin_now={dmin_now:.3f}, r_safe={safe_rad:.3f})",
                f"infeasible = {infeasible_count}",
                "",
                f"containment-coverage = {cov_running:.2%}   (eval_steps={contain_eval_steps})",
                f"mean voxel-recall    = {mean_recall:.2%}",
                "",
                "Legend:",
                "  ● obst(now)",
                "  × pred(i_view)",
                "  ◆ GT(i_view)",
                "  solid contour : CP boundary",
                "  dashed contour: GT boundary",
            ]),
            va="top", ha="left", fontsize=10
        )

        return []

    anim = FuncAnimation(fig, upd, frames=total_frames, interval=80, blit=False, repeat=False)
    plt.tight_layout()

    if save:
        if CP:
            save_path = "quad_cp_mpc_3d.mp4"
        else:
            save_path = "quad_noncp_mpc_3d.mp4"

        anim.save(
            save_path,
            writer="ffmpeg",
            fps=12,            
            dpi=240,
            bitrate=20000
        )
    plt.show()

if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=0.1,
        horizon=20,
        n_obs=350,
        world_bounds_xyz=((-3, 7), (-3, 7), (0.0, 8.0)),
        seed=2,

        pred_model_noise=0.20,

        obs_process_noise=0.22,
        gt_future_noise=0.20,

        mode_switch_p=0.95,
        mode_min_ttl=1,
        mode_max_ttl=6,
        turn_rate_std=3.0,
        stop_go_p=0.6,
        gui = False
    )

    run_one_episode_visual_3d(
        env,
        nx=40, ny=40, nz=40,
        time_horizon=12,
        alpha=0.10,
        p_base=8,
        k_mix=10,
        test_size=0.30,
        random_state=0,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        backend="loky",
        n_skip=4,
        n_paths=2000,
        max_steps=250,
        i_view=3,
        n_calib_samples=120,
        goal_finish_dist=0.3,
        mc_stride=2,
        view_elev=22.0,
        view_azim=-55.0,
        dpi=140,
        CP = False,
        save = False,
    )