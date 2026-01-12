from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import build_grid, distance_field_points
from sims.sim_utils import (
    min_dist_robot_to_peds,
    unicycle_step,
)
from cp.fuctional_spatial_cp import (
    FunctionalCPConfig,
    FunctionalCP,
    FunctionalCPParams,
)

from controllers.func_cp_mpc import FunctionalCPMPC


# ==============================================================================
# 0) Constants
# ==============================================================================

DT = 0.4

ROBOT_RAD = 0.4
OBSTACLE_RAD = 1.0 / np.sqrt(2.0)

MAX_LINEAR_X = 0.8
MIN_LINEAR_X = -0.8
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7

EVAL_TASK_CONFIGS = {
    "zara1": {"init_robot_pose": np.array([12.0, 5.0, np.pi], dtype=np.float32), "goal_pos": np.array([3.0, 6.0], dtype=np.float32)},
    "zara2": {"init_robot_pose": np.array([1.0, 6.0, 0.0], dtype=np.float32), "goal_pos": np.array([14.0, 5.0], dtype=np.float32)},
    "hotel": {"init_robot_pose": np.array([-1.5, 0.0, -np.pi / 2.0], dtype=np.float32), "goal_pos": np.array([2.0, -6.0], dtype=np.float32)},
    "eth":   {"init_robot_pose": np.array([5.0, 1.0, np.pi / 2.0], dtype=np.float32), "goal_pos": np.array([3.0, 10.0], dtype=np.float32)},
    "univ":  {"init_robot_pose": np.array([3.5, 2.0, np.pi / 4.0], dtype=np.float32), "goal_pos": np.array([11.5, 8.5], dtype=np.float32)},
}

SCENARIOS = {
    "zara1": [100, 200, 300],
    "zara2": [100, 200, 300],
    "eth": [100, 200, 300],
    "hotel": [100, 200, 300],
    "univ": [100],
}

INIT_FRAMES = {
    "zara1": 0,
    "zara2": 1,
    "eth": 78,
    "hotel": 0,
    "univ": 0,
}

MAX_N_STEPS = {
    "zara1": 100,
    "zara2": 100,
    "eth": 100,
    "hotel": 100,
    "univ": 300,
}


# ==============================================================================
# 1) PKL loader
# ==============================================================================

def load_prediction_results(dataset: str) -> dict:
    pkl_path = os.path.join(os.path.dirname(__file__), "predictions", f"{dataset}.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    for k in ("prediction", "history", "future"):
        if k not in res:
            raise ValueError(f"PKL is missing key '{k}'")
    return res


# ==============================================================================
# 2) World bounds inference from PKL
# ==============================================================================

def _collect_points_for_bounds(
    pred_all: dict,
    hist_all: dict,
    scenario_begin: int,
    n_steps: int,
    horizon: int,
    init_pose: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    pts = [init_pose[:2][None, :], goal[None, :]]

    for k in range(n_steps):
        ts_key = scenario_begin + k

        if ts_key in hist_all:
            for traj in hist_all[ts_key].values():
                arr = np.asarray(traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    pts.append(arr[-1:])

        if ts_key in pred_all:
            for pred_traj in pred_all[ts_key].values():
                arr = np.asarray(pred_traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                    take = min(horizon, arr.shape[0])
                    pts.append(arr[:take])

    return np.vstack(pts) if len(pts) > 0 else np.zeros((0, 2), dtype=np.float32)


def _infer_world(points_xy: np.ndarray, margin: float = 2.0):
    if points_xy.size == 0:
        world_center = np.array([0.0, 0.0], dtype=np.float32)
        box = 40.0
        bounds = (-20.0, 20.0, -20.0, 20.0)
        return world_center, box, bounds

    xmin = float(np.min(points_xy[:, 0])) - margin
    xmax = float(np.max(points_xy[:, 0])) + margin
    ymin = float(np.min(points_xy[:, 1])) - margin
    ymax = float(np.max(points_xy[:, 1])) + margin

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    xspan = xmax - xmin
    yspan = ymax - ymin
    box = float(max(xspan, yspan))  # square for grid

    world_center = np.array([cx, cy], dtype=np.float32)
    return world_center, box, (xmin, xmax, ymin, ymax)


# ==============================================================================
# 3) Offline calibration from file: build residual tensor (N, H, Hg, Wg)
#    residual = D_pred - D_true  (MIN-over-peds SDF)
# ==============================================================================

def build_training_residuals_from_file(
    all_data_dict: dict,
    scene_ids: List[int],
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    time_horizon: int,
) -> np.ndarray:
    pred_dict = all_data_dict["prediction"]
    fut_dict = all_data_dict["future"]

    Hh = int(time_horizon)
    Hg, Wg = Xg.shape

    residuals: List[np.ndarray] = []

    for sid in scene_ids:
        if sid not in pred_dict or sid not in fut_dict:
            continue

        p_scene = pred_dict[sid]  # {pid: (H,2)}
        f_scene = fut_dict[sid]   # {pid: (H,2)}

        pids = [pid for pid in p_scene.keys() if pid in f_scene]
        if len(pids) == 0:
            continue

        res_i = np.zeros((Hh, Hg, Wg), dtype=np.float32)
        last_valid_i: Optional[int] = None

        for i in range(Hh):
            pred_pts = []
            true_pts = []

            for pid in pids:
                y_pred_traj = np.asarray(p_scene[pid], dtype=np.float32)
                y_true_traj = np.asarray(f_scene[pid], dtype=np.float32)

                if y_pred_traj.ndim != 2 or y_pred_traj.shape[1] != 2:
                    continue
                if y_true_traj.ndim != 2 or y_true_traj.shape[1] != 2:
                    continue
                if i >= y_pred_traj.shape[0] or i >= y_true_traj.shape[0]:
                    continue

                pred_pts.append(y_pred_traj[i])
                true_pts.append(y_true_traj[i])

            if len(pred_pts) == 0 or len(true_pts) == 0:
                if last_valid_i is not None:
                    res_i[i] = res_i[last_valid_i]
                else:
                    res_i[i] = 0.0
                continue

            pred_pts = (np.asarray(pred_pts, dtype=np.float32) - world_center)
            true_pts = (np.asarray(true_pts, dtype=np.float32) - world_center)

            sdf_pred = distance_field_points(pred_pts, Xg, Yg)
            sdf_true = distance_field_points(true_pts, Xg, Yg)

            res_i[i] = sdf_pred - sdf_true
            last_valid_i = i

        residuals.append(res_i)

    if len(residuals) == 0:
        raise RuntimeError("No valid residual samples built from file. (Check PKL contents.)")

    return np.stack(residuals, axis=0)


def calibrate_cp_from_file(
    *,
    all_data: dict,
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    time_horizon: int,
    alpha: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
    backend: str,
    rank: int = 32,
) -> FunctionalCPParams:
    all_scenes = sorted(list(all_data["prediction"].keys()))

    residuals = build_training_residuals_from_file(
        all_data_dict=all_data,
        scene_ids=all_scenes,
        Xg=Xg,
        Yg=Yg,
        world_center=world_center,
        time_horizon=time_horizon,
    )

    print(f"[offline] residual tensor: {residuals.shape} (N_scenes, H, Hg, Wg)")
    print("[offline] calibrating FunctionalCP params...")

    cfg = FunctionalCPConfig(
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        use_shared_basis=True,
        p_base=rank,
    )
    calibrator = FunctionalCP(cfg)
    calibrator.fit(residuals)
    cp_params = calibrator.extract_params(rank=rank)
    return cp_params


def reconstruct_S_grid(params: FunctionalCPParams, t_idx: int) -> np.ndarray:
    """
    S_i(x) = S_mean(x) + coeffs[i] @ phi_basis
    returns (Hg, Wg)
    """
    S_flat = params.S_mean + (params.coeffs[t_idx] @ params.phi_basis)
    return S_flat.reshape(params.grid_shape).astype(np.float32)


# ==============================================================================
# 4) Helpers: controller input from PKL prediction dict
# ==============================================================================

def stack_pred_from_p_dict(p_dict: Dict, horizon: int) -> Tuple[np.ndarray, np.ndarray, List]:
    pids = list(p_dict.keys())
    M = len(pids)
    Hh = int(horizon)

    pred = np.zeros((Hh, M, 2), dtype=np.float32)
    mask = np.zeros((Hh, M), dtype=bool)

    for j, pid in enumerate(pids):
        arr = np.asarray(p_dict[pid], dtype=np.float32)
        take = min(Hh, arr.shape[0])
        if take > 0:
            pred[:take, j] = arr[:take]
            mask[:take, j] = True

    return pred, mask, pids


def get_current_obs_from_history(h_dict: Dict) -> np.ndarray:
    if len(h_dict) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)


def get_future_obs_from_future_dict(f_dict: Dict, i_view: int) -> np.ndarray:
    if len(f_dict) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = []
    for pid, traj in f_dict.items():
        arr = np.asarray(traj, dtype=np.float32)  # (H,2)
        if arr.ndim == 2 and arr.shape[1] == 2 and i_view < arr.shape[0]:
            pts.append(arr[i_view])
    return np.asarray(pts, dtype=np.float32) if len(pts) > 0 else np.zeros((0, 2), dtype=np.float32)


# ==============================================================================
# 5) One-episode visualization (file-based) with AUTO world/grid + FunctionalCP
# ==============================================================================

def run_one_episode_visual_from_file(
    dataset: str,
    *,
    scenario_idx: int = 0,
    time_horizon: int = 12,
    grid_H: int = 128,
    grid_W: int = 128,
    alpha: float = 0.10,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    n_paths: int = 1200,
    max_steps: Optional[int] = None,
):
    if dataset not in EVAL_TASK_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")

    init_robot_pose = EVAL_TASK_CONFIGS[dataset]["init_robot_pose"].copy()
    goal_pos = EVAL_TASK_CONFIGS[dataset]["goal_pos"].copy()
    scenario_begin = SCENARIOS[dataset][scenario_idx]
    _ = INIT_FRAMES[dataset]
    max_n_steps = MAX_N_STEPS[dataset] if max_steps is None else int(max_steps)

    # Load PKL
    all_data = load_prediction_results(dataset)
    pred_all = all_data["prediction"]
    hist_all = all_data["history"]
    fut_all = all_data["future"]

    # --------- AUTO world/grid inference ----------
    points_xy = _collect_points_for_bounds(
        pred_all=pred_all,
        hist_all=hist_all,
        scenario_begin=scenario_begin,
        n_steps=min(max_n_steps, 200),
        horizon=time_horizon,
        init_pose=init_robot_pose,
        goal=goal_pos,
    )
    world_center, box, (xmin, xmax, ymin, ymax) = _infer_world(points_xy, margin=2.0)

    xs, ys, Xg, Yg = build_grid(box, grid_H, grid_W)
    extent = [
        xs[0] + world_center[0], xs[-1] + world_center[0],
        ys[0] + world_center[1], ys[-1] + world_center[1],
    ]

    print(f"[world] inferred bounds: x=[{xmin:.2f},{xmax:.2f}] y=[{ymin:.2f},{ymax:.2f}]")
    print(f"[world] inferred box={box:.2f}, world_center={world_center.tolist()}")

    # --------- Offline calibration (FunctionalCP params) ----------
    cp_params = calibrate_cp_from_file(
        all_data=all_data,
        Xg=Xg,
        Yg=Yg,
        world_center=world_center,
        time_horizon=time_horizon,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        rank=16,
    )

    ctrl = FunctionalCPMPC(
        cp_params=cp_params,
        box=box,
        world_center=world_center,
        grid_H=grid_H,
        grid_W=grid_W,
        n_steps=time_horizon,
        dt=DT,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        min_linear_x=MIN_LINEAR_X,
        max_linear_x=MAX_LINEAR_X,
        min_angular_z=MIN_ANGULAR_Z,
        max_angular_z=MAX_ANGULAR_Z,
        n_paths=n_paths,
        seed=0,
    )

    robot_xy = init_robot_pose[:2].astype(np.float32)
    robot_th = float(init_robot_pose[2])
    goal = goal_pos.astype(np.float32)

    robot_traj = [robot_xy.copy()]
    collision_count = 0
    infeasible_count = 0

    # timestep-wise containment coverage for i_view
    ok_steps = 0
    eval_steps = 0
    current_cov_val = 0.0

    fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    # 왼쪽: lower_field(거리필드)
    im_heat = axs[0].imshow(
        np.zeros((grid_H, grid_W), dtype=np.float32),
        extent=extent,
        origin="lower",
        cmap="viridis_r",
    )
    plt.colorbar(im_heat, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("Lower distance field")

    axs[1].set_title(f"Unsafe set comparison at i_view (alpha={alpha})")
    im_true = axs[1].imshow(
        np.zeros((grid_H, grid_W)),
        extent=extent,
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=1,
        alpha=0.35,
    )
    im_cp = axs[1].imshow(
        np.zeros((grid_H, grid_W)),
        extent=extent,
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
        alpha=0.50,
    )

    robot_dots, traj_lines, plan_lines, obs_scatters, pred_scatters, gt_scatters = [], [], [], [], [], []
    for ax in axs:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.20)
        ax.plot(goal[0], goal[1], "g*", ms=12, mew=0.8)

        dot, = ax.plot([], [], "bo", ms=7, zorder=6)
        line, = ax.plot([], [], "b-", lw=1.2, alpha=0.60, zorder=5)
        plan_p, = ax.plot([], [], "c--", lw=1.5, zorder=4)

        obs_sc = ax.scatter([], [], s=25, edgecolors="black", zorder=7)      # now
        pred_sc = ax.scatter([], [], s=40, marker="x", zorder=8)             # pred i_view
        gt_sc = ax.scatter([], [], s=40, marker="D", zorder=9)               # GT i_view

        robot_dots.append(dot)
        traj_lines.append(line)
        plan_lines.append(plan_p)
        obs_scatters.append(obs_sc)
        pred_scatters.append(pred_sc)
        gt_scatters.append(gt_sc)

    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10, fontweight="bold")

    def update(k: int):
        nonlocal robot_xy, robot_th, collision_count, infeasible_count
        nonlocal eval_steps, ok_steps, current_cov_val

        ts_key = scenario_begin + k
        if ts_key not in pred_all or ts_key not in hist_all or ts_key not in fut_all:
            status_text.set_text(
                f"Ended | step={k} | Cov={current_cov_val:.2%} | infeasible={infeasible_count} | collisions={collision_count}"
            )
            status_text.set_color("red")
            anim.event_source.stop()
            return []

        dist_to_goal = float(np.linalg.norm(robot_xy - goal))
        if dist_to_goal <= 0.6:
            status_text.set_text(
                f"GOAL reached | step={k} | Cov={current_cov_val:.2%} | dist={dist_to_goal:.2f}"
            )
            status_text.set_color("green")
            anim.event_source.stop()
            return []

        p_dict = pred_all[ts_key]   # dict(pid -> (H,2))  for t
        h_dict = hist_all[ts_key]   # dict(pid -> (hist,2))
        f_dict = fut_all[ts_key]    # dict(pid -> (H,2))  GT future

        # current observed peds at time t
        obs_now = get_current_obs_from_history(h_dict)

        # choose i_view (0 => t+1)
        i_view = 7
        i_view = int(np.clip(i_view, 0, time_horizon - 1))

        # GT future at i_view
        gt_i = get_future_obs_from_future_dict(f_dict, i_view)

        # predictions tensor (H,M,2)
        pred, obst_mask, _ = stack_pred_from_p_dict(p_dict, horizon=time_horizon)
        pred_i = pred[i_view]
        mask_i = obst_mask[i_view]
        pred_i_vis = pred_i[mask_i] if pred_i.size else np.zeros((0, 2), dtype=np.float32)

        # -------------------------
        # FunctionalCP lower field:
        # D_pred - U_i
        # -------------------------
        safe_thresh = ROBOT_RAD + OBSTACLE_RAD

        if pred_i_vis.size > 0:
            D_pred = distance_field_points(pred_i_vis - world_center, Xg, Yg)
        else:
            D_pred = np.full((grid_H, grid_W), np.inf, dtype=np.float32)

        # U_i(x) = rho_i * S_i(x)
        S_i = reconstruct_S_grid(cp_params, i_view)  # (Hg,Wg)
        U_i = float(cp_params.rhos[i_view]) * np.maximum(S_i, 1e-9)

        lower_field = np.maximum(D_pred - U_i, 0.0)

        # GT distance field at i_view (for coverage + viz)
        if gt_i.size > 0:
            D_true = distance_field_points(gt_i - world_center, Xg, Yg)
        else:
            D_true = np.full((grid_H, grid_W), np.inf, dtype=np.float32)

        # viz
        im_true.set_data((D_true < safe_thresh).astype(float))
        im_cp.set_data((lower_field < safe_thresh).astype(float))
        im_heat.set_data(lower_field)

        # -------------------------
        # containment coverage @ i_view
        # true_unsafe ⊆ cp_unsafe  (grid-wise)
        # -------------------------
        true_unsafe = (D_true < safe_thresh)
        cp_unsafe = (lower_field < safe_thresh)
        ok_t = (not np.any(true_unsafe & (~cp_unsafe)))

        if k > i_view:
            eval_steps += 1
            ok_steps += int(ok_t)
            current_cov_val = ok_steps / max(1, eval_steps)
        else:
            current_cov_val = 0.0

        # Control uses full horizon pred
        act, info = ctrl(
            pos_x=float(robot_xy[0]),
            pos_y=float(robot_xy[1]),
            orientation_z=float(robot_th),
            boxes=[],
            obst_pred_traj=pred,
            obst_mask=obst_mask,
            goal=goal,
            current_obs=obs_now,
        )
        feasible = bool(info.get("feasible", False))
        if not feasible:
            if k > 15:
                infeasible_count += 1
            v, w = 0.0, 0.0
        else:
            v, w = float(act[0]), float(act[1])

        if k <= 15:
            v, w = 0.0, 0.0

        plan = info.get("final_path", None)

        # collision check uses current obs_now
        if obs_now.size == 0:
            dmin = np.inf
        else:
            dmin = float(min_dist_robot_to_peds(robot_xy, obs_now))
        is_coll = (dmin < safe_thresh)
        if is_coll and k > 15:
            collision_count += 1

        traj = np.stack(robot_traj, axis=0)

        for ax_i in range(2):
            robot_dots[ax_i].set_data([robot_xy[0]], [robot_xy[1]])
            robot_dots[ax_i].set_color("red" if is_coll else "blue")
            traj_lines[ax_i].set_data(traj[:, 0], traj[:, 1])

            if plan is not None:
                plan_lines[ax_i].set_data(plan[:, 0], plan[:, 1])
            else:
                plan_lines[ax_i].set_data([], [])

            # show current observed (●), pred(i_view)(×), gt(i_view)(◆)
            obs_scatters[ax_i].set_offsets(obs_now if obs_now.size else np.zeros((0, 2), dtype=np.float32))
            pred_scatters[ax_i].set_offsets(pred_i_vis if pred_i_vis.size else np.zeros((0, 2), dtype=np.float32))
            gt_scatters[ax_i].set_offsets(gt_i if gt_i.size else np.zeros((0, 2), dtype=np.float32))

        status_text.set_text(
            f"{dataset} | step={k} | i_view={i_view} | contain-Cov={current_cov_val:.2%} | "
            f"v={v:.2f}, w={w:.2f} | "
            f"coll={collision_count}, infeas={infeasible_count} | "
            f"dist_goal={float(np.linalg.norm(robot_xy - goal)):.2f}"
        )
        status_text.set_color("red" if (is_coll or not feasible) else "black")

        # step robot
        robot_xy, robot_th = unicycle_step(robot_xy, robot_th, v, w, DT)
        robot_traj.append(robot_xy.copy())
        return []

    anim = FuncAnimation(fig, update, frames=max_n_steps, interval=100, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    DATASET = "univ"
    run_one_episode_visual_from_file(
        dataset=DATASET,
        scenario_idx=0,
        time_horizon=12,
        grid_H=128,
        grid_W=128,
        alpha=0.1,
        test_size=0.30,
        random_state=0,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        backend="loky",
        n_skip=4,
        n_paths=1200,
    )