from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Local module imports
from preprocess import load_eth_pickle_preprocessed
from cp.functional_cp import compute_cp_upper_envelopes, get_envelopes_function, CPStepParameters
from controllers.func_cp_mpc import FunctionalCPMPC

from utils import build_grid, distance_field_points
from sims.sim_utils import (
    build_cv_predictions,
    compute_lower_field_single_step,
    min_dist_robot_to_peds,
    align_mask_dims,
    unicycle_step,
    reflect_to_box_xy,
    make_cv_prediction_from_two_points,
)


# ==============================================================================
# 0. Config
# ==============================================================================

@dataclass(frozen=True)
class SimConfig:
    base_dir: str
    dataset: str = "univ"

    # grid / world
    H: int = 128
    W: int = 128
    box: float = 100.0

    # sequence
    TSTEPS: int = 20
    TIME_HORIZON: int = 20
    mpc_dt: float = 0.1

    # CP calibration
    seed: int = 2021
    n_train: int = 1000
    num_peds: int = 20
    p_base: int = 3
    k_mix: int = 5
    alpha: float = 0.05
    test_size: float = 0.30
    random_state: int = 0
    n_jobs: int = max(1, (os.cpu_count() or 4) - 2)
    backend: str = "loky"

    # controller
    n_skip: int = 4
    n_paths: int = 1200
    robot_rad: float = 0.4
    obstacle_rad: float = 1.0 / np.sqrt(2.0)
    hard_cp_constraint: bool = True
    w_cp_violation: float = 200.0

    # misc
    goal_reach_threshold: float = 2.0
    init_robot_xy: Tuple[float, float] = (10.0, 10.0)
    init_robot_th: float = 0.0
    goal_xy: Tuple[float, float] = (90.0, 90.0)

    @property
    def world_center(self) -> np.ndarray:
        return np.array([self.box / 2.0, self.box / 2.0], dtype=np.float32)


# ==============================================================================
# 1. Utility: residual construction
# ==============================================================================

def build_training_residuals_valid_only(
    obst_true: np.ndarray,
    obst_pred: np.ndarray,
    masks: np.ndarray,
    Xg: np.ndarray,
    Yg: np.ndarray,
    world_center: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
      residuals: (N_eff, T_eff, H, W)
      t_offset: int
    """
    if obst_true.ndim == 3:
        obst_true, obst_pred, masks = obst_true[:, :, None, :], obst_pred[:, :, None, :], masks[:, :, None]

    N, T, M, _ = obst_true.shape
    t_offset = horizon + 1
    T_eff = T - t_offset

    residuals = []
    for n in range(N):
        for m in range(M):
            if not np.all(masks[n, t_offset:, m]):
                continue
            res = np.zeros((T_eff, Xg.shape[0], Xg.shape[1]), dtype=np.float32)
            for t in range(T_eff):
                y_true = obst_true[n, t + t_offset, m] - world_center
                y_pred = obst_pred[n, t + t_offset, m] - world_center
                res[t] = distance_field_points(y_pred, Xg, Yg) - distance_field_points(y_true, Xg, Yg)
            residuals.append(res)

    if len(residuals) == 0:
        raise RuntimeError("No valid residual trajectories were collected (mask filtering too strict?).")

    return np.stack(residuals, axis=0), t_offset


# ==============================================================================
# 2. Data + Calibration
# ==============================================================================

def prepare_data(cfg: SimConfig):
    train_data = load_eth_pickle_preprocessed(
        dataset=cfg.dataset,
        box=cfg.box,
        T=cfg.TSTEPS,
        split_ratio=0.8,
        seed=cfg.seed,
        base_dir=cfg.base_dir,
        num_peds=cfg.num_peds,
    )
    train_true_raw, _, train_mask, test_true_raw, _, test_mask = train_data

    # -> (N, T, M, 2)
    train_true = np.transpose(train_true_raw, (0, 2, 1, 3))
    test_true = np.transpose(test_true_raw, (0, 2, 1, 3))
    train_mask = align_mask_dims(train_mask, cfg.TSTEPS, cfg.num_peds)
    test_mask = align_mask_dims(test_mask, cfg.TSTEPS, cfg.num_peds)

    return train_true, train_mask, test_true, test_mask


def calibrate_cp(cfg: SimConfig, train_true: np.ndarray, train_mask: np.ndarray, Xg: np.ndarray, Yg: np.ndarray):
    """
    Outputs:
      g_upper_grid: (T_eff, H, W) grid envelope (for visualization / debugging)
      cp_params:    List[CPStepParameters] (for controller online evaluation)
    """
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(train_true.shape[0], size=min(cfg.n_train, train_true.shape[0]), replace=False)

    train_pred_sel = build_cv_predictions(train_true[idx], cfg.box, horizon=1)
    residuals, _ = build_training_residuals_valid_only(
        obst_true=train_true[idx],
        obst_pred=train_pred_sel,
        masks=train_mask[idx],
        Xg=Xg,
        Yg=Yg,
        world_center=cfg.world_center,
        horizon=1,
    )

    print("Calibrating CP envelopes...")

    g_upper_ds = compute_cp_upper_envelopes(
        residuals_train=residuals,
        p_base=cfg.p_base,
        K=cfg.k_mix,
        alpha=cfg.alpha,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
    )

    cp_params = get_envelopes_function(
        residuals_train=residuals,
        p_base=cfg.p_base,
        K=cfg.k_mix,
        alpha=cfg.alpha,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        backend=cfg.backend,
    )

    # --- pad / crop to TIME_HORIZON for convenience ---
    g_upper_grid = pad_or_crop_time(g_upper_ds, cfg.TIME_HORIZON)
    cp_params = pad_or_crop_params(cp_params, cfg.TIME_HORIZON)

    return g_upper_grid.astype(np.float32), cp_params


def pad_or_crop_time(arr: np.ndarray, T: int) -> np.ndarray:
    if arr.shape[0] >= T:
        return arr[:T]
    pad = np.repeat(arr[-1][None], T - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def pad_or_crop_params(params: List[CPStepParameters], T: int) -> List[CPStepParameters]:
    if len(params) >= T:
        return params[:T]
    if len(params) == 0:
        raise ValueError("cp_params is empty.")
    last = params[-1]
    out = list(params)
    for t in range(len(params), T):
        # 같은 파라미터로 t_idx만 바꿔서 padding (보수적/실용적)
        out.append(CPStepParameters(
            t_idx=t,
            mean_field=last.mean_field,
            phi_basis=last.phi_basis,
            epsilon=last.epsilon,
            mus=last.mus,
            sigmas=last.sigmas,
            rks=last.rks,
            K=last.K,
            p_eff=last.p_eff,
        ))
    return out


# ==============================================================================
# 3. Controller init
# ==============================================================================

def init_controller(cfg: SimConfig, cp_params: List[CPStepParameters], obstacle_rad_with_tracking: float):
    return FunctionalCPMPC(
        box=cfg.box,
        world_center=cfg.world_center,
        cp_params=cp_params,
        grid_H=cfg.H,   # (컨트롤러가 grid_res 필요하면 유지, 실제론 param eval만 해도 됨)
        grid_W=cfg.W,
        n_steps=cfg.TIME_HORIZON,
        dt=cfg.mpc_dt,
        n_skip=cfg.n_skip,
        n_paths=cfg.n_paths,
        seed=0,
        robot_rad=cfg.robot_rad,
        obstacle_rad=obstacle_rad_with_tracking,
        hard_cp_constraint=cfg.hard_cp_constraint,
        w_cp_violation=cfg.w_cp_violation,
    )


# ==============================================================================
# 4. Visualization harness (1 episode)
# ==============================================================================

def run_one_episode_visual(cfg: SimConfig):
    # --- data ---
    train_true, train_mask, test_true, test_mask = prepare_data(cfg)
    xs, ys, Xg, Yg = build_grid(cfg.box, cfg.H, cfg.W)

    # --- calibration ---
    g_upper_grid, cp_params = calibrate_cp(cfg, train_true, train_mask, Xg, Yg)

    # --- pick episode ---
    rng = np.random.default_rng(cfg.seed)
    ep = int(rng.integers(test_true.shape[0]))
    obst_ep, mask_ep = test_true[ep], test_mask[ep]

    # --- controller ---
    max_tracking_error = 0.05
    ctrl = init_controller(cfg, cp_params, obstacle_rad_with_tracking=cfg.obstacle_rad + max_tracking_error)

    # --- state ---
    robot_xy = np.array(cfg.init_robot_xy, dtype=np.float32)
    robot_th = float(cfg.init_robot_th)
    goal = np.array(cfg.goal_xy, dtype=np.float32)
    robot_traj = [robot_xy.copy()]
    planned_next_xy: Optional[np.ndarray] = None

    # --- plot ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    extent = [xs[0] + cfg.world_center[0], xs[-1] + cfg.world_center[0],
              ys[0] + cfg.world_center[1], ys[-1] + cfg.world_center[1]]

    im_heat = axs[0].imshow(np.zeros((cfg.H, cfg.W)), extent=extent, origin="lower", vmin=0, vmax=10)
    plt.colorbar(im_heat, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title("CP lower-bound field (grid viz only)")

    axs[1].set_title(f"Unsafe region: True vs CP (alpha={cfg.alpha})")
    im_true = axs[1].imshow(np.zeros((cfg.H, cfg.W)), extent=extent, origin="lower", vmin=0, vmax=1, alpha=0.4)
    im_cp = axs[1].imshow(np.zeros((cfg.H, cfg.W)), extent=extent, origin="lower", vmin=0, vmax=1, alpha=0.5)

    robot_dots, traj_lines, plan_lines, peds_scatters = [], [], [], []
    for ax in axs:
        ax.set_xlim(0, cfg.box)
        ax.set_ylim(0, cfg.box)
        ax.set_aspect("equal")
        ax.plot(goal[0], goal[1], "gx", ms=12, mew=2)

        dot, = ax.plot([], [], "bo", ms=7, zorder=5)
        line, = ax.plot([], [], "b-", lw=1, alpha=0.5)
        plan_p, = ax.plot([], [], "c--", lw=1.5, zorder=4)
        peds = ax.scatter([], [], s=25, edgecolors="black", zorder=6)

        robot_dots.append(dot)
        traj_lines.append(line)
        plan_lines.append(plan_p)
        peds_scatters.append(peds)

    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10, fontweight="bold")

    collision_count = 0
    infeasible_count = 0

    def update(k: int):
        nonlocal robot_xy, robot_th, planned_next_xy, max_tracking_error, collision_count, infeasible_count

        t = k + 1

        # (A) tracking error update (tighten obstacle radius)
        if planned_next_xy is not None:
            curr_err = float(np.linalg.norm(robot_xy - planned_next_xy))
            if curr_err > max_tracking_error:
                max_tracking_error = curr_err
                ctrl.obstacle_rad = cfg.obstacle_rad + max_tracking_error

        safe_thresh = cfg.robot_rad + cfg.obstacle_rad + max_tracking_error

        # goal check
        dist_to_goal = float(np.linalg.norm(robot_xy - goal))
        if dist_to_goal < cfg.goal_reach_threshold:
            status_text.set_text(f"GOAL reached | dist={dist_to_goal:.2f} | infeasible={infeasible_count}")
            status_text.set_color("green")
            anim.event_source.stop()
            return []

        # (B) perception & prediction
        p_t = obst_ep[t]
        vis = mask_ep[t].astype(bool)
        pred = make_cv_prediction_from_two_points(obst_ep[t - 1], p_t, cfg.TIME_HORIZON, cfg.box)
        obst_mask = np.repeat(vis[None], cfg.TIME_HORIZON, axis=0)

        # (C) visualization grid field (not used by controller)
        lower_field = compute_lower_field_single_step(
            pred[0],
            obst_mask[0],
            Xg,
            Yg,
            g_upper_grid[0],          # ✅ grid envelope for viz
            cfg.world_center,
        )

        y_true_rel = p_t[vis] - cfg.world_center
        true_dist = distance_field_points(y_true_rel, Xg, Yg) if y_true_rel.size > 0 else np.full((cfg.H, cfg.W), 20.0)

        im_true.set_data((true_dist < safe_thresh).astype(float))
        im_cp.set_data((lower_field < safe_thresh).astype(float))

        # (D) MPC step (uses cp_params internally; no grid eval)
        act, info = ctrl(
            pos_x=float(robot_xy[0]),
            pos_y=float(robot_xy[1]),
            orientation_z=float(robot_th),
            boxes=None,
            obst_pred_traj=pred,
            obst_mask=obst_mask,
            goal=goal,
        )

        feasible = bool(info.get("feasible", False))
        if not feasible:
            infeasible_count += 1
            v, w = 0.0, 0.0
        else:
            v, w = float(act[0]), float(act[1])

        plan = info.get("final_path")
        planned_next_xy = plan[1].copy() if (plan is not None and len(plan) > 1) else robot_xy.copy()

        # (E) physics + collision
        robot_xy, robot_th = unicycle_step(robot_xy, robot_th, v, w, cfg.mpc_dt)
        robot_xy = reflect_to_box_xy(robot_xy, cfg.box)
        robot_traj.append(robot_xy.copy())

        dmin = float(min_dist_robot_to_peds(robot_xy, p_t[vis]))
        is_coll = dmin < (cfg.robot_rad + cfg.obstacle_rad)
        if is_coll:
            collision_count += 1

        # (F) render
        im_heat.set_data(lower_field)
        curr_traj = np.stack(robot_traj)

        for i in range(2):
            robot_dots[i].set_data([robot_xy[0]], [robot_xy[1]])
            robot_dots[i].set_color("red" if is_coll else "blue")
            traj_lines[i].set_data(curr_traj[:, 0], curr_traj[:, 1])
            if plan is not None:
                plan_lines[i].set_data(plan[:, 0], plan[:, 1])
            peds_scatters[i].set_offsets(p_t[vis] if p_t[vis].size else np.zeros((0, 2)))

        status_text.set_text(
            f"Step:{t} | coll:{collision_count} | infeas:{infeasible_count} | "
            f"track_err_max:{max_tracking_error:.3f} | safe_thr:{safe_thresh:.2f}"
        )
        status_text.set_color("red" if (is_coll or not feasible) else "black")
        return []

    anim = FuncAnimation(fig, update, frames=obst_ep.shape[0] - 1, interval=100, blit=False)
    plt.show()


# ==============================================================================
# entry
# ==============================================================================

if __name__ == "__main__":
    cfg = SimConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))
    run_one_episode_visual(cfg)