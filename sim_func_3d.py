from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# ---- 3D env/utils ----
from quad_env import QuadWorldEnv3D
from quad_env import (
    build_grid_3d,
    distance_field_points_3d,
    build_training_residuals_from_env_3d,
)

from cp.fuctional_spatial_cp import (
    FunctionalCPConfig,
    FunctionalCP,
    FunctionalCPParams,
)

from controllers.func_3d_mpc import FunctionalCPMPC3D


DT = 0.4
ROBOT_RAD = 0.4
OBSTACLE_RAD = 1.0 / np.sqrt(2.0)

MAX_LINEAR_X = 0.8
MIN_LINEAR_X = -0.8
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7
VZ_MIN, VZ_MAX = -0.3, 0.3


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


# ----------------------------
# rendering helpers (robust)
# ----------------------------
def _remove_poly3d(ax, poly: Optional[Poly3DCollection]) -> None:
    """Robustly remove Poly3DCollection from axis."""
    if poly is None:
        return
    try:
        poly.remove()
        return
    except Exception:
        pass
    try:
        if poly in ax.collections:
            ax.collections.remove(poly)
    except Exception:
        pass


def _clear_contours(cont_list: List) -> None:
    """cont_list holds QuadContourSet; remove all its artists robustly."""
    for cs in cont_list:
        try:
            for coll in cs.collections:
                try:
                    coll.remove()
                except Exception:
                    pass
        except Exception:
            pass
    cont_list.clear()


def _draw_isosurface_from_bool(
    ax,
    vol_bool_zyx: np.ndarray,   # (nz,ny,nx) bool
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    alpha: float,
    facecolor: Tuple[float, float, float],
) -> Optional[Poly3DCollection]:
    """
    marching_cubes expects numeric volume; we use {0,1}, level=0.5
    verts in (z_idx, y_idx, x_idx) index-space -> map to world coords via xs,ys,zs.
    """
    if vol_bool_zyx is None:
        return None
    if vol_bool_zyx.ndim != 3:
        return None
    nz, ny, nx = vol_bool_zyx.shape
    if nz < 2 or ny < 2 or nx < 2:
        return None
    if not np.any(vol_bool_zyx):
        return None

    vol = vol_bool_zyx.astype(np.float32)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    except Exception:
        return None

    # verts: (N,3) in (z_idx, y_idx, x_idx)
    vz_idx, vy_idx, vx_idx = verts[:, 0], verts[:, 1], verts[:, 2]

    vx = np.interp(vx_idx, np.arange(xs.size, dtype=np.float32), xs)
    vy = np.interp(vy_idx, np.arange(ys.size, dtype=np.float32), ys)
    vz = np.interp(vz_idx, np.arange(zs.size, dtype=np.float32), zs)

    V = np.stack([vx, vy, vz], axis=1).astype(np.float32)

    mesh = Poly3DCollection(V[faces], alpha=alpha)
    mesh.set_facecolor((*facecolor, alpha))
    mesh.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.add_collection3d(mesh)
    return mesh


def _min_dist_robot_to_points(robot_xyz: np.ndarray, pts_xyz: np.ndarray) -> float:
    if pts_xyz.size == 0:
        return float("inf")
    d = np.linalg.norm(pts_xyz - robot_xyz[None, :], axis=1)
    return float(np.min(d))


def _set_equal_aspect_3d(ax, xlim, ylim, zlim):
    """
    Matplotlib 3D equal aspect workaround: make axes ranges identical.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    r = max((x1 - x0), (y1 - y0), (z1 - z0)) / 2.0
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def run_one_episode_visual_3d(
    env: QuadWorldEnv3D,
    *,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    time_horizon: int = 12,
    alpha: float = 0.10,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    n_paths: int = 600,
    max_steps: int = 250, 
    i_view: int = 3,
    n_calib_samples: int = 120,
    goal_finish_dist: float = 0.8,
    mc_stride: int = 2,
    view_elev: float = 22.0,   # 논문용 카메라
    view_azim: float = -55.0,
    dpi: int = 140,
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    # ----------------------------
    # 1) grid
    # ----------------------------
    xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
    xs, ys, zs, X, Y, Z = build_grid_3d(xlim, ylim, zlim, nx, ny, nz)

    # 2D plane meshes for contours
    Xxy, Yxy = np.meshgrid(xs, ys, indexing="xy")  # (ny,nx)
    Xxz, Zxz = np.meshgrid(xs, zs, indexing="xy")  # (nz,nx)
    Yyz, Zyz = np.meshgrid(ys, zs, indexing="xy")  # (nz,ny)

    # ----------------------------
    # 2) offline calibration (functional CP)
    # ----------------------------
    residuals, _groups = build_training_residuals_from_env_3d(
        env,
        n_samples=n_calib_samples,
        X=X, Y=Y, Z=Z,
        time_horizon=time_horizon,
        episode_len=1000,
        group_by_episode=False,
    )
    print("[3D] residuals:", residuals.shape)  # (N,H,nz,ny,nx)

    cfg = FunctionalCPConfig(
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
        use_shared_basis=True,
        p_base=16,
    )
    calibrator = FunctionalCP(cfg)
    calibrator.fit(residuals)
    cp_params = calibrator.extract_params(rank=256)

    if tuple(cp_params.grid_shape) != (int(nz), int(ny), int(nx)):
        raise ValueError(f"cp_params.grid_shape={cp_params.grid_shape} but grid={(nz,ny,nx)}")

    # ----------------------------
    # 3) controller (baseline)
    # ----------------------------
    ctrl = FunctionalCPMPC3D(
        cp_params=cp_params,
        xs=xs, ys=ys, zs=zs,
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        v_xy_lim=(MIN_LINEAR_X, MAX_LINEAR_X),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        vz_lim=(VZ_MIN, VZ_MAX),
        n_paths=n_paths,
        seed=0,
    )

    # ----------------------------
    # 4) visualization
    # ----------------------------
    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    # 3D axes formatting
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=view_elev, azim=view_azim)
    _set_equal_aspect_3d(ax3d, xlim, ylim, zlim)
    ax3d.grid(False)

    # goal marker
    ax3d.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=140, edgecolors="k", linewidths=0.4)

    # 2D axis formatting (paper-like)
    for ax in (ax_xy, ax_xz, ax_yz):
        ax.grid(True, alpha=0.20, linewidth=0.6)
        ax.tick_params(labelsize=9)
    ax_xy.set_title("XY (min over z): boundary ϕ=0", fontsize=10)
    ax_xy.set_xlim(xlim); ax_xy.set_ylim(ylim); ax_xy.set_aspect("equal", adjustable="box")
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

    # ---- artists ----
    robot_dot, = ax3d.plot([], [], [], marker="o", markersize=5, linewidth=0)
    robot_line, = ax3d.plot([], [], [], linewidth=1.3, alpha=0.95)

    # NOTE: create scatters once; update _offsets3d only
    obs_sc = ax3d.scatter([], [], [], s=18, alpha=0.80)          # current obst
    pred_sc = ax3d.scatter([], [], [], s=32, marker="x", alpha=0.95)
    gt_sc = ax3d.scatter([], [], [], s=30, marker="D", alpha=0.90)

    # 2D projections
    xy_robot_dot, = ax_xy.plot([], [], "o", ms=4)
    xy_robot_line, = ax_xy.plot([], [], "-", lw=1.1, alpha=0.95)
    xy_obs_sc = ax_xy.scatter([], [], s=14, alpha=0.80)
    xy_pred_sc = ax_xy.scatter([], [], s=22, marker="x", alpha=0.95)
    xy_gt_sc = ax_xy.scatter([], [], s=20, marker="D", alpha=0.90)

    xz_robot_dot, = ax_xz.plot([], [], "o", ms=4)
    xz_robot_line, = ax_xz.plot([], [], "-", lw=1.1, alpha=0.95)
    xz_obs_sc = ax_xz.scatter([], [], s=14, alpha=0.80)
    xz_pred_sc = ax_xz.scatter([], [], s=22, marker="x", alpha=0.95)
    xz_gt_sc = ax_xz.scatter([], [], s=20, marker="D", alpha=0.90)

    yz_robot_dot, = ax_yz.plot([], [], "o", ms=4)
    yz_robot_line, = ax_yz.plot([], [], "-", lw=1.1, alpha=0.95)
    yz_obs_sc = ax_yz.scatter([], [], s=14, alpha=0.80)
    yz_pred_sc = ax_yz.scatter([], [], s=22, marker="x", alpha=0.95)
    yz_gt_sc = ax_yz.scatter([], [], s=20, marker="D", alpha=0.90)

    # meshes + contours
    true_mesh: Optional[Poly3DCollection] = None
    cp_mesh: Optional[Poly3DCollection] = None
    xy_contours: List = []
    xz_contours: List = []
    yz_contours: List = []

    robot_traj: List[np.ndarray] = []

    # stats
    collision_count = 0
    infeasible_count = 0

    # containment coverage (binary per step) + additional helpful metrics
    contain_ok_steps = 0
    contain_eval_steps = 0
    voxel_recall_sum = 0.0  # mean over time of |true∩cp|/|true|

    def reconstruct_S_grid(params: FunctionalCPParams, t_idx: int) -> np.ndarray:
        S_flat = params.S_mean + (params.coeffs[t_idx] @ params.phi_basis)
        return S_flat.reshape(params.grid_shape).astype(np.float32)

    def upd(k: int):
        nonlocal obs, true_mesh, cp_mesh
        nonlocal collision_count, infeasible_count
        nonlocal contain_ok_steps, contain_eval_steps, voxel_recall_sum

        # ✅ hard stop: prepared steps exhausted
        if k >= int(max_steps) - 1:
            ax3d.set_title(f"STOP (max_steps reached) | step={k}")
            anim.event_source.stop()
            return []

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj, dtype=np.float32)

        # goal distance
        dist_goal = float(np.linalg.norm(robot - goal))
        if dist_goal <= float(goal_finish_dist):
            ax3d.set_title(f"FINISH | step={k} | dist_goal={dist_goal:.2f}")
            anim.event_source.stop()
            return []

        # current obstacles
        h_dict = obs.get("history", {})
        obs_now = (
            np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)
            if len(h_dict) else np.zeros((0, 3), dtype=np.float32)
        )

        # predictions stacked (H,M,3)
        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)
        i_view_clamped = int(np.clip(i_view, 0, pred.shape[0] - 1)) if pred.size else int(i_view)

        pred_i = pred[i_view_clamped] if pred.size else np.zeros((0, 3), dtype=np.float32)
        mask_i = pred_mask[i_view_clamped] if pred_mask.size else np.zeros((0,), dtype=bool)
        pred_i_vis = pred_i[mask_i] if pred_i.size else np.zeros((0, 3), dtype=np.float32)

        # ground-truth future at i_view (if exists)
        true_future_pts = None
        for key in ["future", "future_gt", "gt_future", "ground_truth_future", "truth_future"]:
            if key in obs:
                fut = obs[key]
                if isinstance(fut, dict):
                    fut_stack, fut_mask, _ = stack_pred3d_from_p_dict(fut, horizon=time_horizon)
                    fut_i = fut_stack[i_view_clamped]
                    fut_m = fut_mask[i_view_clamped]
                    true_future_pts = fut_i[fut_m] if fut_i.size else np.zeros((0, 3), dtype=np.float32)
                else:
                    arr = np.asarray(fut, dtype=np.float32)
                    # expected: (H, M, 3) or (H, 3) etc.
                    if arr.ndim == 3 and arr.shape[-1] == 3 and i_view_clamped < arr.shape[0]:
                        true_future_pts = arr[i_view_clamped]
                break
        gt_i_vis = true_future_pts if true_future_pts is not None else np.zeros((0, 3), dtype=np.float32)

        # distance fields on grid
        D_pred = (
            distance_field_points_3d(pred_i_vis, X, Y, Z)
            if pred_i_vis.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
        )

        # CP lower bound
        S_i = reconstruct_S_grid(cp_params, i_view_clamped)

        x_i = cp_params.x_star[i_view_clamped].reshape(cp_params.grid_shape)  # (nz,ny,nx)

        U_i = x_i + float(cp_params.rhos[i_view_clamped]) * np.maximum(S_i, 1e-9)
        U_i = np.maximum(U_i, 0.0) 
        D_lower = np.maximum(D_pred - U_i, 0.0)
        cp_unsafe = (D_lower < safe_rad)

        # GT unsafe
        D_true = None
        true_unsafe = None
        if gt_i_vis.size:
            D_true = distance_field_points_3d(gt_i_vis, X, Y, Z)
            true_unsafe = (D_true < safe_rad)

        # ✅ containment evaluated *each step* when GT exists
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
                voxel_recall = 1.0  # no true unsafe => trivially fine

        cov_running = (contain_ok_steps / max(1, contain_eval_steps))

        # collision at current time (robot vs current obst)
        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            collision_count += 1

        # control
        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            boxes_3d=[],
        )
        feasible = bool(info.get("feasible", False))
        if not feasible:
            infeasible_count += 1
            v_xy, w, vz = 0.0, 0.0, 0.0
        else:
            v_xy, w, vz = map(float, act)

        # ---- update 3D artists ----
        robot_dot.set_data([robot[0]], [robot[1]])
        robot_dot.set_3d_properties([robot[2]])
        robot_line.set_data(tr[:, 0], tr[:, 1])
        robot_line.set_3d_properties(tr[:, 2])

        # keep _offsets3d shape consistent
        if obs_now.size:
            obs_sc._offsets3d = (obs_now[:, 0], obs_now[:, 1], obs_now[:, 2])
        else:
            obs_sc._offsets3d = ([], [], [])
        if pred_i_vis.size:
            pred_sc._offsets3d = (pred_i_vis[:, 0], pred_i_vis[:, 1], pred_i_vis[:, 2])
        else:
            pred_sc._offsets3d = ([], [], [])
        if gt_i_vis.size:
            gt_sc._offsets3d = (gt_i_vis[:, 0], gt_i_vis[:, 1], gt_i_vis[:, 2])
        else:
            gt_sc._offsets3d = ([], [], [])

        # ---- update 2D projected markers ----
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

        # ---- update iso-surfaces (robust remove + redraw) ----
        _remove_poly3d(ax3d, true_mesh)
        _remove_poly3d(ax3d, cp_mesh)
        true_mesh = None
        cp_mesh = None

        s = int(max(1, mc_stride))
        cp_ds = cp_unsafe[::s, ::s, ::s]
        xs_ds = xs[np.arange(0, xs.size, s)]
        ys_ds = ys[np.arange(0, ys.size, s)]
        zs_ds = zs[np.arange(0, zs.size, s)]

        # paper-like translucency
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

        # ---- update projection contours (robust clear) ----
        _clear_contours(xy_contours)
        _clear_contours(xz_contours)
        _clear_contours(yz_contours)

        phi_cp = D_lower - safe_rad
        phi_true = (D_true - safe_rad) if D_true is not None else None

        # min-projection boundaries
        phi_cp_xy = np.min(phi_cp, axis=0)  # (ny,nx)
        phi_cp_xz = np.min(phi_cp, axis=1)  # (nz,nx)
        phi_cp_yz = np.min(phi_cp, axis=2)  # (nz,ny)

        # CP boundary
        xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_cp_xy, levels=[0.0], linewidths=1.8))
        xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_cp_xz, levels=[0.0], linewidths=1.8))
        yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_cp_yz, levels=[0.0], linewidths=1.8))

        # GT boundary
        if phi_true is not None:
            phi_true_xy = np.min(phi_true, axis=0)
            phi_true_xz = np.min(phi_true, axis=1)
            phi_true_yz = np.min(phi_true, axis=2)
            xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_true_xy, levels=[0.0], linewidths=1.8, linestyles="--"))
            xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_true_xz, levels=[0.0], linewidths=1.8, linestyles="--"))
            yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_true_yz, levels=[0.0], linewidths=1.8, linestyles="--"))

        # ---- titles + text panel ----
        mean_recall = (voxel_recall_sum / max(1, contain_eval_steps)) if contain_eval_steps > 0 else 0.0
        ok_str = "-" if containment_ok is None else ("OK" if containment_ok else "FAIL")

        ax3d.set_title(
            f"3D unsafe iso-surfaces | step={k} | i_view={i_view_clamped} | feasible={feasible} | contain={ok_str}",
            fontsize=11
        )

        ax_txt.clear()
        ax_txt.axis("off")
        ax_txt.text(
            0.02, 0.98,
            "\n".join([
                f"step = {k}/{max_steps-1}",
                f"i_view = {i_view_clamped}",
                f"dist_goal = {dist_goal:.3f} (finish<= {goal_finish_dist})",
                "",
                f"feasible = {feasible}",
                f"act = (v_xy={v_xy:.3f}, yaw_rate={w:.3f}, vz={vz:.3f})",
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

        # step env
        obs = env.step(v_xy, w, vz)
        return []

    anim = FuncAnimation(fig, upd, frames=max_steps, interval=80, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=DT,
        horizon=12,
        n_obs=30,
        world_bounds_xyz=((-5, 15), (-5, 15), (0.0, 20.0)),
        seed=0,
    )

    run_one_episode_visual_3d(
        env,
        nx=40, ny=40, nz=40,
        time_horizon=12,
        alpha=0.10,
        test_size=0.30,
        random_state=0,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        backend="loky",
        n_skip=4,
        n_paths=600,
        max_steps=250,
        i_view=3,
        n_calib_samples=120,
        goal_finish_dist=0.8,
        mc_stride=2,
        view_elev=22.0,
        view_azim=-55.0,
        dpi=140,
    )