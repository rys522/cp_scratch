from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

# ✅ 기존 FunctionalCPMPC3D 대신, 네가 이미 갖고 있는 ECP 컨트롤러 import로 교체
# from controllers.ecp_3d_mpc import EgocentricCPMPC3D
from controllers.ecp_mpc_3d import EgocentricCPMPC3D  # <- 너 프로젝트에 맞게 경로만 고쳐


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


# ------------------------------------------------------------
# Marching-cubes iso-surface renderer
# ------------------------------------------------------------
def _draw_isosurface_from_bool(
    ax,
    vol_bool_zyx: np.ndarray,   # (nz,ny,nx) bool
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    alpha: float,
    color_rgb: Tuple[float, float, float],
) -> Optional[Poly3DCollection]:
    if vol_bool_zyx is None:
        return None
    if vol_bool_zyx.shape[0] < 2 or vol_bool_zyx.shape[1] < 2 or vol_bool_zyx.shape[2] < 2:
        return None
    if not np.any(vol_bool_zyx):
        return None

    vol = vol_bool_zyx.astype(np.float32)
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)
    except Exception:
        return None

    vz_idx = verts[:, 0]
    vy_idx = verts[:, 1]
    vx_idx = verts[:, 2]

    vx = np.interp(vx_idx, np.arange(xs.size, dtype=np.float32), xs)
    vy = np.interp(vy_idx, np.arange(ys.size, dtype=np.float32), ys)
    vz = np.interp(vz_idx, np.arange(zs.size, dtype=np.float32), zs)

    V = np.stack([vx, vy, vz], axis=1).astype(np.float32)

    mesh = Poly3DCollection(V[faces], alpha=alpha)
    mesh.set_facecolor((*color_rgb, alpha))
    mesh.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.add_collection3d(mesh)
    return mesh


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
    nz: int = 12,
    time_horizon: int = 12,
    alpha: float = 0.10,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_skip: int = 4,
    max_steps: int = 200,
    i_view: int = 7,
    n_calib_samples: int = 120,
    goal_finish_dist: float = 0.8,
    mc_stride: int = 2,
):
    safe_rad = ROBOT_RAD + OBSTACLE_RAD

    # ----------------------------
    # 1) 3D grid
    # ----------------------------
    xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
    xs, ys, zs, X, Y, Z = build_grid_3d(xlim, ylim, zlim, nx, ny, nz)

    # 2D plane meshes for contours
    Xxy, Yxy = np.meshgrid(xs, ys, indexing="xy")  # (ny,nx)
    Xxz, Zxz = np.meshgrid(xs, zs, indexing="xy")  # (nz,nx)
    Yyz, Zyz = np.meshgrid(ys, zs, indexing="xy")  # (nz,ny)

    # ----------------------------
    # 2) offline calibration (시각화용 CP bound 유지)
    # ----------------------------
    residuals, groups = build_training_residuals_from_env_3d(
        env,
        n_samples=n_calib_samples,
        X=X, Y=Y, Z=Z,
        time_horizon=time_horizon,
        episode_len=60,
        group_by_episode=True,
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
    cp_params = calibrator.extract_params(rank=16)

    if tuple(cp_params.grid_shape) != (int(nz), int(ny), int(nx)):
        raise ValueError(f"cp_params.grid_shape={cp_params.grid_shape} but grid={(nz,ny,nx)}")

    # ----------------------------
    # 3) ✅ controller: ECP-MPC로 교체
    # ----------------------------
    ctrl = EgocentricCPMPC3D(
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        min_linear_x=MIN_LINEAR_X,
        max_linear_x=MAX_LINEAR_X,
        min_angular_z=MIN_ANGULAR_Z,
        max_angular_z=MAX_ANGULAR_Z,
        vz_min=VZ_MIN,
        vz_max=VZ_MAX,
        calibration_set_size=30,
        miscoverage_level=alpha,
        step_size=0.05,
    )

    # ----------------------------
    # 4) Visualization (3D + projections)
    # ----------------------------
    obs = env.reset()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")

    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)
    ax3d.scatter([goal[0]], [goal[1]], [goal[2]], marker="x", s=90)

    ax_xy.set_title("XY projection (min over z): boundary φ=0")
    ax_xy.set_xlim(xlim); ax_xy.set_ylim(ylim)
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.plot([goal[0]], [goal[1]], "gx", ms=10, mew=2)

    ax_xz.set_title("XZ projection (min over y): boundary φ=0")
    ax_xz.set_xlim(xlim); ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.grid(True, alpha=0.25)
    ax_xz.plot([goal[0]], [goal[2]], "gx", ms=10, mew=2)

    ax_yz.set_title("YZ projection (min over x): boundary φ=0")
    ax_yz.set_xlim(ylim); ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")
    ax_yz.grid(True, alpha=0.25)
    ax_yz.plot([goal[1]], [goal[2]], "gx", ms=10, mew=2)

    # 3D artists
    robot_dot, = ax3d.plot([], [], [], "bo", markersize=6)
    robot_line, = ax3d.plot([], [], [], "b-", linewidth=1.2)
    obs_sc = ax3d.scatter([], [], [], s=35)
    pred_sc = ax3d.scatter([], [], [], s=55, marker="x")

    # 2D artists
    xy_robot_dot, = ax_xy.plot([], [], "bo", ms=6)
    xy_robot_line, = ax_xy.plot([], [], "b-", lw=1)
    xy_obs_sc = ax_xy.scatter([], [], s=25)
    xy_pred_sc = ax_xy.scatter([], [], s=40, marker="x")

    xz_robot_dot, = ax_xz.plot([], [], "bo", ms=6)
    xz_robot_line, = ax_xz.plot([], [], "b-", lw=1)
    xz_obs_sc = ax_xz.scatter([], [], s=25)
    xz_pred_sc = ax_xz.scatter([], [], s=40, marker="x")

    yz_robot_dot, = ax_yz.plot([], [], "bo", ms=6)
    yz_robot_line, = ax_yz.plot([], [], "b-", lw=1)
    yz_obs_sc = ax_yz.scatter([], [], s=25)
    yz_pred_sc = ax_yz.scatter([], [], s=40, marker="x")

    true_mesh: Optional[Poly3DCollection] = None
    cp_mesh: Optional[Poly3DCollection] = None

    xy_contours: List = []
    xz_contours: List = []
    yz_contours: List = []

    robot_traj: List[np.ndarray] = []

    collision_count = 0
    infeasible_count = 0
    ok_steps = 0
    eval_steps = 0
    cov_running = 0.0

    # reconstruct S grid at horizon step (시각화 bound용)
    def reconstruct_S_grid(params: FunctionalCPParams, t_idx: int) -> np.ndarray:
        S_flat = params.S_mean + (params.coeffs[t_idx] @ params.phi_basis)
        return S_flat.reshape(params.grid_shape).astype(np.float32)

    def _clear_contours(cont_list: List):
        for c in cont_list:
            try:
                for coll in c.collections:
                    coll.remove()
            except Exception:
                pass
        cont_list.clear()

    # ✅ ECP가 요구하는 update_*에 넣어줄 dict 포맷 만들기
    def _make_ecp_obs_from_history(history: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        ECP가 기대하는 obs: pid -> trajectory-like
        네 기존 2D ECP 코드에 맞추면 각 value는 (T,dim) 이고 마지막 v[-1]을 씀.
        여기서는 env history를 그대로 넘겨서 호환되게 둔다.
        """
        return history or {}

    def _make_ecp_pred_from_env_pred(prediction: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        ECP가 기대하는 predictions: pid -> (H,3) (혹은 sample dim 포함)
        env prediction을 그대로 넘긴다.
        """
        return prediction or {}

    def upd(k: int):
        nonlocal obs, true_mesh, cp_mesh
        nonlocal collision_count, infeasible_count, ok_steps, eval_steps, cov_running

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj, dtype=np.float32)

        dist_goal = float(np.linalg.norm(robot - goal))
        if dist_goal <= float(goal_finish_dist):
            ax3d.set_title(
                f"FINISH step={k} | dist_goal={dist_goal:.2f} | Cov={cov_running:.2%} | "
                f"coll={collision_count} | infeas={infeasible_count}"
            )
            anim.event_source.stop()
            return []

        # obstacles now
        h_dict = obs.get("history", {})
        obs_now = (
            np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)
            if len(h_dict) else np.zeros((0, 3), dtype=np.float32)
        )

        # predictions stacked (렌더링용)
        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)
        pred_i = pred[i_view]
        mask_i = pred_mask[i_view]
        pred_i_vis = pred_i[mask_i] if pred_i.size else np.zeros((0, 3), dtype=np.float32)

        # true future at i_view (if exists)
        true_future_pts = None
        for key in ["future", "future_gt", "gt_future", "ground_truth_future", "truth_future"]:
            if key in obs:
                fut = obs[key]
                if isinstance(fut, dict):
                    fut_stack, fut_mask, _ = stack_pred3d_from_p_dict(fut, horizon=time_horizon)
                    fut_i = fut_stack[i_view]
                    fut_m = fut_mask[i_view]
                    true_future_pts = fut_i[fut_m] if fut_i.size else np.zeros((0, 3), dtype=np.float32)
                else:
                    arr = np.asarray(fut, dtype=np.float32)
                    if arr.ndim == 3 and arr.shape[-1] == 3 and i_view < arr.shape[0]:
                        true_future_pts = arr[i_view]
                break

        # CP bound distance fields (시각화용; controller는 ECP)
        D_pred = (
            distance_field_points_3d(pred_i_vis, X, Y, Z)
            if pred_i_vis.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
        )

        S_i = reconstruct_S_grid(cp_params, i_view)
        U_i = float(cp_params.rhos[i_view]) * np.maximum(S_i, 1e-9)

        D_lower = np.maximum(D_pred - U_i, 0.0)
        cp_unsafe = (D_lower < safe_rad)

        true_unsafe = None
        D_true = None
        if true_future_pts is not None:
            D_true = (
                distance_field_points_3d(true_future_pts, X, Y, Z)
                if true_future_pts.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
            )
            true_unsafe = (D_true < safe_rad)

        if true_unsafe is not None:
            ok = (not np.any(true_unsafe & (~cp_unsafe)))
            if k > i_view:
                eval_steps += 1
                ok_steps += int(ok)
                cov_running = ok_steps / max(1, eval_steps)

        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            collision_count += 1

        # =====================================================
        # ✅✅✅ ECP-MPC control (컨트롤러만 교체 핵심)
        # 1) queues 업데이트
        # 2) ctrl 호출
        # =====================================================
        pred_for_ecp = _make_ecp_pred_from_env_pred(obs.get("prediction", {}))
        obs_for_ecp = _make_ecp_obs_from_history(obs.get("history", {}))

        # (A) prediction queue 업데이트 (t-1까지 저장한다고 가정하는 구현이 많아서)
        #     네 ECP 구현에 update_predictions가 있으면 이 줄이 필요.
        if hasattr(ctrl, "update_predictions"):
            ctrl.update_predictions(pred_for_ecp)

        # (B) observation queue 업데이트 + alpha 업데이트
        if hasattr(ctrl, "update_observations"):
            ctrl.update_observations(obs_for_ecp)

        # (C) ctrl 호출
        #     네 ECP __call__ 시그니처에 맞춰 2가지 후보를 순서대로 트라이
        act, info = None, {}
        try:
            # 후보 1: (robot_xyz, robot_yaw, predictions, goal_xyz, boxes_3d=[])
            act, info = ctrl(
                robot_xyz=robot,
                robot_yaw=yaw,
                predictions=pred_for_ecp,
                goal_xyz=goal,
                boxes_3d=[],
            )
        except TypeError:
            # 후보 2: (pos_x,pos_y,orientation_z, pos_z, predictions, goal) 류
            act, info = ctrl(
                pos_x=float(robot[0]),
                pos_y=float(robot[1]),
                pos_z=float(robot[2]),
                orientation_z=float(yaw),
                predictions=pred_for_ecp,
                goal=goal,
                boxes_3d=[],
            )

        feasible = bool(info.get("feasible", False)) if isinstance(info, dict) else (act is not None)
        if not feasible or act is None:
            infeasible_count += 1
            v_xy, w, vz = 0.0, 0.0, 0.0
        else:
            # act가 (v_xy, w, vz) 또는 vel[1] 같은 형태일 수 있음
            act_arr = np.asarray(act, dtype=np.float32).reshape(-1)
            if act_arr.size >= 3:
                v_xy, w, vz = map(float, act_arr[:3])
            elif act_arr.size == 2:
                v_xy, w = map(float, act_arr[:2])
                vz = 0.0
            else:
                v_xy, w, vz = 0.0, 0.0, 0.0

        # --- update 3D artists ---
        robot_dot.set_data([robot[0]], [robot[1]])
        robot_dot.set_3d_properties([robot[2]])
        robot_line.set_data(tr[:, 0], tr[:, 1])
        robot_line.set_3d_properties(tr[:, 2])

        if obs_now.size:
            obs_sc._offsets3d = (obs_now[:, 0], obs_now[:, 1], obs_now[:, 2])
        else:
            obs_sc._offsets3d = ([], [], [])
        if pred_i_vis.size:
            pred_sc._offsets3d = (pred_i_vis[:, 0], pred_i_vis[:, 1], pred_i_vis[:, 2])
        else:
            pred_sc._offsets3d = ([], [], [])

        # --- update 2D projected markers ---
        xy_robot_dot.set_data([robot[0]], [robot[1]])
        xy_robot_line.set_data(tr[:, 0], tr[:, 1])

        xz_robot_dot.set_data([robot[0]], [robot[2]])
        xz_robot_line.set_data(tr[:, 0], tr[:, 2])

        yz_robot_dot.set_data([robot[1]], [robot[2]])
        yz_robot_line.set_data(tr[:, 1], tr[:, 2])

        if obs_now.size:
            xy_obs_sc.set_offsets(obs_now[:, [0, 1]])
            xz_obs_sc.set_offsets(obs_now[:, [0, 2]])
            yz_obs_sc.set_offsets(obs_now[:, [1, 2]])
        else:
            xy_obs_sc.set_offsets(np.zeros((0, 2)))
            xz_obs_sc.set_offsets(np.zeros((0, 2)))
            yz_obs_sc.set_offsets(np.zeros((0, 2)))

        if pred_i_vis.size:
            xy_pred_sc.set_offsets(pred_i_vis[:, [0, 1]])
            xz_pred_sc.set_offsets(pred_i_vis[:, [0, 2]])
            yz_pred_sc.set_offsets(pred_i_vis[:, [1, 2]])
        else:
            xy_pred_sc.set_offsets(np.zeros((0, 2)))
            xz_pred_sc.set_offsets(np.zeros((0, 2)))
            yz_pred_sc.set_offsets(np.zeros((0, 2)))

        # --- update iso-surfaces ---
        if true_mesh is not None:
            true_mesh.remove()
            true_mesh = None
        if cp_mesh is not None:
            cp_mesh.remove()
            cp_mesh = None

        s = int(max(1, mc_stride))
        cp_ds = cp_unsafe[::s, ::s, ::s]

        xs_ds = xs[np.arange(0, xs.size, s)]
        ys_ds = ys[np.arange(0, ys.size, s)]
        zs_ds = zs[np.arange(0, zs.size, s)]

        cp_mesh = _draw_isosurface_from_bool(
            ax3d, cp_ds, xs_ds, ys_ds, zs_ds,
            alpha=0.18, color_rgb=(1.0, 0.2, 0.2)
        )

        if true_unsafe is not None:
            true_ds = true_unsafe[::s, ::s, ::s]
            true_mesh = _draw_isosurface_from_bool(
                ax3d, true_ds, xs_ds, ys_ds, zs_ds,
                alpha=0.25, color_rgb=(0.2, 0.4, 1.0)
            )

        # --- update projection contours (φ=0) ---
        phi_cp = D_lower - safe_rad
        phi_true = (D_true - safe_rad) if D_true is not None else None

        phi_cp_xy = np.min(phi_cp, axis=0)
        phi_cp_xz = np.min(phi_cp, axis=1)
        phi_cp_yz = np.min(phi_cp, axis=2)

        phi_true_xy = np.min(phi_true, axis=0) if phi_true is not None else None
        phi_true_xz = np.min(phi_true, axis=1) if phi_true is not None else None
        phi_true_yz = np.min(phi_true, axis=2) if phi_true is not None else None

        _clear_contours(xy_contours)
        _clear_contours(xz_contours)
        _clear_contours(yz_contours)

        xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_cp_xy, levels=[0.0], linewidths=2.0, colors=[(1.0, 0.2, 0.2)]))
        xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_cp_xz, levels=[0.0], linewidths=2.0, colors=[(1.0, 0.2, 0.2)]))
        yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_cp_yz, levels=[0.0], linewidths=2.0, colors=[(1.0, 0.2, 0.2)]))

        if phi_true_xy is not None:
            xy_contours.append(ax_xy.contour(Xxy, Yxy, phi_true_xy, levels=[0.0], linewidths=2.0, colors=[(0.2, 0.4, 1.0)]))
        if phi_true_xz is not None:
            xz_contours.append(ax_xz.contour(Xxz, Zxz, phi_true_xz, levels=[0.0], linewidths=2.0, colors=[(0.2, 0.4, 1.0)]))
        if phi_true_yz is not None:
            yz_contours.append(ax_yz.contour(Yyz, Zyz, phi_true_yz, levels=[0.0], linewidths=2.0, colors=[(0.2, 0.4, 1.0)]))

        ax3d.set_title(
            f"3D iso-surfaces (unsafe): CP=red, True=blue | step={k} | feasible={feasible}"
        )

        ax_txt.clear()
        ax_txt.axis("off")
        ax_txt.text(
            0.02, 0.98,
            "\n".join([
                f"step = {k}",
                f"i_view = {i_view}",
                f"dist_goal = {dist_goal:.3f} (finish<= {goal_finish_dist})",
                f"feasible = {feasible}",
                f"v_xy = {v_xy:.3f}, w = {w:.3f}, vz = {vz:.3f}",
                f"collisions = {collision_count} (dmin_now={dmin_now:.3f}, r_safe={safe_rad:.3f})",
                f"infeasible = {infeasible_count}",
                f"coverage (containment) = {cov_running:.2%}  (eval_steps={eval_steps})",
                "",
                "Legend:",
                "  3D iso-surface red  : CP unsafe set (visualization)",
                "  3D iso-surface blue : True unsafe set (visualization)",
                "  controller          : ECP-MPC (online calibrated)",
            ]),
            va="top", ha="left", fontsize=10
        )

        # step env
        obs = env.step(v_xy, w, vz)
        return []

    anim = FuncAnimation(fig, upd, frames=max_steps, interval=80, blit=False)
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
        max_steps=250,
        i_view=10,
        n_calib_samples=120,
        goal_finish_dist=0.8,
        mc_stride=2,
    )