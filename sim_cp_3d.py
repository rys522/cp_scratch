from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from quad_env import QuadWorldEnv3D
from controllers.cp_3d_mpc import ConformalController3D

# ----------------------------
# Constants
# ----------------------------
ROBOT_RAD = 0.1
OBSTACLE_RAD = 0.2

MAX_LINEAR_VEL = 3.0
MAX_VZ = 0.7
MAX_ANGULAR_Z = 0.7
MIN_ANGULAR_Z = -0.7


# ----------------------------
# Helpers
# ----------------------------
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


def _set_equal_aspect_3d(ax, xlim, ylim, zlim):
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    r = max(x1 - x0, y1 - y0, z1 - z0) / 2
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def _get_obs_positions_from_history(obs_dict) -> np.ndarray:
    h_dict = obs_dict.get("history", {})
    if not h_dict:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([traj[-1] for traj in h_dict.values()], dtype=np.float32)


# ----------------------------
# Main
# ----------------------------
def run_one_episode_visual_3d_simple(
    env: QuadWorldEnv3D,
    *,
    time_horizon: int = 12,
    n_skip: int = 4,
    max_steps: int = 250,
    goal_finish_dist: float = 0.3,
    view_elev: float = 22.0,
    view_azim: float = -55.0,
    dpi: int = 140,
    save: bool = False,
    save_path: str = "quad_cc_3d_simple.mp4",
):
    # ----------------------------
    # Controller
    # ----------------------------
    ctrl = ConformalController3D(
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        v_xy_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
        vz_lim=(-MAX_VZ, MAX_VZ),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        w_terminal=10.0,
        w_intermediate=1.0,
        w_control=0.001,
        use_dynamic=True,

    )

    # simple wall boxes
    xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
    margin = 5.0
    cov_min, cov_max = -50.0, 50.0
    x_min, x_max = xlim
    y_min, y_max = ylim
    z_min, z_max = zlim
    wall_boxes = [
        [x_min - margin, x_min, cov_min, cov_max, cov_min, cov_max],
        [x_max, x_max + margin, cov_min, cov_max, cov_min, cov_max],
        [cov_min, cov_max, y_min - margin, y_min, cov_min, cov_max],
        [cov_min, cov_max, y_max, y_max + margin, cov_min, cov_max],
        [cov_min, cov_max, cov_min, cov_max, z_min - margin, z_min],
        [cov_min, cov_max, cov_min, cov_max, z_max, z_max + margin],
    ]

    # ----------------------------
    # Record Phase
    # ----------------------------
    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    episode_history = []
    timing = {"ctrl_ms": [], "step_ms": [], "loop_ms": []}

    vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0

    for k in range(max_steps):
        t_loop0 = time.perf_counter()

        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])

        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            print(f"Goal reached at step {k}")
            break

        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)

        # ---- controller timing ----
        t0 = time.perf_counter()
        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            boxes_3d=wall_boxes,
        )

        t1 = time.perf_counter()

        feasible = bool(info.get("feasible", False)) and (act is not None)

        if not feasible:
            target_pos = robot.copy()
            vx_global, vy_global, vz_global, yaw_rate = 0.0, 0.0, 0.0, 0.0
            act_to_store = None
        else:
            target_pos, target_vel = act
            target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3,)
            target_vel = np.asarray(target_vel, dtype=np.float32).reshape(4,)
            vx_global, vy_global, vz_global, yaw_rate = map(float, target_vel)
            act_to_store = (target_pos.copy(), target_vel.copy())

        # ---- env.step timing ----
        t2 = time.perf_counter()
        obs = env.step(target_pos, (vx_global, vy_global, vz_global, yaw_rate))
        t3 = time.perf_counter()

        ctrl_ms = (t1 - t0) * 1000.0
        step_ms = (t3 - t2) * 1000.0
        loop_ms = (t3 - t_loop0) * 1000.0
        timing["ctrl_ms"].append(ctrl_ms)
        timing["step_ms"].append(step_ms)
        timing["loop_ms"].append(loop_ms)

        episode_history.append(
            {
                "step": k,
                "obs": _deepcopy_obs_dict(obs),
                "robot": robot,
                "act": act_to_store,
                "feasible": feasible,
                "timing": {"ctrl_ms": ctrl_ms, "step_ms": step_ms, "loop_ms": loop_ms},
            }
        )

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
    _summ(timing["ctrl_ms"], "controller")
    _summ(timing["step_ms"], "env.step (physics)")
    _summ(timing["loop_ms"], "total loop")
    print("===================================\n")

    print(f"Simulation finished. Total frames: {total_frames}. Starting Visualization...")

    # ----------------------------
    # Visualization
    # ----------------------------
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax3d = fig.add_subplot(111, projection="3d")

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=view_elev, azim=view_azim)
    _set_equal_aspect_3d(ax3d, xlim, ylim, zlim)
    ax3d.grid(False)

    # goal marker
    ax3d.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=160, edgecolors="k", linewidths=0.4)

    # artists
    robot_dot, = ax3d.plot([], [], [], marker="o", markersize=8, linewidth=0)
    robot_line, = ax3d.plot([], [], [], linewidth=2.5, alpha=0.9)
    obs_sc = ax3d.scatter([], [], [], s=18, alpha=0.8)

    robot_traj: List[np.ndarray] = []

    safe_rad = ROBOT_RAD + OBSTACLE_RAD
    collision_count = 0

    info_txt = ax3d.text2D(
        0.02, 0.98, "",
        transform=ax3d.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )

    def upd(k: int):
        nonlocal collision_count

        if k >= total_frames:
            return []

        data = episode_history[k]
        obs_k = data["obs"]

        robot = np.asarray(data["robot"], dtype=np.float32).reshape(3,)
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj, dtype=np.float32)

        obs_now = _get_obs_positions_from_history(obs_k)  # (N,3)

        # ---- collision count (for reporting only) ----
        if obs_now.size:
            dmin_now = float(np.min(np.linalg.norm(obs_now - robot[None, :], axis=1)))
        else:
            dmin_now = float("inf")
        if dmin_now < safe_rad:
            collision_count += 1

        # ---- action (cmd) ----
        act = data.get("act", None)
        if act is None:
            pos_cmd = robot
            vel_cmd = np.zeros((4,), dtype=np.float32)
        else:
            pos_cmd, vel_cmd = act
            pos_cmd = np.asarray(pos_cmd, dtype=np.float32).reshape(3,)
            vel_cmd = np.asarray(vel_cmd, dtype=np.float32).reshape(4,)

        robot_vel = np.asarray(obs_k.get("robot_vel", [0, 0, 0]), dtype=np.float32).reshape(3,)
        dist_goal = float(np.linalg.norm(robot - goal))

        tinfo = data["timing"]
        ctrl_ms = tinfo["ctrl_ms"]
        step_ms = tinfo["step_ms"]
        loop_ms = tinfo["loop_ms"]

        # update artists
        robot_dot.set_data([robot[0]], [robot[1]])
        robot_dot.set_3d_properties([robot[2]])
        robot_line.set_data(tr[:, 0], tr[:, 1])
        robot_line.set_3d_properties(tr[:, 2])
        obs_sc._offsets3d = (obs_now[:, 0], obs_now[:, 1], obs_now[:, 2]) if obs_now.size else ([], [], [])

        ax3d.set_title(f"step={k} | feasible={data['feasible']}", fontsize=11)

        info_txt.set_text(
            "\n".join([
                f"step: {k}/{total_frames-1}",
                f"feasible: {data['feasible']}",
                f"dist_goal: {dist_goal:.3f} (finish<= {goal_finish_dist})",
                "",
                f"pos_cmd: ({pos_cmd[0]:.2f}, {pos_cmd[1]:.2f}, {pos_cmd[2]:.2f})",
                f"vel_cmd: (vx={vel_cmd[0]:.2f}, vy={vel_cmd[1]:.2f}, vz={vel_cmd[2]:.2f}, yaw_rate={vel_cmd[3]:.2f})",
                f"robot_pos: ({robot[0]:.2f}, {robot[1]:.2f}, {robot[2]:.2f})",
                f"robot_vel: (vx={robot_vel[0]:.2f}, vy={robot_vel[1]:.2f}, vz={robot_vel[2]:.2f})",
                "",
                f"collisions: {collision_count} (dmin_now={dmin_now:.3f}, r_safe={safe_rad:.3f})",
                "",
                f"timing(ms): ctrl={ctrl_ms:.2f} | step={step_ms:.2f} | loop={loop_ms:.2f}",
            ])
        )

        return []

    anim = FuncAnimation(fig, upd, frames=total_frames, interval=80, blit=False, repeat=False)
    plt.tight_layout()

    if save:
        anim.save(
            save_path,
            writer="ffmpeg",
            fps=12,
            dpi=240,
            bitrate=20000,
        )
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    env = QuadWorldEnv3D(
        dt=0.1,
        horizon=20,
        n_obs=280,
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
        gui=False,
    )

    run_one_episode_visual_3d_simple(
        env,
        time_horizon=12,
        n_skip=4,
        max_steps=250,
        goal_finish_dist=0.3,
        view_elev=22.0,
        view_azim=-55.0,
        dpi=140,
        save=False,
        save_path="quad_cc_3d_simple.mp4",
    )