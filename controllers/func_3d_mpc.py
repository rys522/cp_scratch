from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import time
from cp.functional_cp import CPStepParameters


@dataclass
class MPC3DWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001


class FunctionalCPMPC3D:

    def __init__(
        self,
        *,
        cp_params: List[CPStepParameters],
        xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
        n_steps: int,
        dt: float,
        n_skip: int,
        robot_rad: float,
        obstacle_rad: float,
        v_lim: Tuple[float, float],
        vz_lim: Tuple[float, float],
        yaw_rate_lim: Tuple[float, float],
        n_paths: int,
        seed: int = 0,
        weights: Optional[MPC3DWeights] = None,
        CP: bool = True,
        default_U: float = 1.0,
        endpoint_sigma: float = 1.0,
    ):
        self.params: Dict[int, CPStepParameters] = {int(p.t_idx): p for p in cp_params}

        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        self.zs = np.asarray(zs, dtype=np.float32)
        self.nx, self.ny, self.nz = self.xs.size, self.ys.size, self.zs.size

        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.vmin, self.vmax = map(float, v_lim)
        self.wmin, self.wmax = map(float, yaw_rate_lim)
        self.vzmin, self.vzmax = map(float, vz_lim)

        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.weights = weights or MPC3DWeights()
        self.last_best_vels: Optional[np.ndarray] = None
        self.CP = bool(CP)

        self.default_U = float(default_U)
        self.endpoint_sigma = float(endpoint_sigma)

        self.hold_beta = 0.35 
        self.hold_vel_eps = 1e-2
        self.hold_active = False
        self.hold_pos: Optional[np.ndarray] = None

        self.last_cmd_vel_env = np.zeros(4, dtype=np.float32)

    def _world_to_grid_ijk_float_batch(self, pos_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized version of _world_to_grid_ijk_float for N points.

        Returns:
          ijk_float: (N,3) with (kf, jf, if_) = (k0+tz, j0+ty, i0+tx)
          valid:     (N,)  True if inside bounds
        """
        pos_world = np.asarray(pos_world, dtype=np.float32)
        x = pos_world[:, 0]
        y = pos_world[:, 1]
        z = pos_world[:, 2]

        valid = (
            (self.xs[0] <= x) & (x <= self.xs[-1]) &
            (self.ys[0] <= y) & (y <= self.ys[-1]) &
            (self.zs[0] <= z) & (z <= self.zs[-1])
        )

        # searchsorted vectorized
        i1 = np.searchsorted(self.xs, x, side="left")
        j1 = np.searchsorted(self.ys, y, side="left")
        k1 = np.searchsorted(self.zs, z, side="left")

        i1 = np.clip(i1, 1, self.nx - 1)
        j1 = np.clip(j1, 1, self.ny - 1)
        k1 = np.clip(k1, 1, self.nz - 1)

        i0 = i1 - 1
        j0 = j1 - 1
        k0 = k1 - 1

        x0 = self.xs[i0]; x1 = self.xs[i1]
        y0 = self.ys[j0]; y1 = self.ys[j1]
        z0 = self.zs[k0]; z1 = self.zs[k1]

        # tx,ty,tz
        tx = np.where(x1 == x0, 0.0, (x - x0) / (x1 - x0))
        ty = np.where(y1 == y0, 0.0, (y - y0) / (y1 - y0))
        tz = np.where(z1 == z0, 0.0, (z - z0) / (z1 - z0))

        if_ = i0.astype(np.float32) + tx.astype(np.float32)
        jf  = j0.astype(np.float32) + ty.astype(np.float32)
        kf  = k0.astype(np.float32) + tz.astype(np.float32)

        ijk_float = np.stack([kf, jf, if_], axis=1)  # (N,3)
        return ijk_float, valid

    def _grid_flat_index_batch(self, ijk_float: np.ndarray) -> np.ndarray:
        """
        Vectorized version of _grid_flat_index for N points.

        Input:
          ijk_float: (N,3) with (kf,jf,if_)
        Return:
          idx: (N,) flat indices
        """
        ijk_float = np.asarray(ijk_float, dtype=np.float32)
        kf = ijk_float[:, 0]
        jf = ijk_float[:, 1]
        if_ = ijk_float[:, 2]

        k = np.clip(np.rint(kf).astype(np.int64), 0, self.nz - 1)
        j = np.clip(np.rint(jf).astype(np.int64), 0, self.ny - 1)
        i = np.clip(np.rint(if_).astype(np.int64), 0, self.nx - 1)

        idx = (k * self.ny + j) * self.nx + i
        return idx.astype(np.int64)

    def evaluate_U_batch(self, pos_world: np.ndarray, t_idx: int) -> np.ndarray:
        """
        pos_world: (N,3)
        returns:
          U: (N,) float32
        """
        pos_world = np.asarray(pos_world, dtype=np.float32)
        N = pos_world.shape[0]

        p = self.params.get(int(t_idx))
        if p is None:
            return np.full((N,), self.default_U, dtype=np.float32)

        ijk_float, valid = self._world_to_grid_ijk_float_batch(pos_world)
        idx = self._grid_flat_index_batch(ijk_float)  # (N,)

        # phi: (D,N)
        phi = p.phi_basis[:, idx].astype(np.float32, copy=False)

        # centers: (K,N)
        mus = np.asarray(p.mus, dtype=np.float32)
        centers = mus @ phi

        # quads: (K,N) where quad_k(n) = phi_n^T Sigma_k phi_n
        sigmas = np.asarray(p.sigmas, dtype=np.float32)
        tmp = np.einsum("kij,jn->kin", sigmas, phi, optimize=True)   # (K,D,N)
        quads = np.einsum("in,kin->kn", phi, tmp, optimize=True)     # (K,N)

        rks = np.asarray(p.rks, dtype=np.float32).reshape(-1, 1)     # (K,1)
        upper = centers + rks * np.sqrt(np.maximum(quads, 0.0))      # (K,N)

        U = (float(p.epsilon) + np.max(upper, axis=0)).astype(np.float32)  # (N,)

        # out-of-bounds -> default_U (match your old scalar behavior)
        if not np.all(valid):
            U = U.copy()
            U[~valid] = self.default_U
        return U


    def _solve_quintic_coeffs(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T):
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        c0 = start_pos
        c1 = start_vel
        c2 = 0.5 * start_acc

        H = end_pos - c0 - c1 * T - c2 * T2
        V = end_vel - c1 - 2.0 * c2 * T
        A = end_acc - 2.0 * c2

        c3 = (10.0 * H - 4.0 * V * T + 0.5 * A * T2) / T3
        c4 = (-15.0 * H + 7.0 * V * T - 1.0 * A * T2) / T4
        c5 = (6.0 * H - 3.0 * V * T + 0.5 * A * T2) / T5
        return c0, c1, c2, c3, c4, c5

    def generate_paths_trajectory(
        self,
        x0yz: np.ndarray,
        yaw0: float,
        current_vel: Optional[np.ndarray],
        goal_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0yz = np.asarray(x0yz, dtype=np.float32).reshape(3,)
        goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)

        start_vel = (np.asarray(current_vel, dtype=np.float32).reshape(3,) * 0.8) if current_vel is not None else np.zeros(3, dtype=np.float32)
        start_acc = np.zeros(3, dtype=np.float32)
        current_speed = float(np.linalg.norm(start_vel))

        dist_to_goal = float(np.linalg.norm(goal_xyz - x0yz))
        T_horizon = self.n_steps * self.dt

        if dist_to_goal > 1e-3:
            dir_vec = (goal_xyz - x0yz) / dist_to_goal
        else:
            dir_vec = np.zeros(3, dtype=np.float32)

        max_reachable_dist = self.vmax * T_horizon
        if dist_to_goal < max_reachable_dist:
            center_pt = goal_xyz
        else:
            center_pt = x0yz + dir_vec * max_reachable_dist

        sigma = self.endpoint_sigma
        term_pos = self.rng.normal(loc=center_pt, scale=sigma, size=(self.n_paths, 3)).astype(np.float32)
        term_pos[0] = center_pt

        diff = term_pos - x0yz[None, :]
        dists = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        directions = diff / dists

        low_speed = min(current_speed, self.vmax * 0.5)
        sampled_speeds = self.rng.uniform(low=low_speed, high=self.vmax, size=(self.n_paths, 1)).astype(np.float32)
        sampled_speeds[0] = self.vmax
        term_vel = directions * sampled_speeds
        term_acc = np.zeros_like(term_pos)

        c0, c1, c2, c3, c4, c5 = self._solve_quintic_coeffs(
            x0yz[None, :], start_vel[None, :], start_acc[None, :],
            term_pos, term_vel, term_acc, T_horizon
        )

        t_eval = np.linspace(0.0, T_horizon, self.n_steps + 1, dtype=np.float32)
        tt = t_eval[:, None, None]

        paths_t = c0 + c1*tt + c2*tt**2 + c3*tt**3 + c4*tt**4 + c5*tt**5
        paths = np.transpose(paths_t, (1, 0, 2))  # (P,T+1,3)

        vels_t = c1 + 2*c2*tt + 3*c3*tt**2 + 4*c4*tt**3 + 5*c5*tt**4
        vels_xyz = np.transpose(vels_t, (1, 0, 2))  # (P,T+1,3)

        # Output: [vx, vy, w, vz] (keep your original internal order)
        out_vels = np.zeros((self.n_paths, self.n_steps, 4), dtype=np.float32)

        raw_vx = vels_xyz[:, :-1, 0]
        raw_vy = vels_xyz[:, :-1, 1]
        raw_vz = vels_xyz[:, :-1, 2]

        clamped_vz = np.clip(raw_vz, self.vzmin, self.vzmax)

        v_xy_norm = np.sqrt(raw_vx**2 + raw_vy**2)
        scale_factor = np.minimum(1.0, self.vmax / (v_xy_norm + 1e-6))
        clamped_vx = raw_vx * scale_factor
        clamped_vy = raw_vy * scale_factor

        out_vels[..., 0] = clamped_vx
        out_vels[..., 1] = clamped_vy
        out_vels[..., 3] = clamped_vz

        full_vx = vels_xyz[..., 0]
        full_vy = vels_xyz[..., 1]
        headings = np.arctan2(full_vy, full_vx)
        headings = np.unwrap(headings, axis=1)

        w_all = np.diff(headings, axis=1) / self.dt
        speed_sq = full_vx[:, :-1]**2 + full_vy[:, :-1]**2
        moving_mask = speed_sq > (0.1**2)

        w_final = np.where(moving_mask, w_all, 0.0)
        w_final = np.clip(w_final, self.wmin, self.wmax)
        out_vels[..., 2] = w_final

        return paths, out_vels
    
    def aabb_contains_points(self, p: np.ndarray, box, inflate: float):
        xmin, xmax, ymin, ymax, zmin, zmax = box
        xmin -= inflate; xmax += inflate
        ymin -= inflate; ymax += inflate
        zmin -= inflate; zmax += inflate
        return (
            (p[..., 0] >= xmin) & (p[..., 0] <= xmax) &
            (p[..., 1] >= ymin) & (p[..., 1] <= ymax) &
            (p[..., 2] >= zmin) & (p[..., 2] <= zmax)
        )

    def filter_unsafe_paths(
        self,
        paths: np.ndarray,
        vels: np.ndarray,
        boxes_3d: List[Any],
        pred_xyz: np.ndarray,
        pred_mask: np.ndarray,
    ):
        P, T1, _ = paths.shape
        unsafe = np.zeros(P, dtype=bool)
        pts = paths[:, 1:, :]  # (P,T,3)

        # ---- (1) Static safety (AABB) ----
        if boxes_3d:
            for box in boxes_3d:
                inside = self.aabb_contains_points(pts, box, inflate=self.robot_rad)  # (P,T)
                unsafe |= np.any(inside, axis=1)
                if np.all(unsafe):
                    break

        # ---- (2) Dynamic safety (CP-buffer) ----
        if pred_xyz is None or pred_mask is None:
            safe_mask = ~unsafe
            if not np.any(safe_mask):
                return None, None
            return paths[safe_mask], vels[safe_mask]

        T = min(T1 - 1, int(pred_xyz.shape[0]))
        for t in range(T):
            if np.all(unsafe):
                break

            check_idxs = np.where(~unsafe)[0]
            if check_idxs.size == 0:
                continue

            # valid obstacles at this time
            mask_t = np.asarray(pred_mask[t], dtype=bool)
            if not np.any(mask_t):
                continue
            obs_pts = np.asarray(pred_xyz[t], dtype=np.float32)[mask_t]  # (M,3)
            if obs_pts.shape[0] == 0:
                continue

            x_t = paths[check_idxs, t + 1, :].astype(np.float32, copy=False)  # (N,3)

            # nearest obstacle distance
            dists = np.min(np.linalg.norm(x_t[:, None, :] - obs_pts[None, :, :], axis=-1), axis=1).astype(np.float32)

            if self.CP:
                U_vals = self.evaluate_U_batch(x_t, t)  # (N,)
            else:
                U_vals = np.zeros((x_t.shape[0],), dtype=np.float32)

            unsafe_t = (dists - U_vals) < self.safe_rad
            unsafe[check_idxs] |= unsafe_t

        safe_mask = ~unsafe
        if not np.any(safe_mask):
            return None, None
        return paths[safe_mask], vels[safe_mask]

    def score_paths(self, paths: np.ndarray, vels: np.ndarray, goal_xyz: np.ndarray):
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)

        term_cost = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        inter_cost = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-1, -2))
        ctrl_cost = self.weights.w_control * np.sum(vels ** 2, axis=(-1, -2))

        total_cost = term_cost + inter_cost + ctrl_cost
        best_idx = int(np.argmin(total_cost))
        return best_idx, float(total_cost[best_idx])
    
    def _hold_action(self, robot_xyz: np.ndarray):
        if (not self.hold_active) or (self.hold_pos is None):
            self.hold_active = True
            self.hold_pos = np.asarray(robot_xyz, dtype=np.float32).reshape(3,).copy()
            self.hold_step = 0  # retry counter reset

        beta = float(self.hold_beta)

        self.last_cmd_vel_env *= (1.0 - beta)

        v = self.last_cmd_vel_env
        if (abs(float(v[0])) + abs(float(v[1])) + abs(float(v[2]))) < self.hold_vel_eps:
            v[:3] = 0.0
        if abs(float(v[3])) < self.hold_vel_eps:
            v[3] = 0.0

        act = (self.hold_pos, v)
        info = {"feasible": False, "hold_active": True, "target_pos": self.hold_pos, "target_vel": v}
        return act, info

    # ============================================================
    # Public API
    # ============================================================
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        pred_xyz: np.ndarray,
        pred_mask: np.ndarray,
        boxes_3d: Optional[List[Any]] = None,
        robot_vel: Optional[np.ndarray] = None,
    ):
        boxes_3d = boxes_3d or []
        t0 = time.perf_counter()

        robot_xyz = np.asarray(robot_xyz, dtype=np.float32).reshape(3,)
        goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)
        robot_vel = np.asarray(robot_vel, dtype=np.float32).reshape(3,) if robot_vel is not None else np.zeros(3, dtype=np.float32)

        # 1) Generate
        paths, vels = self.generate_paths_trajectory(robot_xyz, float(robot_yaw), robot_vel, goal_xyz)

        # 2) Filter
        safe_paths, safe_vels = self.filter_unsafe_paths(paths, vels, boxes_3d, pred_xyz, pred_mask)
        if safe_paths is None:
            
            act, info_hold = self._hold_action(robot_xyz)
            info_hold["timing_ms"] = (time.perf_counter() - t0) * 1000.0
            info_hold["final_path"] = None
            return act, info_hold
            
            # return None, {"feasible": False, "final_path": None, "timing_ms": (time.perf_counter() - t0) * 1000.0}

        
        # 3) Score
        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, goal_xyz)

        best_path = safe_paths[best_idx]
        best_vel_traj = safe_vels[best_idx]
        self.last_best_vels = best_vel_traj.copy()

        next_pos = best_path[1]

        # internal vel format: [vx, vy, w, vz]
        cmd_vx, cmd_vy, cmd_w, cmd_vz = best_vel_traj[1]

        cmd_vx = float(np.clip(cmd_vx, self.vmin, self.vmax))
        cmd_vy = float(np.clip(cmd_vy, self.vmin, self.vmax))
        cmd_vz = float(np.clip(cmd_vz, self.vzmin, self.vzmax))
        cmd_w  = float(np.clip(cmd_w,  self.wmin, self.wmax))

        # Env expects: [vx, vy, vz, yaw_rate]
        target_vel_env = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_w], dtype=np.float32)

        self.hold_active = False
        self.hold_pos = None

        self.last_cmd_vel_env = target_vel_env.copy()

        info = {
            "feasible": True,
            "final_path": best_path,
            "cost": best_cost,
            "timing_ms": (time.perf_counter() - t0) * 1000.0,
            "target_pos": next_pos,
            "target_vel": target_vel_env,
        }
        act = (next_pos, target_vel_env)
        return act, info