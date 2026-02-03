import numpy as np
import time
from typing import Optional, List, Any, Tuple


class ConformalController3D:

    def __init__(
        self,
        n_steps: int = 20,
        dt: float = 0.1,
        v_xy_lim: Tuple[float, float] = (-0.8, 0.8),
        vz_lim: Tuple[float, float] = (-0.4, 0.4),
        yaw_rate_lim: Tuple[float, float] = (-0.7, 0.7),
        n_skip: int = 4,  # kept for API parity; not used in quintic eval directly
        conformal_control_variable: float = 1.0,   # lambda (avoidance shaping)
        risk_level: float = -1.0,                  # epsilon (lambda update rule)
        step_size: float = 10.0,                 # eta (lambda update rule)
        robot_rad: float = 0.4,
        obstacle_rad: float = 1.0 / np.sqrt(2.0),
        w_terminal: float = 10.0,
        w_intermediate: float = 1.0,
        w_control: float = 0.001,
        use_dynamic: bool = True,
        n_paths: int = 512,
        seed: int = 0,
        endpoint_sigma: float = 1.0,
    ):
        self._n_steps = int(n_steps)
        self._dt = float(dt)
        self.n_skip = int(n_skip)

        self.vxy_min, self.vxy_max = map(float, v_xy_lim)
        self.vz_min, self.vz_max = map(float, vz_lim)
        self.w_min, self.w_max = map(float, yaw_rate_lim)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.w_terminal = float(w_terminal)
        self.w_intermediate = float(w_intermediate)
        self.w_control = float(w_control)

        self.use_dynamic = bool(use_dynamic)

        assert conformal_control_variable > 0.0
        self._lambda = float(conformal_control_variable)
        self._epsilon = float(risk_level)
        self._eta = float(step_size)

        # quintic path sampling
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.endpoint_sigma = float(endpoint_sigma)

    # ============================================================
    # Quintic trajectory generation
    # ============================================================
    @staticmethod
    def _solve_quintic_coeffs(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T):
        """
        Vectorized quintic boundary solver.
        p(t)=c0+c1 t+c2 t^2+c3 t^3+c4 t^4+c5 t^5
        Returns c0..c5 with shape (N,3)
        """
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
        current_vel_xyz: Optional[np.ndarray],
        goal_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          paths: (P, T+1, 3)
          vels:  (P, T,   4) with columns [vx, vy, vz, yaw_rate] (ENV ORDER)
        """
        x0yz = np.asarray(x0yz, dtype=np.float32).reshape(3,)
        goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)

        # Start boundary
        if current_vel_xyz is None:
            start_vel = np.zeros(3, dtype=np.float32)
        else:
            start_vel = np.asarray(current_vel_xyz, dtype=np.float32).reshape(3,) * 0.8
        start_acc = np.zeros(3, dtype=np.float32)

        T_horizon = self._n_steps * self._dt

        # pick center point (reachable)
        diff_g = goal_xyz - x0yz
        dist_to_goal = float(np.linalg.norm(diff_g))
        if dist_to_goal > 1e-6:
            dir_vec = diff_g / dist_to_goal
        else:
            dir_vec = np.zeros(3, dtype=np.float32)

        vmax_xy = self.vxy_max
        max_reachable_dist = vmax_xy * T_horizon
        if dist_to_goal <= max_reachable_dist:
            center_pt = goal_xyz
        else:
            center_pt = x0yz + dir_vec * max_reachable_dist

        # sample endpoints around center
        term_pos = self.rng.normal(loc=center_pt, scale=self.endpoint_sigma, size=(self.n_paths, 3)).astype(np.float32)
        term_pos[0] = center_pt  # ensure one "direct" candidate

        # choose terminal velocity toward endpoint with random speed
        diff = term_pos - x0yz[None, :]
        dists = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        directions = diff / dists

        current_speed = float(np.linalg.norm(start_vel))
        low_speed = min(current_speed, vmax_xy * 0.5)
        sampled_speeds = self.rng.uniform(low=low_speed, high=vmax_xy, size=(self.n_paths, 1)).astype(np.float32)
        sampled_speeds[0] = vmax_xy
        term_vel = directions * sampled_speeds

        term_acc = np.zeros_like(term_pos)

        # solve quintic
        c0, c1, c2, c3, c4, c5 = self._solve_quintic_coeffs(
            x0yz[None, :], start_vel[None, :], start_acc[None, :],
            term_pos, term_vel, term_acc, T_horizon
        )

        # evaluate
        t_eval = np.linspace(0.0, T_horizon, self._n_steps + 1, dtype=np.float32)
        tt = t_eval[:, None, None]

        paths_t = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4 + c5 * tt**5
        paths = np.transpose(paths_t, (1, 0, 2))  # (P,T+1,3)

        vels_t = c1 + 2*c2*tt + 3*c3*tt**2 + 4*c4*tt**3 + 5*c5*tt**4
        vels_xyz = np.transpose(vels_t, (1, 0, 2))  # (P,T+1,3)

        # env-ordered vels: [vx,vy,vz,yaw_rate]
        out_vels = np.zeros((self.n_paths, self._n_steps, 4), dtype=np.float32)

        raw_vx = vels_xyz[:, :-1, 0]
        raw_vy = vels_xyz[:, :-1, 1]
        raw_vz = vels_xyz[:, :-1, 2]

        # clamp vz
        out_vels[..., 2] = np.clip(raw_vz, self.vz_min, self.vz_max)

        # clamp xy speed to vmax
        v_xy_norm = np.sqrt(raw_vx**2 + raw_vy**2)
        scale = np.minimum(1.0, self.vxy_max / (v_xy_norm + 1e-6))
        out_vels[..., 0] = raw_vx * scale
        out_vels[..., 1] = raw_vy * scale

        # yaw-rate from heading (deadband)
        full_vx = vels_xyz[..., 0]
        full_vy = vels_xyz[..., 1]
        headings = np.arctan2(full_vy, full_vx)
        headings = np.unwrap(headings, axis=1)
        w_all = np.diff(headings, axis=1) / self._dt  # (P,T)

        speed_sq = full_vx[:, :-1]**2 + full_vy[:, :-1]**2
        moving_mask = speed_sq > (0.1**2)
        w_final = np.where(moving_mask, w_all, 0.0)
        out_vels[..., 3] = np.clip(w_final, self.w_min, self.w_max)

        return paths, out_vels

    # ============================================================
    # Safety filtering
    # ============================================================
    @staticmethod
    def aabb_contains_points(p: np.ndarray, box, inflate: float):
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
        *,
        boxes_3d: List[Any],
        pred_xyz: Optional[np.ndarray],   # kept for signature compatibility; ignored here
        pred_mask: Optional[np.ndarray],  # kept for signature compatibility; ignored here
    ):
        P, T1, _ = paths.shape
        unsafe = np.zeros(P, dtype=bool)
        pts = paths[:, 1:, :]  # (P,T,3)

        # Static AABB only
        if boxes_3d:
            for box in boxes_3d:
                inside = self.aabb_contains_points(pts, box, inflate=self.robot_rad)
                unsafe |= np.any(inside, axis=1)
                if np.all(unsafe):
                    break

        safe_mask = ~unsafe
        if not np.any(safe_mask):
            return None, None
        return paths[safe_mask], vels[safe_mask]

    # ============================================================
    # Scoring
    # ============================================================
    def score_paths(
        self,
        paths: np.ndarray,
        vels: np.ndarray,
        goal_xyz: np.ndarray,
        pred_xyz: Optional[np.ndarray],
        pred_mask: Optional[np.ndarray],
    ):
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)

        inter_cost = self.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        ctrl_cost  = self.w_control * np.sum(vels ** 2, axis=(-2, -1))
        term_cost  = self.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        total_cost = inter_cost + ctrl_cost + term_cost

        avoidance = 0.0
        if self.use_dynamic and (pred_xyz is not None) and (pred_mask is not None):
            Tp = min(paths.shape[1] - 1, pred_xyz.shape[0])
            Pn = paths.shape[0]
            min_distances = np.zeros((Pn, Tp), dtype=np.float32)

            for t in range(Tp):
                mask_t = np.asarray(pred_mask[t], dtype=bool)
                if not np.any(mask_t):
                    continue
                obs = np.asarray(pred_xyz[t], dtype=np.float32)[mask_t]  # (M,3)
                dist = np.linalg.norm(paths[:, t + 1, None, :] - obs[None, :, :], axis=-1)
                min_distances[:, t] = np.min(dist, axis=1).astype(np.float32)

            # lambda-scaled avoidance shaping (reward being far)
            avoidance = -self._lambda * np.sum(min_distances, axis=1)

        score = total_cost + avoidance
        best = int(np.argmin(score))
        return best, float(total_cost[best])

    # ============================================================
    # Public API
    # ============================================================
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        pred_xyz: Optional[np.ndarray] = None,
        pred_mask: Optional[np.ndarray] = None,
        boxes_3d: Optional[List[Any]] = None,
        robot_vel: Optional[np.ndarray] = None,
    ):
        boxes_3d = boxes_3d or []
        robot_xyz = np.asarray(robot_xyz, dtype=np.float32).reshape(3,)
        goal_xyz  = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)
        yaw0 = float(robot_yaw)

        t0 = time.perf_counter()

        # 1) Generate candidates (quintic)
        paths, vels = self.generate_paths_trajectory(robot_xyz, yaw0, robot_vel, goal_xyz)

        # 2) Filter (STATIC ONLY)
        safe_paths, safe_vels = self.filter_unsafe_paths(
            paths, vels,
            boxes_3d=boxes_3d,
            pred_xyz=pred_xyz,
            pred_mask=pred_mask,
        )
        if safe_paths is None:
            return None, {"feasible": False, "final_path": None, "timing_ms": (time.perf_counter()-t0)*1000.0}

        # 3) Score (includes lambda avoidance shaping)
        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, goal_xyz, pred_xyz, pred_mask)

        best_path = safe_paths[best_idx]
        best_vel_traj = safe_vels[best_idx]

        next_pos = best_path[1]
        cmd_vx, cmd_vy, cmd_vz, cmd_w = best_vel_traj[1]  # env order

        # Clamp to limits
        cmd_vx = float(np.clip(cmd_vx, self.vxy_min, self.vxy_max))
        cmd_vy = float(np.clip(cmd_vy, self.vxy_min, self.vxy_max))
        cmd_vz = float(np.clip(cmd_vz, self.vz_min, self.vz_max))
        cmd_w  = float(np.clip(cmd_w,  self.w_min,  self.w_max))

        target_vel_env = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_w], dtype=np.float32)

        info = {
            "feasible": True,
            "final_path": best_path,
            "cost": best_cost,
            "timing_ms": (time.perf_counter()-t0)*1000.0,
            "target_pos": next_pos,
            "target_vel": target_vel_env,
            "candidate_paths": paths,
            "safe_paths": safe_paths,
        }
        return (next_pos, target_vel_env), info

    # ============================================================
    # Lambda update (kept)
    # ============================================================
    def update_conformal_var(self, robot_xyz: np.ndarray, tracking_res: dict):
        robot_xyz = np.asarray(robot_xyz, dtype=np.float32).reshape(3,)
        positions = np.array([np.asarray(xyz_hist)[-1][:3] for xyz_hist in tracking_res.values()], dtype=np.float32)
        if positions.size == 0:
            return {"loss": 0.0, "conformal_var": self._lambda}

        loss = -np.min(np.linalg.norm(robot_xyz[None, :] - positions, axis=1))
        self._lambda -= self._eta * (self._epsilon - loss)
        self._lambda = max(0.01, self._lambda)
        return {"loss": float(loss), "conformal_var": float(self._lambda)}