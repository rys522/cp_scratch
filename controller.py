from __future__ import annotations

import numpy as np

from utils import build_grid, distance_field_points  


def bilinear_sample(field: np.ndarray, x: float, y: float, box: float) -> float:
    """
    field: (H, W) on coordinates xs in [-box/2, box/2], ys in [-box/2, box/2]
    query point (x,y) in same coordinate system.
    """
    H, W = field.shape
    # map x in [-box/2, box/2] -> u in [0, W-1]
    # map y in [-box/2, box/2] -> v in [0, H-1]
    u = (x + box / 2.0) / box * (W - 1)
    v = (y + box / 2.0) / box * (H - 1)

    # clamp
    u = float(np.clip(u, 0.0, W - 1.000001))
    v = float(np.clip(v, 0.0, H - 1.000001))

    u0 = int(np.floor(u))
    v0 = int(np.floor(v))
    u1 = min(u0 + 1, W - 1)
    v1 = min(v0 + 1, H - 1)

    du = u - u0
    dv = v - v0

    f00 = field[v0, u0]
    f10 = field[v0, u1]
    f01 = field[v1, u0]
    f11 = field[v1, u1]

    return float((1 - du) * (1 - dv) * f00 + du * (1 - dv) * f10 + (1 - du) * dv * f01 + du * dv * f11)


class CPFieldMPCController:

    def __init__(
        self,
        *,
        box: float,
        world_center: np.ndarray,
        g_upper: np.ndarray,
        grid_H: int = 64,
        grid_W: int = 64,
        n_steps: int = 20,
        dt: float = 0.1,
        n_skip: int = 4,
        # control limits
        min_linear_x: float = -20.0,
        max_linear_x: float = 20.0,
        min_angular_z: float = -1.1,
        max_angular_z: float = 1.1,
        # sampling / MPC
        n_paths: int = 1500,
        seed: int = 0,
        # geometry
        robot_rad: float = 0.4,
        obstacle_rad: float = 1.0 / np.sqrt(2.0),
        # costs
        w_goal: float = 1.0,
        w_terminal: float = 10.0,
        w_control: float = 1e-3,
        w_cp_violation: float = 200.0,
        hard_cp_constraint: bool = True,
    ):
        self.BOX = float(box)
        self.WORLD_CENTER = np.asarray(world_center, dtype=np.float32)
        self.g_upper = np.asarray(g_upper, dtype=np.float32)

        self.H = int(grid_H)
        self.W = int(grid_W)
        _, _, self.Xg, self.Yg = build_grid(self.BOX, self.H, self.W)

        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        self.min_linear_x = float(min_linear_x)
        self.max_linear_x = float(max_linear_x)
        self.min_angular_z = float(min_angular_z)
        self.max_angular_z = float(max_angular_z)

        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(seed)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_control = float(w_control)
        self.w_cp_violation = float(w_cp_violation)
        self.hard_cp_constraint = bool(hard_cp_constraint)

    # -------------------------
    # public API
    # -------------------------
    def set_g_upper(self, g_upper: np.ndarray):
        self.g_upper = np.asarray(g_upper, dtype=np.float32)

    def __call__(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
        boxes,
        obst_pred_traj: np.ndarray,
        obst_mask: np.ndarray,
        goal: np.ndarray,
    ):
        """
        obst_pred_traj: (T_pred, M, 2) world coords
        obst_mask: (T_pred, M) bool
        goal: (2,) world coords
        """
        goal = np.asarray(goal, dtype=np.float32)

        paths, vels = self.generate_paths_random(
            pos_x=float(pos_x),
            pos_y=float(pos_y),
            th=float(orientation_z),
        )

        # 1) static collision filter (rectangles)
        paths_s, vels_s = self.filter_unsafe_paths_static(paths, vels, boxes)
        if paths_s is None:
            return None, {"feasible": False, "reason": "all paths collide with static boxes"}

        # 2) build CP lower-bound distance fields (obstacle prediction + g_upper)
        lower_multi = self.build_lower_fields_cp(
            obst_pred=obst_pred_traj,
            masks=obst_mask,
        )

        # 3) score paths with CP safety
        best_idx, info = self.score_paths_cp(
            paths_s, vels_s, lower_multi, goal
        )

        v_best = vels_s[best_idx, 0, 0]
        w_best = vels_s[best_idx, 0, 1]
        info["feasible"] = True
        info["final_path"] = paths_s[best_idx]
        info["final_vel_profile"] = vels_s[best_idx]
        return np.array([v_best, w_best], dtype=np.float32), info

    # -------------------------
    # core pieces
    # -------------------------
    def generate_paths_random(self, pos_x: float, pos_y: float, th: float):

        linear_set = np.array([self.min_linear_x, 0.0, self.max_linear_x], dtype=np.float32)
        angular_set = np.array([self.min_angular_z, 0.0, self.max_angular_z], dtype=np.float32)

        n_epochs = max(1, self.n_steps // self.n_skip)

        idx_v = self.rng.integers(0, linear_set.size, size=(self.n_paths, n_epochs))
        idx_w = self.rng.integers(0, angular_set.size, size=(self.n_paths, n_epochs))
        v_epoch = linear_set[idx_v]  # (P, E)
        w_epoch = angular_set[idx_w]  # (P, E)

        # expand to step-wise
        v = np.repeat(v_epoch, repeats=self.n_skip, axis=1)[:, : self.n_steps]  # (P, T)
        w = np.repeat(w_epoch, repeats=self.n_skip, axis=1)[:, : self.n_steps]  # (P, T)

        # rollout unicycle
        P = self.n_paths
        T = self.n_steps
        x = np.zeros((P, T + 1), dtype=np.float32)
        y = np.zeros((P, T + 1), dtype=np.float32)
        th_arr = np.zeros((P, T + 1), dtype=np.float32)

        x[:, 0] = pos_x
        y[:, 0] = pos_y
        th_arr[:, 0] = th

        dt = self.dt
        for t in range(T):
            x[:, t + 1] = x[:, t] + dt * v[:, t] * np.cos(th_arr[:, t])
            y[:, t + 1] = y[:, t] + dt * v[:, t] * np.sin(th_arr[:, t])
            th_arr[:, t + 1] = th_arr[:, t] + dt * w[:, t]

        paths = np.stack([x, y], axis=-1)              # (P, T+1, 2)
        vels = np.stack([v, w], axis=-1)               # (P, T, 2)
        return paths, vels

    def filter_unsafe_paths_static(self, paths, vels, boxes):

        if boxes is None or len(boxes) == 0:
            return paths, vels

        P = paths.shape[0]
        masks_all = np.zeros((P, self.n_steps), dtype=bool)  # True = collision

        for box in boxes:
            center = np.asarray(box.pos, dtype=np.float32)
            sz = np.array([box.w, box.h], dtype=np.float32)
            th = float(box.rad)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)  # rotate by -th

            lb = -0.5 * sz - self.robot_rad
            ub =  0.5 * sz + self.robot_rad

            # ignore t=0 (current state), check future steps 1..T
            transformed = (paths[:, 1:, :] - center) @ R  # (P,T,2)
            coll = np.logical_and(
                np.all(transformed <= ub, axis=-1),
                np.all(transformed >= lb, axis=-1),
            )
            masks_all |= coll

        safe = ~np.any(masks_all, axis=1)
        if np.any(safe):
            return paths[safe], vels[safe]
        return None, None

    def build_lower_fields_cp(self, obst_pred: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        obst_pred: (T_pred, M, 2) world
        masks: (T_pred, M)
        g_upper: (T_pred, H, W)

        return:
            lower_multi: (T_use, H, W)  lower bound on true distance to nearest obstacle
        """
        obst_pred = np.asarray(obst_pred, dtype=np.float32)
        masks = np.asarray(masks, dtype=bool)

        T_pred, M, _ = obst_pred.shape
        T_use = min(self.n_steps, T_pred, self.g_upper.shape[0])

        lower_multi = np.full((T_use, self.H, self.W), np.inf, dtype=np.float32)

        for t in range(T_use):
            g_up_t = self.g_upper[t]  # (H,W)
            for m in range(M):
                if not masks[t, m]:
                    continue
                y_hat = obst_pred[t, m] - self.WORLD_CENTER  # -> [-BOX/2, BOX/2] frame
                d_pred = distance_field_points(y_hat, self.Xg, self.Yg)  # (H,W)
                lower_m = np.maximum(d_pred - g_up_t, 0.0)
                lower_multi[t] = np.minimum(lower_multi[t], lower_m)

        return lower_multi

    def score_paths_cp(self, paths, vels, lower_multi, goal):
        """
        paths: (P, T+1, 2) world
        vels:  (P, T, 2)
        lower_multi: (T_use, H, W)
        """
        P, T1, _ = paths.shape
        T = T1 - 1
        T_use = min(T, lower_multi.shape[0])

        # goal costs
        diff = paths[:, 1:, :] - goal[None, None, :]         # (P,T,2)
        intermediate = self.w_goal * np.sum(diff[:, :T_use] ** 2, axis=(1, 2))
        terminal = self.w_terminal * np.sum((paths[:, T_use, :] - goal[None, :]) ** 2, axis=1)

        # control cost
        control = self.w_control * np.sum(vels[:, :T_use, :] ** 2, axis=(1, 2))

        # CP safety (lower bound distance sampled on robot trajectory)
        violation = np.zeros((P,), dtype=np.float32)
        infeasible = np.zeros((P,), dtype=bool)

        for t in range(T_use):
            field = lower_multi[t]
            # sample each path at step t+1
            # convert world->centered for sampling
            xy = paths[:, t + 1, :] - self.WORLD_CENTER[None, :]
            # bilinear sample per path
            d_lb = np.array(
                [bilinear_sample(field, xy[i, 0], xy[i, 1], self.BOX) for i in range(P)],
                dtype=np.float32
            )
            # violation when lower bound distance is below safe radius
            v_t = np.maximum(self.safe_rad - d_lb, 0.0)
            violation += v_t

            if self.hard_cp_constraint:
                infeasible |= (d_lb < self.safe_rad)

        cost = intermediate + terminal + control + self.w_cp_violation * violation

        if self.hard_cp_constraint:
            # hard infeasible paths get +inf
            cost = np.where(infeasible, np.inf, cost)

        best = int(np.argmin(cost))
        info = {
            "min_cost": float(cost[best]),
            "n_infeasible": int(np.sum(np.isinf(cost))),
            "avg_violation": float(np.mean(violation)),
        }
        return best, info