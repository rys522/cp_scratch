# controllers/func_cp_mpc_3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
import time
from cp.functional_cp import CPStepParameters

@dataclass
class MPC3DWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001


class FunctionalCPMPC3D:
    """
    Trajectory-Based 3D CP-MPC (Quintic Polynomials):
      - Samples *End-points* instead of random velocities.
      - Generates smooth 5th-order polynomial trajectories.
      - Output: Next desired position & feedforward velocity for PID.
    """

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
        v_lim: Tuple[float,float],
        vz_lim: Tuple[float,float],
        yaw_rate_lim: Tuple[float,float],
        n_paths: int,
        seed: int = 0,
        weights: Optional[MPC3DWeights] = None,
        CP: bool = True,
    ):
        self.params: Dict[int, CPStepParameters] = {int(p.t_idx): p for p in cp_params}
        self.xs, self.ys, self.zs = xs, ys, zs
        self.nx, self.ny, self.nz = xs.size, ys.size, zs.size
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
        self.CP = CP

    # -----------------------------
    # Grid Lookup & CP Evaluation
    # -----------------------------
    def _world_to_grid_ijk_float(self, p: np.ndarray) -> Optional[Tuple[float,float,float]]:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1] and self.zs[0] <= z <= self.zs[-1]):
            return None
        i1 = int(np.clip(np.searchsorted(self.xs, x), 1, self.nx - 1))
        j1 = int(np.clip(np.searchsorted(self.ys, y), 1, self.ny - 1))
        k1 = int(np.clip(np.searchsorted(self.zs, z), 1, self.nz - 1))
        i0, j0, k0 = i1 - 1, j1 - 1, k1 - 1
        x0, x1 = float(self.xs[i0]), float(self.xs[i1])
        y0, y1 = float(self.ys[j0]), float(self.ys[j1])
        z0, z1 = float(self.zs[k0]), float(self.zs[k1])
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        tz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
        return (k0 + tz, j0 + ty, i0 + tx)

    def _grid_flat_index(self, ijk_float: Tuple[float, float, float]) -> int:
        kf, jf, if_ = ijk_float
        k = int(np.clip(int(round(kf)), 0, self.nz - 1))
        j = int(np.clip(int(round(jf)), 0, self.ny - 1))
        i = int(np.clip(int(round(if_)), 0, self.nx - 1))
        return (k * self.ny + j) * self.nx + i
        
    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        p = self.params.get(int(t_idx))
        if p is None:
            return 1.0
        ijk = self._world_to_grid_ijk_float(pos_world)
        if ijk is None:
            return 1.0
        idx = self._grid_flat_index(ijk)
        phi_x = p.phi_basis[:, idx].astype(np.float32, copy=False)
        centers = p.mus @ phi_x
        quads = np.einsum("i,kij,j->k", phi_x, p.sigmas, phi_x)
        upper_k = centers + p.rks * np.sqrt(np.maximum(quads, 0.0))
        return float(p.epsilon + np.max(upper_k))

    # -----------------------------
    # Trajectory Generation: Quintic Polynomials
    # -----------------------------
    def _solve_quintic_coeffs(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, T):
        """
        Vectorized solver for Quintic Polynomial coefficients.
        Returns c0..c5 for p(t) = c0 + c1*t + ... + c5*t^5
        """
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        c0 = start_pos
        c1 = start_vel
        c2 = 0.5 * start_acc

        H = end_pos - c0 - c1*T - c2*T2
        V = end_vel - c1 - 2*c2*T
        A = end_acc - 2*c2

        # Inverse matrix solution for quintic boundary conditions
        c3 = (10*H - 4*V*T + 0.5*A*T2) / T3
        c4 = (-15*H + 7*V*T - 1.0*A*T2) / T4
        c5 = (6*H - 3*V*T + 0.5*A*T2) / T5
        
        return c0, c1, c2, c3, c4, c5

    def generate_paths_trajectory(
        self, 
        x0yz: np.ndarray, 
        yaw0: float, 
        current_vel: np.ndarray, 
        goal_xyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples endpoints and generates smooth polynomial trajectories.
        """
        # 1. Setup Start Conditions
        start_vel = current_vel * 0.8 if current_vel is not None else np.zeros(3, dtype=np.float32)
        start_acc = np.zeros(3, dtype=np.float32) # Assume 0 accel start

        current_speed = np.linalg.norm(start_vel)

        # 2. Sample Goal Candidates
        dist_to_goal = np.linalg.norm(goal_xyz - x0yz)
        T_horizon = self.n_steps * self.dt
                
        if dist_to_goal > 1e-3:
            dir_vec = (goal_xyz - x0yz) / dist_to_goal
        else:
            dir_vec = np.zeros(3)

        max_reachable_dist = self.vmax * T_horizon
        
        if dist_to_goal < max_reachable_dist:
            center_pt = goal_xyz
            is_near_goal = True
        else:
            center_pt = x0yz + dir_vec * max_reachable_dist
            is_near_goal = False    

        # Sampling endpoints (Gaussian around target)
        sigma = 1.0
        term_pos = self.rng.normal(loc=center_pt, scale=sigma, size=(self.n_paths, 3)).astype(np.float32)
        
        # Ensure at least one path goes exactly to goal/center
        term_pos[0] = center_pt

        term_vel = np.zeros_like(term_pos)
        term_acc = np.zeros_like(term_pos)


        diff = term_pos - x0yz[None, :] 
        dists = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        directions = diff / dists 
        
        low_speed = min(current_speed, self.vmax * 0.5)
        
        sampled_speeds = self.rng.uniform(
            low=low_speed, 
            high=self.vmax, 
            size=(self.n_paths, 1)
        ).astype(np.float32)
        
        sampled_speeds[0] = self.vmax

        term_vel = directions * sampled_speeds

        # 3. Solve Polynomials
        # Broadcasting: x0yz (3,) -> (1,3)
        c0, c1, c2, c3, c4, c5 = self._solve_quintic_coeffs(
            x0yz[None, :], start_vel[None, :], start_acc[None, :],
            term_pos, term_vel, term_acc, T_horizon
        )

        # 4. Evaluate paths over time
        t_eval = np.linspace(0, T_horizon, self.n_steps + 1, dtype=np.float32)
        tt = t_eval[:, None, None] # (T+1, 1, 1)

        # Position: (T+1, N, 3) -> Transpose to (N, T+1, 3)
        paths_t = c0 + c1*tt + c2*tt**2 + c3*tt**3 + c4*tt**4 + c5*tt**5
        paths = np.transpose(paths_t, (1, 0, 2))

        # Velocity: (T+1, N, 3)
        vels_t = c1 + 2*c2*tt + 3*c3*tt**2 + 4*c4*tt**3 + 5*c5*tt**4
        vels_xyz = np.transpose(vels_t, (1, 0, 2))

        # 5. Format Velocities & Compute Yaw
        # Output format: [vx, vy, w, vz]
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
        out_vels[..., 3] = clamped_vz # index 3 is Z

        # 6. Yaw Calculation with Deadband (Singularity Fix)

        full_vx = vels_xyz[..., 0]
        full_vy = vels_xyz[..., 1]
        
        headings = np.arctan2(full_vy, full_vx)
        headings = np.unwrap(headings, axis=1)
        
        w_all = np.diff(headings, axis=1) / self.dt
        
        speed_sq = full_vx[:, :-1]**2 + full_vy[:, :-1]**2
        moving_mask = speed_sq > (0.1**2)
        
        w_final = np.where(moving_mask, w_all, 0.0)
        
        w_final = np.clip(w_final, self.wmin, self.wmax)
        
        out_vels[..., 2] = w_final # index 2 is Yaw Rate

        return paths, out_vels

    # -----------------------------
    # Filtering & Scoring (Reuse existing logic)
    # -----------------------------
    def aabb_contains_points(self, p, box, inflate):
        xmin,xmax,ymin,ymax,zmin,zmax = box
        xmin -= inflate
        xmax += inflate
        ymin -= inflate
        ymax += inflate
        zmin -= inflate 
        zmax += inflate
        return ((p[...,0] >= xmin) & (p[...,0] <= xmax) &
                (p[...,1] >= ymin) & (p[...,1] <= ymax) &
                (p[...,2] >= zmin) & (p[...,2] <= zmax))

    def filter_unsafe_paths(self, paths, vels, boxes_3d, pred_xyz, pred_mask):
        # Static Safety
        P, T1, _ = paths.shape
        unsafe = np.zeros(P, dtype=bool)
        pts = paths[:, 1:, :] # Skip t=0
    

        # Dynamic Safety (CP)
        T = min(T1-1, pred_xyz.shape[0])
        for t in range(T):
            if np.all(unsafe): 
                break # Optimization
            
            # Check only feasible paths
            check_idxs = np.where(~unsafe)[0]
            if len(check_idxs) == 0:
                continue
            
            x_t = paths[check_idxs, t+1, :]
            
            obs_pts = pred_xyz[t] # (M, 3) (Simplified: assume all M valid for now)
            
            # Distance to nearest obstacle
            # (N_check, 1, 3) - (1, M, 3) -> (N, M, 3) -> norm -> min
            dists = np.min(np.linalg.norm(x_t[:, None, :] - obs_pts[None, :, :], axis=-1), axis=1)
            
            # CP Buffering
            if self.CP:
                U_vals = np.array([self.evaluate_U(p, t) for p in x_t], dtype=np.float32)
            else:
                U_vals = np.zeros(len(check_idxs), dtype=np.float32)
            # Safety Condition
            unsafe_t = (dists - U_vals) < self.safe_rad

            # Update global unsafe mask
            unsafe[check_idxs] |= unsafe_t
            
        safe_mask = ~unsafe
        if not np.any(safe_mask): 
            return None, None
        return paths[safe_mask], vels[safe_mask]

    def score_paths(self, paths, vels, goal_xyz):
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)
        
        # 1. Goal Distance Cost (Terminal)
        term_cost = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal)**2, axis=-1)
        
        # 2. Intermediate Tracking Cost
        inter_cost = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal)**2, axis=(-1, -2))
        
        # 3. Control Effort / Smoothness Cost
        # vels: [vx, vy, w, vz]
        ctrl_cost = self.weights.w_control * np.sum(vels**2, axis=(-1, -2))
        
        total_cost = term_cost + inter_cost + ctrl_cost
        best_idx = np.argmin(total_cost)
        return int(best_idx), float(total_cost[best_idx])

    # -----------------------------
    # Public API
    # -----------------------------
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        pred_xyz: np.ndarray,
        pred_mask: np.ndarray,
        boxes_3d: Optional[List[Any]] = None,
        robot_vel: Optional[np.ndarray] = None # Added for smooth trajectory
    ):
        boxes_3d = boxes_3d or []
        t0 = time.perf_counter()

        # Input Sanitization
        robot_vel = robot_vel if robot_vel is not None else np.zeros(3, dtype=np.float32)

        # 1. Generate Trajectories (Quintic Polynomials)
        paths, vels = self.generate_paths_trajectory(
            np.asarray(robot_xyz, dtype=np.float32), 
            float(robot_yaw),
            np.asarray(robot_vel, dtype=np.float32),
            np.asarray(goal_xyz, dtype=np.float32)
        )

        # 2. Safety Check
        safe_paths, safe_vels = self.filter_unsafe_paths(paths, vels, boxes_3d, pred_xyz, pred_mask)

        if safe_paths is None:
            # Emergency fallback: Stop
            return None, {"feasible": False, "final_path": None, "timing_ms": (time.perf_counter()-t0)*1000.0}

        # 3. Select Best Path
        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, np.asarray(goal_xyz, dtype=np.float32))
        
        best_path = safe_paths[best_idx]
        best_vel_traj = safe_vels[best_idx]
        self.last_best_vels = best_vel_traj.copy()

        # 4. Prepare Output for PID Controller
        # next_pos: Desired position at t=1 (dt seconds later)
        next_pos = best_path[1]
        
        # target_vel: Feedforward velocity command
        # best_vel_traj[0] contains [vx, vy, w, vz] for the first step
        cmd_vx, cmd_vy, cmd_w, cmd_vz = best_vel_traj[1]

        cmd_vx = np.clip(cmd_vx, self.vmin, self.vmax)
        cmd_vy = np.clip(cmd_vy, self.vmin, self.vmax)
        cmd_vz = np.clip(cmd_vz, self.vzmin, self.vzmax)
        cmd_w  = np.clip(cmd_w, self.wmin, self.wmax)
        
        # Reorder to match Env Interface: (vx, vy, vz, w) typically, check your env!
        # Assuming Env expects: target_vel = [vx, vy, vz, yaw_rate]
        target_vel_env = np.array([cmd_vx, cmd_vy, cmd_vz, cmd_w], dtype=np.float32)

        info = {
            "feasible": True,
            "final_path": best_path,
            "cost": best_cost,
            "timing_ms": (time.perf_counter()-t0)*1000.0,
            "target_pos": next_pos,
            "target_vel": target_vel_env
        }

        act = (next_pos, target_vel_env)
        
        return act , info