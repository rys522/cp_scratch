# controllers/func_cp_mpc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from cp.functional_cp import CPStepParameters
import time
import math


@dataclass
class FuncMPCWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001
    w_safety: float = 10.0
    w_margin: float = 50.0
    safety_scale: float = 0.2


class FunctionalCPMPC:
    """
    Functional CP-MPC Controller (parametric envelope evaluation).

    Safety constraint (hard):
        D_lower(x) = max(D_nom(x) - U_t(x), 0) >= r_safe
    where:
        r_safe = robot_rad + obstacle_rad

    """

    def __init__(
        self,
        *,
        cp_params: List[any],
        box: float,
        world_center: np.ndarray,
        grid_H: int,
        grid_W: int,
        n_steps: int,
        dt: float,
        n_skip: int,
        robot_rad: float,
        obstacle_rad: float,
        min_linear_x: float,
        max_linear_x: float,
        min_angular_z: float,
        max_angular_z: float,
        n_paths: int,
        seed: int = 0,
        weights: Optional[FuncMPCWeights] = None,
        risk_level: float = -1.0,  # Target safety margin (e.g., -1.0 means maintaining 1.0m)
        step_size: float = 5000.     # Adaptation rate for the safety weight (eta)
    ):
        # Map and Grid configuration
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H, self.grid_W = int(grid_H), int(grid_W)
        
        # Store CP parameters by time horizon index t_idx
        self.params = {int(p.t_idx): p for p in cp_params}

        # MPC Horizon and Robot Kinematics
        self.n_steps, self.dt, self.n_skip = int(n_steps), float(dt), int(n_skip)
        self.robot_rad, self.obstacle_rad = float(robot_rad), float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.min_v, self.max_v = float(min_linear_x), float(max_linear_x)
        self.min_w, self.max_w = float(min_angular_z), float(max_angular_z)

        # Path generation settings
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.weights = weights or FuncMPCWeights()
        self.last_best_vels = None

        # Adaptive parameters (CC logic)
        self._epsilon = risk_level
        self._eta = step_size

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------

    def _world_to_grid_ij(self, pos_world: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Map world position to nearest grid index (i,j).

        Grid is assumed to represent coordinates:
            x_rel, y_rel in [-box/2, box/2]
        where rel = pos_world - world_center.
        """
        rel = np.asarray(pos_world, dtype=np.float32) - self.world_center
        u = (rel[0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        if not (0.0 <= u <= (self.grid_W - 1) and 0.0 <= v <= (self.grid_H - 1)):
            return None
        j = int(np.rint(u))
        i = int(np.rint(v))
        return i, j

    def _grid_flat_index(self, ij: Tuple[int, int]) -> int:
        i, j = ij
        return i * self.grid_W + j
    
    def update_adaptive_weight(self, pos_x, pos_y, tracking_res):
        positions = np.array([xy[-1] for xy in tracking_res.values()])
        loss = -np.min(np.sum((np.array([pos_x, pos_y]) - positions) ** 2, axis=-1) ** .5)
        self.weights.w_safety -= self._eta * (self._epsilon - loss)
        self.weights.w_safety = max(0.01, self.weights.w_safety)
        return self.weights.w_safety

    # ---------------------------------------------------------------------
    # Functional CP envelope evaluation: U_t(x)
    # ---------------------------------------------------------------------

    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        """
        Evaluate parametric upper envelope U_t(x) from CPStepParameters.

        U_t(x) = epsilon + max_k { mu_k^T phi(x) + r_k * sqrt(phi(x)^T Sigma_k phi(x)) }

        Notes
        -----
        - phi_basis is (p_eff, D)
        - phi(x) is a length-p_eff vector: phi_basis[:, idx]
        """
        p = self.params.get(int(t_idx))
        if p is None:
            # Conservative fallback: growing bound with horizon
            return float(0.5 * self.dt * (t_idx + 1))

        ij = self._world_to_grid_ij(pos_world)
        if ij is None:
            # outside map -> conservative
            return float(1.5)

        idx = self._grid_flat_index(ij)

        # basis evaluation at x
        # phi_basis: (p_eff, D) -> phi_x: (p_eff,)
        phi_x = p.phi_basis[:, idx].astype(np.float32, copy=False)
        eps = float(p.epsilon)

        best = -np.inf
        for k in range(int(p.K)):
            mu_k = p.mus[k]            # (p_eff,)
            Sig_k = p.sigmas[k]        # (p_eff, p_eff)
            rk = float(p.rks[k])

            center = float(mu_k @ phi_x)
            quad = float(phi_x.T @ Sig_k @ phi_x)
            radius = rk * float(np.sqrt(max(quad, 0.0)))
            best = max(best, center + radius)

        return float(eps + best)

    # ---------------------------------------------------------------------
    # MPC main API
    # ---------------------------------------------------------------------

    def __call__(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
        boxes=None,
        predictions=None,
        goal=None,
        *,
        obst_pred_traj: Optional[np.ndarray] = None,  # (H, M, 2) or (H,2) etc.
        obst_mask: Optional[np.ndarray] = None,       # (H, M) or (H,)
    ):
        """
        Compatible call interface.

        Supports BOTH:
        (A) dict predictions format:
            predictions: {track_id: np.ndarray(T,2)}  (world coords)

        (B) trajectory+mask format (your sim code):
            obst_pred_traj: np.ndarray(H, M, 2)  (world coords)
            obst_mask:      np.ndarray(H, M)     (bool visibility)

        Returns:
            act: np.ndarray(2,) == [v,w] or None
            info: dict
        """
        # -----------------------------
        # 0) Normalize goal
        # -----------------------------
        if goal is None:
            raise ValueError("goal must be provided.")
        goal = np.asarray(goal, dtype=np.float32)

        # -----------------------------
        # 1) Normalize predictions input
        # -----------------------------
        # If caller provided obst_pred_traj, convert it to dict-format predictions
        if obst_pred_traj is not None:
            pred_arr = np.asarray(obst_pred_traj, dtype=np.float32)

            # allow (H,2) single obstacle -> (H,1,2)
            if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
                pred_arr = pred_arr[:, None, :]

            if pred_arr.ndim != 3 or pred_arr.shape[-1] != 2:
                raise ValueError("obst_pred_traj must have shape (H,M,2) (or (H,2) for single).")

            H, M, _ = pred_arr.shape

            if obst_mask is None:
                # assume all visible
                mask = np.ones((H, M), dtype=bool)
            else:
                mask = np.asarray(obst_mask, dtype=bool)
                if mask.ndim == 1 and M == 1 and mask.shape[0] == H:
                    mask = mask[:, None]
                if mask.shape != (H, M):
                    raise ValueError(f"obst_mask shape must be (H,M) == {(H,M)}, got {mask.shape}.")

            # Build dict {m: (H,2)} using ONLY visible rows per agent (optional)
            # For your current filter_paths, simplest is to keep fixed length (H,2)
            # and let filtering use all rows; visibility can be handled by masking out with +inf.
            pred_dict = {}
            for m in range(M):
                traj_m = pred_arr[:, m, :].copy()
                # If not visible at some horizon steps, push far away so it doesn't constrain
                invis = ~mask[:, m]
                if np.any(invis):
                    traj_m[invis] = 1e9  # effectively removes that obstacle at those times
                pred_dict[m] = traj_m
            predictions = pred_dict

        # If neither provided, fallback to empty
        if predictions is None:
            predictions = {}

        # boxes normalize
        boxes = boxes or []

        t0 = time.perf_counter()

        # -----------------------------
        # 2) rollout
        # -----------------------------
        t_roll0 = time.perf_counter()
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z) # Monte Carlo random
        #paths, vels = self.generate_paths(pos_x, pos_y, orientation_z)  # discrete input

        t_roll1 = time.perf_counter()

        t_filt0 = time.perf_counter()
        safe_paths, vels_safe, cp_violation = self.filter_unsafe_paths(paths, vels, boxes, predictions)
        t_filt1 = time.perf_counter()

        stats = {
            "n_paths": int(paths.shape[0]),
            "n_safe": int(0 if safe_paths is None else safe_paths.shape[0]),
            "cp_violation": float(cp_violation),
        }

        if safe_paths is None or vels_safe is None or len(vels_safe) == 0:
            return None, {
                "feasible": False,
                "final_path": None,
                "cost": None,
                "timing": {
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                    "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                    "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                },
                "counts": stats,
            }

        # -----------------------------
        # 4) score
        # -----------------------------
        t_score0 = time.perf_counter()
        best_idx, cost = self.score_paths(safe_paths, vels_safe, goal, predictions)
        self.last_best_vels = vels_safe[best_idx].copy()
        t_score1 = time.perf_counter()

        # Return action as [v,w] (shape (2,))
        action_idx = 0
        action_idx = int(np.clip(action_idx, 0, vels_safe.shape[1] - 1))
        act = np.asarray(vels_safe[best_idx, action_idx], dtype=np.float32)

        info = {
            "feasible": True,
            "final_path": safe_paths[best_idx],
            "cost": float(cost),
            "timing": {
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                "score_ms": (t_score1 - t_score0) * 1000.0,
            },
            "counts": stats,
        }
        return act, info

    # ---------------------------------------------------------------------
    # Safety filtering
    # ---------------------------------------------------------------------

    def filter_unsafe_paths(self, paths, vels, boxes, predictions):
        """
        Returns
        -------
        safe_paths, safe_vels, total_violation (float)
        """
        P, T1, _ = paths.shape
        T = T1 - 1

        boxes = boxes or []
        predictions = predictions or {}

        # ---- (A) static obstacles ----
        mask_static_unsafe = np.zeros(P, dtype=bool)
        if len(boxes) > 0:
            for box in boxes:
                center = box.pos
                sz = np.array([box.w, box.h], dtype=np.float32)
                th = float(box.rad)
                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                lb = -0.5 * sz - self.robot_rad
                ub = 0.5 * sz + self.robot_rad

                transformed = (paths[:, 1:, :] - center) @ R  # (P,T,2)
                coll = np.any(
                    np.all((transformed >= lb) & (transformed <= ub), axis=-1),
                    axis=-1
                )
                mask_static_unsafe |= coll

        # ---- (B) dynamic CP constraint ----
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        cp_violation = 0.0

        if len(predictions) > 0:
            # predictions dict -> array (T_pred, M, 2)
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)  # (M,T,2) or (M,?,2)
            if pred_arr.ndim != 3:
                raise ValueError("predictions must be dict of arrays shaped (T,2).")
            pred_arr = pred_arr.transpose(1, 0, 2)  # (T_pred, M, 2)

            T_use = min(T, pred_arr.shape[0], 0)

            for i in range(P):
                # early stop if already unsafe
                for t in range(T_use):
                    x = paths[i, t + 1]
                    d_nom = float(np.min(np.linalg.norm(x[None, :] - pred_arr[t], axis=1)))
                    U = self.evaluate_U(x, t)

                    # D_lower = max(d_nom - U, 0)
                    d_lower = max(d_nom - U, 0.0)

                    if d_lower < self.safe_rad:
                        mask_dynamic_unsafe[i] = True
                        break

        mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe], float(cp_violation)

        return None, None, float(cp_violation)

    # ---------------------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------------------
    
    def generate_paths_random(self, pos_x, pos_y, orientation_z):
        # --- robust epoch count (works even if n_steps < n_skip) ---
        n_steps = int(self.n_steps)
        n_skip = int(self.n_skip)

        if n_steps <= 0:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        # ceil so we always have enough epochs to cover n_steps
        n_epochs = int(math.ceil(n_steps / max(1, n_skip)))

        # epoch-wise random controls
        v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs)).astype(np.float32)
        w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs)).astype(np.float32)

        if self.last_best_vels is not None:
            v_warm = np.append(self.last_best_vels[1:, 0], self.rng.uniform(self.min_v, self.max_v))
            w_warm = np.append(self.last_best_vels[1:, 1], self.rng.uniform(self.min_w, self.max_w))
        
            v_epoch[0, :] = v_warm[:n_epochs]
            w_epoch[0, :] = w_warm[:n_epochs]

        # expand to per-step and slice exactly n_steps
        v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]
        w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]

        # allocate paths with length n_steps+1 (THIS is important)
        paths = np.zeros((self.n_paths, n_steps + 1, 2), dtype=np.float32)
        paths[:, 0, 0] = float(pos_x)
        paths[:, 0, 1] = float(pos_y)

        th = np.full((self.n_paths,), float(orientation_z), dtype=np.float32)
        dt = float(self.dt)

        for t in range(n_steps):
            paths[:, t + 1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(th)
            paths[:, t + 1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(th)
            th = th + dt * w[:, t]

        vels = np.stack([v, w], axis=-1).astype(np.float32)  # (P, n_steps, 2)
        return paths, vels
    
    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            n_skip=4
    ):
        """
        Generate multiple paths starting at (x, y, theta) = (0, 0, 0)
        """

        # TODO: Employing pruning techniques would reduce the number of the paths, but would be also challenging to optimize...
        # TODO: use numba?
        dt = self.dt

        linear_xs = np.array([self.min_v, .0, self.max_v])
        angular_zs = np.array([self.min_w, .0, self.max_w])

        n_points = linear_xs.size * angular_zs.size

        linear_xs, angular_zs = np.meshgrid(linear_xs, angular_zs)

        linear_xs = np.reshape(linear_xs, newshape=(-1,))
        angular_zs = np.reshape(angular_zs, newshape=(-1,))

        # (# grid points, 2)
        # velocity_profile = np.stack((linear_xs, angular_zs), axis=0)

        n_decision_epochs = self.n_steps // n_skip

        # profiles = [velocity_profile for _ in range(n_decision_epochs)]

        # n_paths = n_points ** n_decision_epochs

        state_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self.n_steps + 1,)
        x = np.zeros(state_shape)
        y = np.zeros(state_shape)
        th = np.zeros(state_shape)

        # state initialization
        x[..., 0] = pos_x
        y[..., 0] = pos_y
        th[..., 0] = orientation_z

        control_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self.n_steps,)
        v = np.zeros(control_shape)
        w = np.zeros(control_shape)

        for e in range(n_decision_epochs):
            augmented_shape = [1] * n_decision_epochs
            augmented_shape[e] = -1
            v_epoch = linear_xs.reshape(augmented_shape)
            w_epoch = angular_zs.reshape(augmented_shape)
            for t in range(e * n_skip, (e + 1) * n_skip):
                v[..., t] = v_epoch
                w[..., t] = w_epoch

                x[..., t + 1] = x[..., t] + dt * v_epoch * np.cos(th[..., t])
                y[..., t + 1] = y[..., t] + dt * v_epoch * np.sin(th[..., t])
                th[..., t + 1] = th[..., t] + dt * w_epoch

        x = np.reshape(x, (-1, self.n_steps + 1))
        y = np.reshape(y, (-1, self.n_steps + 1))
        # th = np.reshape(th, (-1, self._n_steps))
        v = np.reshape(v, (-1, self.n_steps))
        w = np.reshape(w, (-1, self.n_steps))

        return np.stack((x, y), axis=-1), np.stack((v, w), axis=-1)

    # ---------------------------------------------------------------------
    # Cost
    # ---------------------------------------------------------------------

    def score_paths(self, paths, vels, goal, predictions):
        """
        Scores candidate paths based on distance to goal, control effort, and safety.
        Incorporates the adaptive w_safety and Functional CP uncertainty U.
        """
        P, T1, _ = paths.shape
        T = T1 - 1
        
        # 1. Standard MPC Costs: Goal Tracking and Control Smoothness
        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total_cost = intermediate + terminal + control

        # 2. Risk-level Based Dynamic Safety Cost (Soft Constraint)
        if self.weights.w_safety > 0.0 and predictions and len(predictions) > 0:
            # predictions format: (Time, Agents, XY)
            pred_arr = np.asarray(list(predictions.values())).transpose(1, 0, 2)
            T_use = min(T, pred_arr.shape[0])
            
            safety_acc = np.zeros(P, dtype=np.float32)
            scale = self.weights.safety_scale

            for t in range(T_use):
                x_t = paths[:, t + 1, :]
                
                # (B) Calculate Nominal Distance to the nearest predicted agent
                diff = x_t[:, None, :] - pred_arr[t][None, :, :]
                d_nom = np.min(np.linalg.norm(diff, axis=-1), axis=1)
                
                # (C) Evaluate Spatial Uncertainty U using Functional CP parameters
                U_vec = np.array([self.evaluate_U(x_t[i], t) for i in range(P)])
                
                # (D) Compute Lower Bound Distance: d_lower = d_nom - U
                d_lower = np.maximum(d_nom - U_vec, 0.0)
                
                # (E) Apply Exponential Penalty for safety margin violations
                # Penalty increases sharply as d_lower approaches safe_rad

                urgency_weight = 2.0 / (t + 0.5) 
                violation = np.maximum(self.safe_rad - d_lower, 0.0)  # shape (P,)

                # hard threshold (optional)
                hard_mask = violation > 0.0

                penalty = np.zeros_like(d_lower)

                # unsafe: large penalty
                penalty[hard_mask] = 1000.0 * urgency_weight
                penalty[~hard_mask] = np.exp(-d_lower[~hard_mask])
                reward = 0.0
                if t > int(T_use / 1.7) : 
                    reward = np.maximum(0.0, -5.0 * (d_lower - self.safe_rad))  # 멀어질수록 cost 감소(보상)

                safety_acc +=  urgency_weight * (penalty + reward)     


            # Integrate the adaptive w_safety into the total cost
            total_cost += self.weights.w_safety * (safety_acc / T_use)

        best_idx = np.argmin(total_cost)
        return best_idx, total_cost[best_idx]