# controllers/func_cp_mpc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from cp.functional_cp import CPStepParameters


@dataclass
class FuncMPCWeights:
    # kept for interface compatibility (unused in pure field controller)
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001


@dataclass
class FieldCtrlParams:
    # goal attraction
    k_goal: float = 2.0

    # safety repulsion (bigger => stronger push away)
    k_rep: float = 6.0

    # safety margin added on top of (robot_rad + obstacle_rad)
    safety_margin: float = 0.1

    # finite-difference step size for gradient (meters)
    grad_eps: float = 0.10

    # unicycle mapping
    k_w: float = 2.0          # heading error -> angular velocity gain
    v_max_scale: float = 1.0  # multiply by max_linear_x
    v_min_scale: float = -1.0  # allow backward if you want (<0), default no backward for stability

    # if hazard close, slow down v
    slow_down: bool = False
    slow_down_dist: float = 1.0  # distance scale for slow down

    # exponential moving average for safety distance
    time_gamma: float = 0.6


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class FunctionalCPFieldController:
    """
    PURE field-based controller (no sampling MPC), but API-compatible with your old FunctionalCPMPC.

    Safety lower-bound at state x for horizon step i:
        D_lower^{(i)}(x) = max( D_nom^{(i)}(x) - U_i(x), 0 )

    where D_nom^{(i)}(x) = min_m || x - p_hat_m[i] || (predicted obstacle positions at i)
    and U_i(x) is evaluated from CPStepParameters for that i.

    We use a conservative aggregate field:
        D_lower_min(x) = min_i D_lower^{(i)}(x)

    Control:
        u = k_goal * dir_to_goal  +  k_rep * (barrier_strength) * grad(D_lower_min)
        then map direction u -> (v,w) for unicycle.
    """

    def __init__(
        self,
        *,
        cp_params: List[CPStepParameters],
        box: float,
        world_center: np.ndarray,
        grid_H: int,
        grid_W: int,
        n_steps: int,
        dt: float,
        n_skip: int,  # kept for compatibility (unused here)
        robot_rad: float,
        obstacle_rad: float,
        min_linear_x: float,
        max_linear_x: float,
        min_angular_z: float,
        max_angular_z: float,
        n_paths: int,  # kept for compatibility (unused here)
        seed: int = 0,  # kept for compatibility
        weights: Optional[FuncMPCWeights] = None,  # kept for compatibility
        hard_cp_constraint: bool = False,  # kept for compatibility
        field_params: Optional[FieldCtrlParams] = None,
        field_T_cap: Optional[int] = None,  # optionally cap how many horizon steps are used (speed)
    ):
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H = int(grid_H)
        self.grid_W = int(grid_W)

        # store params by horizon index i (a.k.a. t_idx in your CPStepParameters)
        self.params: Dict[int, CPStepParameters] = {int(p.t_idx): p for p in cp_params}

        self.n_steps = int(n_steps)
        self.dt = float(dt)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.min_v, self.max_v = float(min_linear_x), float(max_linear_x)
        self.min_w, self.max_w = float(min_angular_z), float(max_angular_z)

        self.field = field_params or FieldCtrlParams()
        self.field_T_cap = field_T_cap

        # keep for compatibility (unused)
        self.weights = weights or FuncMPCWeights()
        self.hard_cp_constraint = bool(hard_cp_constraint)

    # ---------------------------------------------------------------------
    # Grid helpers
    # ---------------------------------------------------------------------

    def _world_to_grid_ij(self, pos_world: np.ndarray) -> Optional[Tuple[int, int]]:
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

    # ---------------------------------------------------------------------
    # U_i(x) evaluation (same as before)
    # ---------------------------------------------------------------------

    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        p = self.params.get(int(t_idx))
        if p is None:
            # conservative fallback
            return float(0.5 * self.dt * (t_idx + 1))

        ij = self._world_to_grid_ij(pos_world)
        if ij is None:
            return float(1.5)

        idx = self._grid_flat_index(ij)

        phi_x = p.phi_basis[:, idx].astype(np.float32, copy=False)
        eps = float(p.epsilon)

        best = -np.inf
        for k in range(int(p.K)):
            mu_k = p.mus[k]      # (p_eff,)
            Sig_k = p.sigmas[k]  # (p_eff,p_eff)
            rk = float(p.rks[k])

            center = float(mu_k @ phi_x)
            quad = float(phi_x.T @ Sig_k @ phi_x)
            radius = rk * float(np.sqrt(max(quad, 0.0)))
            best = max(best, center + radius)

        return float(eps + best)

    # ---------------------------------------------------------------------
    # Lower-bound distance field at a point
    # ---------------------------------------------------------------------

    def _d_lower_i(self, x: np.ndarray, pred_i: np.ndarray, i: int) -> float:
        """
        x: (2,)
        pred_i: (M,2) predicted obstacle positions at step i (world coords)
        """
        if pred_i.size == 0:
            return 1e6
        d_nom = float(np.min(np.linalg.norm(x[None, :] - pred_i, axis=1)))
        U = self.evaluate_U(x, i)
        return float(max(d_nom - U, 0.0))

    def _get_aggregate_hinge(self, x: np.ndarray, pred_arr: np.ndarray, T_use: int) -> float:
        total_hinge = 0.0
        safe_target = self.safe_rad + self.field.safety_margin
        gamma = self.field.time_gamma

        for i in range(T_use):
            di = self._d_lower_i(x, pred_arr[i], i)
            w = gamma ** i 
            hinge = max(0.0, safe_target - di)
            total_hinge += w * hinge
        
        return total_hinge

    def _grad_weighted_hinge(self, x: np.ndarray, pred_arr: np.ndarray, T_use: int, eps: float) -> Tuple[np.ndarray, float]:

        h0 = self._get_aggregate_hinge(x, pred_arr, T_use)

        ex = np.array([eps, 0.0], dtype=np.float32)
        ey = np.array([0.0, eps], dtype=np.float32)

        hpx = self._get_aggregate_hinge(x + ex, pred_arr, T_use)
        hmx = self._get_aggregate_hinge(x - ex, pred_arr, T_use)
        hpy = self._get_aggregate_hinge(x + ey, pred_arr, T_use)
        hmy = self._get_aggregate_hinge(x - ey, pred_arr, T_use)

        gx = (hpx - hmx) / (2.0 * eps)
        gy = (hpy - hmy) / (2.0 * eps)
        
        return np.array([gx, gy], dtype=np.float32), float(h0)

    # ---------------------------------------------------------------------
    # Main call (API-compatible)
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
        obst_pred_traj: Optional[np.ndarray] = None,  # (H,M,2) or (H,2)
        obst_mask: Optional[np.ndarray] = None,       # (H,M) or (H,)
    ):
        if goal is None:
            raise ValueError("goal must be provided.")
        goal = np.asarray(goal, dtype=np.float32)

        # ---- normalize predictions to pred_arr: (T_pred, M, 2) ----
        if obst_pred_traj is not None:
            pred_arr = np.asarray(obst_pred_traj, dtype=np.float32)
            if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
                pred_arr = pred_arr[:, None, :]
            if pred_arr.ndim != 3 or pred_arr.shape[-1] != 2:
                raise ValueError("obst_pred_traj must have shape (H,M,2) or (H,2).")

            H, M, _ = pred_arr.shape
            if obst_mask is None:
                mask = np.ones((H, M), dtype=bool)
            else:
                mask = np.asarray(obst_mask, dtype=bool)
                if mask.ndim == 1 and M == 1 and mask.shape[0] == H:
                    mask = mask[:, None]
                if mask.shape != (H, M):
                    raise ValueError(f"obst_mask shape must be (H,M)=={(H,M)}, got {mask.shape}.")

            pred_arr = pred_arr.copy()
            for m in range(M):
                invis = ~mask[:, m]
                if np.any(invis):
                    pred_arr[invis, m] = 1e9  # ignore at those steps

        elif predictions is not None and len(predictions) > 0:
            # dict {id: (T,2)} -> (T,M,2)
            pred_list = list(predictions.values())
            arr = np.asarray(pred_list, dtype=np.float32)  # (M,T,2)
            if arr.ndim != 3:
                raise ValueError("predictions must be dict of arrays shaped (T,2).")
            pred_arr = arr.transpose(1, 0, 2)
        else:
            pred_arr = np.zeros((0, 0, 2), dtype=np.float32)

        # ---- decide horizon used in field ----
        T_pred = int(pred_arr.shape[0])
        T_use = min(self.n_steps, T_pred) if T_pred > 0 else 0
        if self.field_T_cap is not None:
            T_use = min(T_use, int(self.field_T_cap))

        x = np.array([float(pos_x), float(pos_y)], dtype=np.float32)
        th = float(orientation_z)

        # goal direction
        to_goal = goal - x
        dist_goal = float(np.linalg.norm(to_goal) + 1e-9)
        dir_goal = to_goal / (dist_goal + 1e-9)

        # safety grad
        if T_use <= 0:
            total_h = 0.0
            grad_h = np.zeros((2,), dtype=np.float32)
        else:
            grad_h, total_h = self._grad_weighted_hinge(
                x, pred_arr, T_use, eps=float(self.field.grad_eps)
            )

        safe_target = self.safe_rad + float(self.field.safety_margin)

        repulse = -grad_h 

        u = float(self.field.k_goal) * dir_goal + float(self.field.k_rep) * repulse

        u_norm = float(np.linalg.norm(u) + 1e-9)
        if u_norm < 1e-7:
            v, w = 0.0, 0.0
        else:
            heading_des = float(np.arctan2(u[1], u[0]))
            e = _wrap_pi(heading_des - th)

            # Angular Control
            w = float(np.clip(self.field.k_w * e, self.min_w, self.max_w))

            # Linear Control
            v_cmd = float(self.max_v) * float(self.field.v_max_scale)

            # 감속 로직 (total_h를 기준으로 활용 가능)
            if self.field.slow_down:
                # 가중 위험도가 높을수록(장애물과 가깝거나 미래 충돌이 확실할수록) 감속
                s = float(np.clip(1.0 - (total_h / 1.0), 0.1, 1.0)) # 1.0은 임의의 threshold
                v_cmd *= s

            v_cmd *= float(np.clip(np.cos(e), 0.0, 1.0))
            v_min_allowed = max(self.min_v, float(self.max_v) * float(self.field.v_min_scale))
            v = float(np.clip(v_cmd, v_min_allowed, self.max_v))

        act = np.array([v, w], dtype=np.float32)
        
        info = {
            "feasible": True,
            "field_debug": {
                "T_use": int(T_use),
                "total_hinge": float(total_h),
                "grad_norm": float(np.linalg.norm(grad_h)),
                "dist_goal": float(dist_goal),
            },
        }
        return act, info
    