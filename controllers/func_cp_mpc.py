# controllers/func_cp_mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import time
import numpy as np

from cp.fuctional_spatial_cp import FunctionalCPParams


def min_dist_robot_to_peds(robot_xy, peds_xy):
    if peds_xy is None:
        return float("inf")
    peds_xy = np.asarray(peds_xy, dtype=np.float32)
    if peds_xy.size == 0:
        return float("inf")
    d = peds_xy - np.asarray(robot_xy, dtype=np.float32)[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class FuncMPCWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001


# =============================================================================
# Functional CP-MPC Controller (FUNCTIONAL CP VERSION, HARD FILTER ONLY)
# =============================================================================

class FunctionalCPMPC:
    """
    Functional CP-informed Monte Carlo MPC controller (hard filtering only).

    FunctionalCPParams encodes residual field basis:
        S_i(x) = S_mean(x) + coeffs[i] @ phi_basis[:, x]
        U_i(x) = rhos[i] * max(S_i(x), eps)

    Safety filtering uses:
        d_lower = max(d_nom - U_i(x), 0)
        keep path if d_lower >= safe_rad for all steps along the horizon
    """

    def __init__(
        self,
        *,
        cp_params: FunctionalCPParams,
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
        eps_S: float = 1e-9,
    ):
        # Workspace/grid configuration
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H, self.grid_W = int(grid_H), int(grid_W)

        # MPC rollout configuration
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        # Robot and safety geometry
        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        # Control bounds
        self.min_v, self.max_v = float(min_linear_x), float(max_linear_x)
        self.min_w, self.max_w = float(min_angular_z), float(max_angular_z)

        # Monte Carlo sampling
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.weights = weights or FuncMPCWeights()
        self.last_best_vels: Optional[np.ndarray] = None

        # Functional CP params
        self.cp_params = cp_params
        self.eps_S = float(eps_S)

        # Sanity checks / alignment
        if int(self.cp_params.grid_shape[0]) != self.grid_H or int(self.cp_params.grid_shape[1]) != self.grid_W:
            raise ValueError(
                f"cp_params.grid_shape={self.cp_params.grid_shape} "
                f"but controller grid={(self.grid_H, self.grid_W)}"
            )
        if int(self.cp_params.rhos.shape[0]) != self.n_steps:
            raise ValueError(f"cp_params.rhos has H={self.cp_params.rhos.shape[0]} but n_steps={self.n_steps}")
        if int(self.cp_params.coeffs.shape[0]) != self.n_steps:
            raise ValueError(f"cp_params.coeffs has H={self.cp_params.coeffs.shape[0]} but n_steps={self.n_steps}")

        # Flattened grid size D
        self.D = int(self.grid_H * self.grid_W)

        # S_mean is expected flattened (D,)
        self.S_mean = np.asarray(self.cp_params.S_mean, dtype=np.float32).reshape(-1)
        if self.S_mean.shape[0] != self.D:
            raise ValueError(f"cp_params.S_mean has D={self.S_mean.shape[0]} but expected {self.D}")

        # coeffs: (H, r)
        self.coeffs = np.asarray(self.cp_params.coeffs, dtype=np.float32)
        # phi_basis: (r, D) (recommended) OR (D, r) (we accept and transpose)
        self.phi = np.asarray(self.cp_params.phi_basis, dtype=np.float32)

        if self.phi.ndim != 2:
            raise ValueError("cp_params.phi_basis must be 2D")

        r = int(self.coeffs.shape[1])
        if self.phi.shape == (r, self.D):
            pass
        elif self.phi.shape == (self.D, r):
            self.phi = self.phi.T  # -> (r, D)
        else:
            raise ValueError(
                f"phi_basis shape mismatch. Expected (r,D)=({r},{self.D}) or (D,r)=({self.D},{r}), got {self.phi.shape}"
            )

        self.rhos = np.asarray(self.cp_params.rhos, dtype=np.float32).reshape(self.n_steps,)

        # Simulation timestep counter
        self.step_count = 0

    # ---------------------------------------------------------------------
    # Grid geometry helpers (world <-> grid)
    # ---------------------------------------------------------------------

    def _world_to_grid_ij(self, pos_world: np.ndarray) -> Optional[Tuple[int, int]]:
        rel = np.asarray(pos_world, dtype=np.float32) - self.world_center
        u = (rel[0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        if not (0.0 <= u <= (self.grid_W - 1) and 0.0 <= v <= (self.grid_H - 1)):
            return None

        j = int(np.rint(u))
        i = int(np.rint(v))
        return (i, j)

    def _grid_flat_index(self, ij: Tuple[int, int]) -> int:
        i, j = ij
        return i * self.grid_W + j

    # ---------------------------------------------------------------------
    # Functional CP evaluation: U_i(x)
    # ---------------------------------------------------------------------

    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        """
        U_i(x) = rho_i * max( S_i(x), eps )
        S_i(x) = S_mean(x) + coeffs[i] @ phi_basis[:, x]
        """
        idx = int(t_idx)
        if not (0 <= idx < self.n_steps):
            return 1.0

        ij = self._world_to_grid_ij(pos_world)
        if ij is None:
            return 1.0

        flat = self._grid_flat_index(ij)  # in [0, D)
        # S_mean(x)
        s = float(self.S_mean[flat])
        # + coeffs[i] @ phi[:, flat]
        # (r,) dot (r,)
        s += float(np.dot(self.coeffs[idx], self.phi[:, flat]))
        s = max(s, self.eps_S)

        return float(self.rhos[idx] * s)

    # ---------------------------------------------------------------------
    # Public MPC API
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
        obst_pred_traj: Optional[np.ndarray] = None,  # (H, M, 2) or (H,2)
        obst_mask: Optional[np.ndarray] = None,       # (H, M) or (H,)
        current_obs: Optional[np.ndarray] = None,     # kept only for external collision stats; not used here
    ):
        if goal is None:
            raise ValueError("goal must be provided.")
        goal = np.asarray(goal, dtype=np.float32)

        # Normalize dynamic predictions if obst_pred_traj provided
        if obst_pred_traj is not None:
            predictions = self._normalize_predictions(obst_pred_traj, obst_mask)
        if predictions is None:
            predictions = {}

        boxes = boxes or []

        t0 = time.perf_counter()

        # 1) Sample candidate controls and roll out
        t_roll0 = time.perf_counter()
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z)
        t_roll1 = time.perf_counter()

        # 2) Filter infeasible (hard)
        t_filt0 = time.perf_counter()
        safe_paths, safe_vels, cp_violation = self.filter_unsafe_paths(paths, vels, boxes, predictions)
        t_filt1 = time.perf_counter()

        stats = {
            "n_paths": int(paths.shape[0]),
            "n_safe": int(0 if safe_paths is None else safe_paths.shape[0]),
            "cp_violation": float(cp_violation),
        }

        if safe_paths is None or safe_vels is None or safe_vels.shape[0] == 0:
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

        # 3) Score feasible candidates (goal+effort only)
        t_score0 = time.perf_counter()
        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, goal)
        self.last_best_vels = safe_vels[best_idx].copy()
        t_score1 = time.perf_counter()

        act = np.asarray(safe_vels[best_idx, 0], dtype=np.float32)
        best_path = safe_paths[best_idx]

        info = {
            "feasible": True,
            "final_path": best_path,
            "cost": float(best_cost),
            "timing": {
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                "score_ms": (t_score1 - t_score0) * 1000.0,
            },
            "counts": stats,
        }

        self.step_count += 1
        return act, info

    # ---------------------------------------------------------------------
    # Prediction normalization helper
    # ---------------------------------------------------------------------

    def _normalize_predictions(
        self,
        obst_pred_traj: np.ndarray,
        obst_mask: Optional[np.ndarray],
    ) -> Dict[int, np.ndarray]:
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
                raise ValueError(f"obst_mask must have shape {(H, M)}, got {mask.shape}.")

        pred_dict: Dict[int, np.ndarray] = {}
        for m in range(M):
            traj_m = pred_arr[:, m, :].copy()
            invis = ~mask[:, m]
            if np.any(invis):
                traj_m[invis] = 1e9
            pred_dict[m] = traj_m

        return pred_dict

    # ---------------------------------------------------------------------
    # Safety filtering (hard constraints)
    # ---------------------------------------------------------------------

    def filter_unsafe_paths(
        self,
        paths: np.ndarray,           # (P, T+1, 2)
        vels: np.ndarray,            # (P, T, 2)
        boxes: List[Any],
        predictions: Dict[Any, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        P, T1, _ = paths.shape
        T = T1 - 1

        boxes = boxes or []
        predictions = predictions or {}

        # (A) Static obstacles (same as before)
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

                transformed = (paths[:, 1:, :] - center) @ R  # (P, T, 2)
                coll = np.any(np.all((transformed >= lb) & (transformed <= ub), axis=-1), axis=-1)
                mask_static_unsafe |= coll

        # (B) Dynamic obstacles with Functional CP envelope (hard)
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        cp_violation = 0.0

        if len(predictions) > 0:
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)  # (M, T_pred, 2)
            if pred_arr.ndim != 3:
                raise ValueError("predictions must be dict of (T_pred,2).")
            pred_arr = pred_arr.transpose(1, 0, 2)  # (T_pred, M, 2)

            T_use = min(T, pred_arr.shape[0])

            for i in range(P):
                for t in range(T_use):
                    x = paths[i, t + 1]
                    d_nom = float(np.min(np.linalg.norm(x[None, :] - pred_arr[t], axis=1)))
                    U = self.evaluate_U(x, t)
                    d_lower = max(d_nom - U, 0.0)

                    if d_lower < self.safe_rad:
                        mask_dynamic_unsafe[i] = True
                        break  # early exit for this path

        mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe], float(cp_violation)

        return None, None, float(cp_violation)

    # ---------------------------------------------------------------------
    # Candidate generation
    # ---------------------------------------------------------------------

    def generate_paths_random(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = int(self.n_steps)
        n_skip = int(self.n_skip)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        n_epochs = int(math.ceil(n_steps / max(1, n_skip)))

        v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs)).astype(np.float32)
        w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs)).astype(np.float32)

        if self.last_best_vels is not None and self.last_best_vels.shape[0] >= 2:
            v_warm = np.append(self.last_best_vels[1:, 0], self.rng.uniform(self.min_v, self.max_v))
            w_warm = np.append(self.last_best_vels[1:, 1], self.rng.uniform(self.min_w, self.max_w))
            v_epoch[0, :] = v_warm[:n_epochs]
            w_epoch[0, :] = w_warm[:n_epochs]

        v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]
        w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]

        paths = np.zeros((self.n_paths, n_steps + 1, 2), dtype=np.float32)
        paths[:, 0, 0] = float(pos_x)
        paths[:, 0, 1] = float(pos_y)

        th = np.full((self.n_paths,), float(orientation_z), dtype=np.float32)
        dt = float(self.dt)

        for t in range(n_steps):
            paths[:, t + 1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(th)
            paths[:, t + 1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(th)
            th = th + dt * w[:, t]

        vels = np.stack([v, w], axis=-1).astype(np.float32)
        return paths, vels

    # ---------------------------------------------------------------------
    # Scoring (goal + control only)
    # ---------------------------------------------------------------------

    def score_paths(
        self,
        paths: np.ndarray,                # (P, T+1, 2)
        vels: np.ndarray,                 # (P, T, 2)
        goal: np.ndarray,                 # (2,)
    ) -> Tuple[int, float]:
        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total_cost = intermediate + terminal + control

        best_idx = int(np.argmin(total_cost))
        best_cost = float(total_cost[best_idx])
        return best_idx, best_cost