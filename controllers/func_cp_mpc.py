# controllers/func_cp_mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import math
import time

import numpy as np

# NOTE:
# - CPStepParameters is assumed to be produced offline for each horizon index (time-to-go),
#   and cached for online pointwise queries along rollouts.
from cp.functional_cp import CPStepParameters


def min_dist_robot_to_peds(robot_xy, peds_xy):
    if peds_xy.size == 0:
        return float("inf")
    d = peds_xy - robot_xy[None, :]
    return float(np.sqrt(np.sum(d * d, axis=1)).min())

# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class FuncMPCWeights:
    """
    Weights for the MPC objective.

    The controller uses:
      - goal tracking (intermediate + terminal),
      - control effort penalty,
      - soft safety shaping (optional), with a separate adaptive multiplier.

    Parameters
    ----------
    w_terminal : float
        Terminal goal tracking weight.
    w_intermediate : float
        Intermediate goal tracking weight (sum along the horizon).
    w_control : float
        Control effort weight.
    w_safety : float
        Adaptive multiplier (updated online) for the soft safety shaping accumulator.
    w_margin : float
        Weight for the soft barrier against safety margin violation.
    safety_scale : float
        Reserved knob (kept for compatibility); can be used to scale the safety term.
    """
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001

    # soft-safety shaping
    w_safety: float = 10.0
    w_margin: float = 10.0
    safety_scale: float = 0.2


# =============================================================================
# Functional CP-MPC Controller
# =============================================================================

class FunctionalCPMPC:
    """
    Functional CP-informed Monte Carlo MPC controller.

    Summary of the online logic:
      1) Sample candidate control sequences.
      2) Roll out unicycle dynamics to generate candidate paths.
      3) Filter infeasible paths using:
           - static obstacle collision checks, and
           - CP-conformalized distance lower bound to predicted dynamic obstacles.
      4) Score the remaining feasible paths with an MPC objective
         (goal tracking + control effort + optional soft safety shaping),
         and select the first control of the best plan.

    Key idea:
      - Offline, you precompute (per horizon index i) a parameterized upper envelope U_i(x)
        for the residual field, and cache its parameters.
      - Online, you only evaluate U_i(x) at the finite set of rollout states.
    """

    # ---------------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------------

    def __init__(
        self,
        *,
        cp_params: List[Any],
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
        risk_level: float = 1.0,
        step_size: float = 9.,
    ):
        """
        Parameters
        ----------
        cp_params : list
            Offline-cached parameters for the CP envelope, one per horizon index i.
            Each element is expected to have a field `t_idx` used as a dictionary key.
        box, world_center, grid_H, grid_W
            Workspace grid configuration used for fast basis lookup:
              - world -> grid index -> basis vector phi(x).
        n_steps, dt, n_skip
            MPC horizon length, timestep, and control blocking factor.
        robot_rad, obstacle_rad
            Used to define the safety radius r_safe = robot_rad + obstacle_rad.
        min_linear_x, max_linear_x, min_angular_z, max_angular_z
            Sampling bounds for candidate controls.
        n_paths
            Number of Monte Carlo candidate paths per MPC call.
        seed
            RNG seed for reproducibility.
        weights
            MPC objective weights. If None, defaults are used.
        risk_level
            Target slack level used by the adaptive safety-weight update.
        step_size
            Step size for the adaptive update of w_safety.
        """
        # Workspace/grid configuration
        self.box = float(box)
        self.world_center = np.asarray(world_center, dtype=np.float32)
        self.grid_H, self.grid_W = int(grid_H), int(grid_W)

        # Cache CP parameters by horizon index i ("time-to-go")
        self.params: Dict[int, CPStepParameters] = {int(p.t_idx): p for p in cp_params}

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
        self.last_best_vels: Optional[np.ndarray] = None  # warm-start storage

        # Adaptive soft-safety tuning (simple feedback controller)
        self._target_slack = float(risk_level)
        self._eta = float(step_size)

        # -----------------------------
        # Online CP calibration (Plan A)
        # -----------------------------
        # theta_t[idx] is an additive bias for the horizon-wise envelope U at step idx (0-based)
        # where idx=0 corresponds to t+1 in the prediction tensor.
        self.theta_t = np.zeros(self.n_steps, dtype=np.float32)
        self.gamma_t = np.ones(self.n_steps, dtype=np.float32)

        # Target miscoverage level (e.g., 0.1 => 90% coverage)
        self.cp_alpha = 0.1

        # Online step size (learning rate)
        self.cp_eta = 0.01

        # Stability clipping range for theta (distance units)
        self.theta_min = -2.0
        self.theta_max = 2.0

        # Simulation timestep counter
        self.step_count = 0

        # Store past prediction tensors for delayed updates.
        # Each entry: {"time": int, "pred": np.ndarray(H, M, 2)}
        self.prediction_history = deque(maxlen=self.n_steps + 1)

        # Debug prints
        self._cp_debug = True

    # ---------------------------------------------------------------------
    # Grid geometry helpers (world <-> grid)
    # ---------------------------------------------------------------------

    def _world_to_grid_ij(self, pos_world: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Map a world coordinate (x,y) to the nearest grid index (i,j).

        The grid represents coordinates in:
          rel = pos_world - world_center
          rel_x, rel_y in [-box/2, box/2].
        """
        rel = np.asarray(pos_world, dtype=np.float32) - self.world_center
        u = (rel[0] + self.box / 2.0) / self.box * (self.grid_W - 1)
        v = (rel[1] + self.box / 2.0) / self.box * (self.grid_H - 1)

        if not (0.0 <= u <= (self.grid_W - 1) and 0.0 <= v <= (self.grid_H - 1)):
            return None

        j = int(np.rint(u))
        i = int(np.rint(v))
        return (i, j)

    def _grid_flat_index(self, ij: Tuple[int, int]) -> int:
        """Flatten (i,j) index into [0, H*W)."""
        i, j = ij
        return i * self.grid_W + j
    
    def _pred_safety_mask_for_paths(
        self,
        paths: np.ndarray,        # (K, T+1, 2)
        pred_arr: np.ndarray,     # (T_pred, M, 2)
    ) -> np.ndarray:
        """
        Return mask_pred_safe: (K, T_use) boolean
        True  = predicted safe (d_lower >= safe_rad)
        False = predicted unsafe
        """
        K, T1, _ = paths.shape
        T = T1 - 1
        T_use = min(T, pred_arr.shape[0])

        mask = np.ones((K, T_use), dtype=bool)

        for k in range(K):
            for t in range(T_use):
                x = paths[k, t + 1]  # step t position
                d_nom = float(np.min(np.linalg.norm(x[None, :] - pred_arr[t], axis=1)))
                U = self.evaluate_U(x, t)
                d_lower = max(d_nom - U, 0.0)
                mask[k, t] = (d_lower >= self.safe_rad)

        return mask

    # ---------------------------------------------------------------------
    # Adaptive update helper
    # ---------------------------------------------------------------------

    def _update_w_safety_from_slack(
        self,
        min_slack: float,
        *,
        target_slack: float,
        w_min: float = 1e-2,
        w_max: float = 1e4,
    ) -> float:

        if not np.isfinite(min_slack):
            return float(self.weights.w_safety)

        # -----------------------------
        # 1. instantaneous target weight
        # -----------------------------
        raw_w = self._eta * (max(target_slack - float(min_slack), 0.0)) ** 2

        if min_slack > 1.5 * target_slack:
            raw_w = w_min
            self._eta += 0.1

        raw_w = float(np.clip(raw_w, w_min, w_max))

        # -----------------------------
        # 2. smoothing using n_steps
        # -----------------------------
        # effective smoothing horizon ≈ n_steps
        alpha = 1.0 / max(self.n_steps, 1)   # or 2/(n_steps+1)

        w_prev = float(self.weights.w_safety)
        w_new = (1.0 - alpha) * w_prev + alpha * raw_w

        w_new = float(np.clip(w_new, w_min, w_max))
        self.weights.w_safety = w_new

        return w_new
    
    def update_online_cp(self, *, current_obs, robot_pos_world: np.ndarray, safety_pred: bool, idx: int):
        if current_obs is None:
            return
        if not np.all(np.isfinite(np.asarray(current_obs))):
            return

        x_now = np.asarray(robot_pos_world, dtype=np.float32).reshape(2,)
        d_true = float(min_dist_robot_to_peds(x_now, current_obs))
        if not np.isfinite(d_true):
            return

        safety_pred = bool(safety_pred)

        # FP: predicted unsafe but actually safe (over-conservative)
        true_safety = (d_true >= self.safe_rad)

        if true_safety and (not safety_pred):
            self.theta_t[idx] = float(np.clip(self.theta_t[idx] - self.cp_eta, self.theta_min, self.theta_max))
            self.gamma_t[idx] = float(np.clip(self.gamma_t[idx] - 0.02 * idx, 0.0, 5.0))

        # FN: predicted safe but actually unsafe (under-conservative)
        elif not true_safety and safety_pred:
            self.theta_t[idx] = float(np.clip(self.theta_t[idx] + self.cp_eta, self.theta_min, self.theta_max))
            self.gamma_t[idx] = float(np.clip(self.gamma_t[idx] + 0.05 , 0.0, 5.0))

        print(f"[CP Online Update] idx={idx}, d_true={d_true:.3f}, true_safety = {true_safety}, safety_pred={safety_pred}, theta={self.theta_t[idx]:.3f}, gamma={self.gamma_t[idx]:.3f}")

    def _evaluate_U_offline(self, pos_world: np.ndarray, t_idx: int) -> float:
        """Evaluate ONLY the offline envelope U_i(x) (no online theta added)."""
        p = self.params.get(int(t_idx))
        if p is None:
            return 1.0

        ij = self._world_to_grid_ij(pos_world)
        if ij is None:
            return 1.0

        flat = self._grid_flat_index(ij)
        phi_x = p.phi_basis[:, flat].astype(np.float32, copy=False)  # (p_eff,)

        centers = p.mus @ phi_x
        quads = np.einsum("i,kij,j->k", phi_x, p.sigmas, phi_x)
        gamma = 1.0
        idx = int(t_idx)
        if 0 <= idx < self.gamma_t.shape[0]:
            gamma = float(self.gamma_t[idx])

        upper_k = centers + gamma * p.rks * np.sqrt(np.maximum(quads, 0.0))
        return float(p.epsilon + np.max(upper_k))

    # ---------------------------------------------------------------------
    # Functional CP envelope evaluation: U_i(x)
    # ---------------------------------------------------------------------

    def evaluate_U(self, pos_world: np.ndarray, t_idx: int) -> float:
        """
        Evaluate the cached upper envelope U_i(x) at a world position x.

        Expected parameterization in CPStepParameters:
          - phi_basis: shape (p_eff, H*W)   (basis evaluated on grid)
          - mus:       shape (K, p_eff)     (ellipsoid centers in coefficient space)
          - sigmas:    shape (K, p_eff, p_eff)
          - rks:       shape (K,)           (ellipsoid radii)
          - epsilon:   scalar               (projection slack)
        """
        u_offline = self._evaluate_U_offline(pos_world, int(t_idx))
        idx = int(t_idx)
        if 0 <= idx < self.theta_t.shape[0]:
            return float(u_offline + float(self.theta_t[idx]))
        return float(u_offline)
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
        current_obs: Optional[float] = None, 
    ):
        """
        Compute the control action [v, w] for the current state.

        Inputs
        ------
        pos_x, pos_y, orientation_z
            Current robot pose.
        goal
            Goal position as (2,) array-like in world coordinates.
        boxes
            Optional list of static obstacle boxes with fields: pos, w, h, rad.
        predictions
            Optional dict-format dynamic predictions:
              {agent_id: np.ndarray(T_pred, 2)}.

        Alternative input format (trajectory + mask)
        -------------------------------------------
        obst_pred_traj:
            Array with shape (H, M, 2) (or (H,2) for single obstacle).
        obst_mask:
            Visibility mask with shape (H, M) (or (H,) for single).
            Invisible steps are treated as "far away" and thus ignored.
        """
        if goal is None:
            raise ValueError("goal must be provided.")
        goal = np.asarray(goal, dtype=np.float32)

        if current_obs is not None:
            t = int(self.step_count)  

            for h in list(self.prediction_history):
                t0 = int(h["time"])
                i_horizon = t - t0
                idx = i_horizon - 1  

                if not (0 <= idx < self.n_steps):
                    continue

                best_path = h.get("best_path", None)  # (H,2)
                if best_path is not None:
                    best_path = np.asarray(best_path, dtype=np.float32)
                    if 0 <= idx < best_path.shape[0]:
                        self.update_online_cp(
                            current_obs=current_obs,
                            robot_pos_world=best_path[idx],
                            safety_pred=True, 
                            idx=idx,
                        )

                unsafe_hits = h.get("unsafe_hits", None)
                if unsafe_hits is not None:
                    for (t_hit, x_hit) in unsafe_hits:
                        if t_hit == idx:
                            self.update_online_cp(
                                current_obs=current_obs,
                                robot_pos_world=x_hit,
                                safety_pred=False,
                                idx=idx,
                            )

        # Normalize dynamic predictions into dict format if obst_pred_traj was provided.
        if obst_pred_traj is not None:
            predictions = self._normalize_predictions(obst_pred_traj, obst_mask)

        if predictions is None:
            predictions = {}

        boxes = boxes or []

        t0 = time.perf_counter()

        # 1) Sample candidate controls and roll out dynamics
        t_roll0 = time.perf_counter()
        paths, vels = self.generate_paths_random(pos_x, pos_y, orientation_z)
        t_roll1 = time.perf_counter()

        # 2) Hard feasibility filtering (static obstacles + CP-safe distance checks)
        t_filt0 = time.perf_counter()
        safe_paths, safe_vels, cp_violation, unsafe_pts = self.filter_unsafe_paths(paths, vels, boxes, predictions)
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

        # 3) Score feasible candidates and pick the best
        t_score0 = time.perf_counter()
        best_idx, best_cost= self.score_paths(safe_paths, safe_vels, goal, predictions)

        self.last_best_vels = safe_vels[best_idx].copy()
        t_score1 = time.perf_counter()

        # Apply the first control in the best plan
        act = np.asarray(safe_vels[best_idx, 0], dtype=np.float32)
        best_path = safe_paths[best_idx]
        self.last_best_vels = safe_vels[best_idx].copy()

        # pred: (H, M, 2) where pred[0] is t+1
        if obst_pred_traj is not None:
            pred_store = np.asarray(obst_pred_traj, dtype=np.float32)
            if pred_store.ndim == 2:  # (H,2)
                pred_store = pred_store[:, None, :]

            if obst_mask is not None:
                mask = np.asarray(obst_mask, dtype=bool)
                if mask.ndim == 1 and pred_store.shape[1] == 1:
                    mask = mask[:, None]
                if mask.shape == pred_store.shape[:2]:
                    pred_store = pred_store.copy()
                    pred_store[~mask] = 1e9
            best_path_store = best_path[1:, :].copy()  # (H,2)

            # unsafe_pts: list[(t_hit, x_hit)]
            if len(unsafe_pts) > 0:
                sel = self.rng.choice(len(unsafe_pts), size=min(len(unsafe_pts), 10), replace=False)
                unsafe_hits_store = [(int(unsafe_pts[j][0]), unsafe_pts[j][1]) for j in sel]
            else:
                unsafe_hits_store = None

            self.prediction_history.append({
                "time": int(self.step_count),
                "best_path": best_path_store,        # (H,2)
                "unsafe_hits": unsafe_hits_store,    # list of (t_hit, (2,))
            })

        info = {
            "feasible": True,
            "final_path": safe_paths[best_idx],
            "cost": float(best_cost),
            "timing": {
                "total_ms": (time.perf_counter() - t0) * 1000.0,
                "rollout_ms": (t_roll1 - t_roll0) * 1000.0,
                "filter_ms": (t_filt1 - t_filt0) * 1000.0,
                "score_ms": (t_score1 - t_score0) * 1000.0,
            },
            "counts": stats,
            "safety_weight": float(self.weights.w_safety),
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
        """
        Convert (H, M, 2) (+ mask) predictions into dict format:
          {m: (H,2)} in world coordinates.

        Invisible steps are set to a very large value so they do not constrain distances.
        """
        pred_arr = np.asarray(obst_pred_traj, dtype=np.float32)

        # Allow (H,2) for single obstacle
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
                traj_m[invis] = 1e9  # effectively removes obstacle at those steps
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
        """
        Hard feasibility filtering.

        A candidate path is rejected if:
          (A) It collides with any static box obstacle (expanded by robot radius), OR
          (B) It violates dynamic safety when compared against:
                d_lower(x) = max(d_nom(x) - U_i(x), 0).

        Notes
        -----
        - This stage is intentionally strict: it defines the feasible set.
        - Soft safety shaping (used in scoring) is only for ranking within feasible candidates.
        """
        P, T1, _ = paths.shape
        T = T1 - 1

        boxes = boxes or []
        predictions = predictions or {}


        # (A) Static obstacles: OBB collision check in obstacle local frame
        mask_static_unsafe = np.zeros(P, dtype=bool)
        if len(boxes) > 0:
            for box in boxes:
                center = box.pos
                sz = np.array([box.w, box.h], dtype=np.float32)
                th = float(box.rad)

                c, s = np.cos(th), np.sin(th)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                # Expand box by robot radius (Minkowski sum approximation for axis-aligned bounds in local frame)
                lb = -0.5 * sz - self.robot_rad
                ub = 0.5 * sz + self.robot_rad

                # Transform candidate points into obstacle local coordinates
                transformed = (paths[:, 1:, :] - center) @ R  # (P, T, 2)

                coll = np.any(
                    np.all((transformed >= lb) & (transformed <= ub), axis=-1),
                    axis=-1,
                )
                mask_static_unsafe |= coll

        # (B) Dynamic obstacles: CP-conformalized distance lower bound checks
        mask_dynamic_unsafe = np.zeros(P, dtype=bool)
        cp_violation = 0.0  # (reserved: you can accumulate margin violations here if desired)

        if len(predictions) > 0:
            pred_list = list(predictions.values())
            pred_arr = np.asarray(pred_list, dtype=np.float32)  # (M, T_pred, 2)
            if pred_arr.ndim != 3:
                raise ValueError("predictions must be a dict of arrays shaped (T_pred, 2).")
            pred_arr = pred_arr.transpose(1, 0, 2)  # (T_pred, M, 2)

            T_use = min(T, pred_arr.shape[0])

            unsafe_pts = []   # list of (t_hit, x_hit)

            for i in range(P):
                # Early exit once unsafe
                for t in range(T_use):
                    x = paths[i, t + 1]  # position at step t

                    # Nominal distance to predicted obstacles
                    d_nom = float(np.min(np.linalg.norm(x[None, :] - pred_arr[t], axis=1)))

                    # Spatial uncertainty envelope at horizon t
                    U = self.evaluate_U(x, t)

                    # Conformalized lower bound distance (nonnegative)
                    d_lower = max(d_nom - U, 0.0)

                    # Conservative filtering near the first few steps
                    if d_lower < self.safe_rad:
                        mask_dynamic_unsafe[i] = True
                        unsafe_pts.append((t, x.copy()))

        mask_safe = ~(mask_static_unsafe | mask_dynamic_unsafe)
        

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe], float(cp_violation), unsafe_pts


        return None, None, float(cp_violation), None

    # ---------------------------------------------------------------------
    # Candidate generation (Monte Carlo control sampling)
    # ---------------------------------------------------------------------

    def generate_paths_random(
        self,
        pos_x: float,
        pos_y: float,
        orientation_z: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample random piecewise-constant control sequences and roll out unicycle dynamics.

        Implementation details:
          - Control blocking: every `n_skip` steps share the same (v, w).
          - Warm start: if last_best_vels exists, reuse it (shifted) as the first candidate.
        """
        n_steps = int(self.n_steps)
        n_skip = int(self.n_skip)
        if n_steps <= 0:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        # Number of control epochs after blocking
        n_epochs = int(math.ceil(n_steps / max(1, n_skip)))

        # Sample epoch-wise controls
        v_epoch = self.rng.uniform(self.min_v, self.max_v, size=(self.n_paths, n_epochs)).astype(np.float32)
        w_epoch = self.rng.uniform(self.min_w, self.max_w, size=(self.n_paths, n_epochs)).astype(np.float32)

        # Warm start (first candidate)
        if self.last_best_vels is not None and self.last_best_vels.shape[0] >= 2:
            v_warm = np.append(self.last_best_vels[1:, 0], self.rng.uniform(self.min_v, self.max_v))
            w_warm = np.append(self.last_best_vels[1:, 1], self.rng.uniform(self.min_w, self.max_w))
            v_epoch[0, :] = v_warm[:n_epochs]
            w_epoch[0, :] = w_warm[:n_epochs]

        # Expand to per-step controls and truncate to horizon length
        v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)
        w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]  # (P, n_steps)

        # Roll out positions
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

    # ---------------------------------------------------------------------
    # Scoring (MPC objective over feasible paths)
    # ---------------------------------------------------------------------

    def score_paths(
        self,
        paths: np.ndarray,                # (P, T+1, 2)
        vels: np.ndarray,                 # (P, T, 2)
        goal: np.ndarray,                 # (2,)
        predictions: Dict[Any, np.ndarray],
    ) -> Tuple[int, float, float]:
        """
        Score feasible candidate paths.

        Base costs (standard MPC):
          - intermediate goal tracking,
          - terminal goal tracking,
          - control effort.

        Optional soft safety shaping (only if predictions exist):
          - penalize safety margin violations using a smooth barrier on (safe_rad - d_lower),
          - include an urgency weight to emphasize earlier steps,
          - compute a "soft minimum slack" summary used to adapt w_safety online.

        Returns
        -------
        best_idx : int
            Index of the minimum-cost path.
        best_cost : float
            Minimum total cost value.
        best_min_slack : float
            Slack-like scalar for the selected path (used for adaptive tuning).
        """
        P, T1, _ = paths.shape
        T = T1 - 1

        # Standard MPC terms
        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total_cost = intermediate + terminal + control

        best_idx = int(np.argmin(total_cost))
        best_cost = float(total_cost[best_idx])

        return best_idx, best_cost