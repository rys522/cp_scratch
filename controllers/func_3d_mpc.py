# controllers/func_cp_mpc_3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
import time
from cp.fuctional_spatial_cp import FunctionalCPParams


@dataclass
class MPC3DWeights:
    w_terminal: float = 10.0
    w_intermediate: float = 1.0
    w_control: float = 0.001

class FunctionalCPMPC3D:
    """
    Baseline 3D CP-MPC (hard constraint only):
      - state: (x,y,z,yaw)
      - control: (v_xy, yaw_rate, vz)
      - dynamic obstacle safety uses:
          d_nom(x_t) - U[t, grid(x_t)] >= safe_rad
    """

    def __init__(
        self,
        *,
        cp_params: FunctionalCPParams,  
        xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,  # grid axes
        n_steps: int,
        dt: float,
        n_skip: int,
        robot_rad: float,
        obstacle_rad: float,
        v_xy_lim: Tuple[float,float],
        yaw_rate_lim: Tuple[float,float],
        vz_lim: Tuple[float,float],
        n_paths: int,
        seed: int = 0,
        weights: Optional[MPC3DWeights] = None,
    ):
        self.cp = cp_params
        if int(n_steps) != int(self.cp.rhos.shape[0]):
            raise ValueError(f"n_steps={n_steps} but cp_params has T={self.cp.rhos.shape[0]}")
    

        self.xs = np.asarray(xs, dtype=np.float32)
        self.ys = np.asarray(ys, dtype=np.float32)
        self.zs = np.asarray(zs, dtype=np.float32)
        self.nx = self.xs.size
        self.ny = self.ys.size
        self.nz = self.zs.size

        self._phi = np.asarray(self.cp.phi_basis, dtype=np.float32)      # (p, D)
        self._coeffs = np.asarray(self.cp.coeffs, dtype=np.float32)      # (T, p)
        self._S_mean = np.asarray(self.cp.S_mean, dtype=np.float32)      
        self._rhos = np.asarray(self.cp.rhos, dtype=np.float32)          
        self._x_star = np.asarray(self.cp.x_star, dtype=np.float32)     

        # grid_shape must match (nz,ny,nx) and flatten order
        if tuple(self.cp.grid_shape) != (int(self.nz), int(self.ny), int(self.nx)):
            raise ValueError(
                f"cp_params.grid_shape={self.cp.grid_shape} but controller grid={(self.nz,self.ny,self.nx)}"
            )

        if self._phi.size == 0 or self._coeffs.size == 0 or self._S_mean is None:
            raise ValueError("cp_params must contain shared-basis fields (phi_basis, coeffs, S_mean).")

        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = int(n_skip)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.vmin, self.vmax = map(float, v_xy_lim)
        self.wmin, self.wmax = map(float, yaw_rate_lim)
        self.vzmin, self.vzmax = map(float, vz_lim)

        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))
        self.weights = weights or MPC3DWeights()

        self.last_best_vels: Optional[np.ndarray] = None  # (T,3)

    # -----------------------------
    # grid lookup: world -> nearest voxel index (k,j,i) = (z,y,x)
    # -----------------------------
    def _world_to_grid_ijk_float(self, p: np.ndarray) -> Optional[Tuple[float,float,float]]:
        x, y, z = float(p[0]), float(p[1]), float(p[2])

        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1] and self.zs[0] <= z <= self.zs[-1]):
            return None

        # searchsorted gives upper index
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

        # return fractional index as (k,j,i) parts
        return (k0 + tz, j0 + ty, i0 + tx)  # floats
    
    def _S_value_at_flat_index(self, t_idx: int, flat_idx: int) -> float:
        """
        S_t(flat_idx) = S_mean[flat_idx] + coeffs[t] @ phi_basis[:, flat_idx]
        where:
          phi_basis: (p, D)
          coeffs:   (T, p)
          S_mean:   (D,)
        """
        t = int(t_idx)
        if not (0 <= t < self.n_steps):
            return 0.0

        # guard
        if self._phi.size == 0:
            raise RuntimeError("Shared basis not available: phi_basis is empty.")
        if flat_idx < 0 or flat_idx >= self._S_mean.shape[0]:
            raise IndexError(f"flat_idx={flat_idx} out of range [0,{self._S_mean.shape[0]-1}]")

        # dot over p only (p is small)
        return float(self._S_mean[flat_idx] + (self._coeffs[t] @ self._phi[:, flat_idx]))
    def _xstar_value_at_flat_index(self, t_idx: int, flat_idx: int) -> float:
        """
        x_star_t(flat_idx) where x_star: (T, D) stored in cp_params.
        """
        t = int(t_idx)
        if not (0 <= t < self.n_steps):
            return 0.0
        if flat_idx < 0 or flat_idx >= self._x_star.shape[1]:
            raise IndexError(f"flat_idx={flat_idx} out of range [0,{self._x_star.shape[1]-1}]")
        return float(self._x_star[t, flat_idx])


    def evaluate_U(self, p: np.ndarray, t_idx: int) -> float:
        """
        One-sided (+) split conformal upper bound for residual Y = D_pred - D_true:

            Y(x) <= max( x_star(x) + rho * S(x), 0 )

        We evaluate x_star and S at p via trilinear interpolation on the (z,y,x) grid.
        """
        idx = int(t_idx)
        if not (0 <= idx < self.n_steps):
            return 0.0

        frac = self._world_to_grid_ijk_float(p)
        if frac is None:
            return float(self.safe_rad)

        kf, jf, if_ = frac
        k0 = int(np.clip(math.floor(kf), 0, self.nz - 2))
        j0 = int(np.clip(math.floor(jf), 0, self.ny - 2))
        i0 = int(np.clip(math.floor(if_), 0, self.nx - 2))
        tz = float(kf - k0)
        ty = float(jf - j0)
        tx = float(if_ - i0)

        # flatten order MUST match reshape(grid_shape) = (nz, ny, nx) with C-order
        def flat(k, j, i):
            return (k * self.ny + j) * self.nx + i

        # ---- 8 corners for S ----
        S000 = self._S_value_at_flat_index(idx, flat(k0,   j0,   i0))
        S001 = self._S_value_at_flat_index(idx, flat(k0,   j0,   i0+1))
        S010 = self._S_value_at_flat_index(idx, flat(k0,   j0+1, i0))
        S011 = self._S_value_at_flat_index(idx, flat(k0,   j0+1, i0+1))
        S100 = self._S_value_at_flat_index(idx, flat(k0+1, j0,   i0))
        S101 = self._S_value_at_flat_index(idx, flat(k0+1, j0,   i0+1))
        S110 = self._S_value_at_flat_index(idx, flat(k0+1, j0+1, i0))
        S111 = self._S_value_at_flat_index(idx, flat(k0+1, j0+1, i0+1))

        # ---- 8 corners for x_star ----
        X000 = self._xstar_value_at_flat_index(idx, flat(k0,   j0,   i0))
        X001 = self._xstar_value_at_flat_index(idx, flat(k0,   j0,   i0+1))
        X010 = self._xstar_value_at_flat_index(idx, flat(k0,   j0+1, i0))
        X011 = self._xstar_value_at_flat_index(idx, flat(k0,   j0+1, i0+1))
        X100 = self._xstar_value_at_flat_index(idx, flat(k0+1, j0,   i0))
        X101 = self._xstar_value_at_flat_index(idx, flat(k0+1, j0,   i0+1))
        X110 = self._xstar_value_at_flat_index(idx, flat(k0+1, j0+1, i0))
        X111 = self._xstar_value_at_flat_index(idx, flat(k0+1, j0+1, i0+1))

        def lerp(a, b, t):
            return a * (1.0 - t) + b * t

        # trilinear for S
        S00 = lerp(S000, S001, tx); S01 = lerp(S010, S011, tx)
        S10 = lerp(S100, S101, tx); S11 = lerp(S110, S111, tx)
        S0  = lerp(S00,  S01,  ty); S1  = lerp(S10,  S11,  ty)
        S   = lerp(S0,   S1,   tz)

        # trilinear for x_star
        X00 = lerp(X000, X001, tx); X01 = lerp(X010, X011, tx)
        X10 = lerp(X100, X101, tx); X11 = lerp(X110, X111, tx)
        X0  = lerp(X00,  X01,  ty); X1  = lerp(X10,  X11,  ty)
        X   = lerp(X0,   X1,   tz)

        S = max(float(S), 1e-9)
        rho = float(self._rhos[idx])

        U = float(X + rho * S)

        # residual upper bound는 음수면 의미 없으니 0으로 clamp
        if U < 0.0:
            U = 0.0

        return U

    # -----------------------------
    # rollout + sampling
    # -----------------------------
    def generate_paths_random(self, x0yz: np.ndarray, yaw0: float) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = self.n_steps
        n_skip = max(1, self.n_skip)
        n_epochs = int(math.ceil(n_steps / n_skip))

        v_epoch = self.rng.uniform(self.vmin, self.vmax, size=(self.n_paths, n_epochs)).astype(np.float32)
        w_epoch = self.rng.uniform(self.wmin, self.wmax, size=(self.n_paths, n_epochs)).astype(np.float32)
        vz_epoch = self.rng.uniform(self.vzmin, self.vzmax, size=(self.n_paths, n_epochs)).astype(np.float32)

        if self.last_best_vels is not None and self.last_best_vels.shape[0] >= 2:
            v_warm = np.append(self.last_best_vels[1:, 0], self.rng.uniform(self.vmin, self.vmax))
            w_warm = np.append(self.last_best_vels[1:, 1], self.rng.uniform(self.wmin, self.wmax))
            vz_warm = np.append(self.last_best_vels[1:, 2], self.rng.uniform(self.vzmin, self.vzmax))
            v_epoch[0, :] = v_warm[:n_epochs]
            w_epoch[0, :] = w_warm[:n_epochs]
            vz_epoch[0, :] = vz_warm[:n_epochs]

        v = np.repeat(v_epoch, n_skip, axis=1)[:, :n_steps]
        w = np.repeat(w_epoch, n_skip, axis=1)[:, :n_steps]
        vz = np.repeat(vz_epoch, n_skip, axis=1)[:, :n_steps]

        paths = np.zeros((self.n_paths, n_steps + 1, 3), dtype=np.float32)
        paths[:, 0, :] = np.asarray(x0yz, dtype=np.float32)[None, :]

        yaw = np.full((self.n_paths,), float(yaw0), dtype=np.float32)
        dt = self.dt

        for t in range(n_steps):
            yaw = yaw + dt * w[:, t]
            paths[:, t+1, 0] = paths[:, t, 0] + dt * v[:, t] * np.cos(yaw)
            paths[:, t+1, 1] = paths[:, t, 1] + dt * v[:, t] * np.sin(yaw)
            paths[:, t+1, 2] = paths[:, t, 2] + dt * vz[:, t]

        vels = np.stack([v, w, vz], axis=-1).astype(np.float32)  # (P,T,3)
        return paths, vels

    # -----------------------------
    # HARD filtering split: static / dynamic
    # -----------------------------
    def filter_static_unsafe_paths(
        self,
        paths: np.ndarray,           # (P, T+1, 3)
        boxes_3d: List[Any],         # optional (baseline empty)
    ) -> np.ndarray:
        # baseline: no static obstacles
        return np.zeros((paths.shape[0],), dtype=bool)

    def filter_dynamic_unsafe_paths(
        self,
        paths: np.ndarray,           # (P, T+1, 3)
        pred_xyz: np.ndarray,        # (H, M, 3)
        pred_mask: np.ndarray,       # (H, M)
    ) -> np.ndarray:
        P, T1, _ = paths.shape
        T = T1 - 1
        H, M, _ = pred_xyz.shape
        T_use = min(T, H)

        unsafe = np.zeros((P,), dtype=bool)

        for t in range(T_use):
            x_t = paths[:, t+1, :]  # (P,3)

            # distances to visible obstacles only
            vis = pred_mask[t]  # (M,)
            if not np.any(vis):
                continue
            pts = pred_xyz[t, vis, :]  # (Mv,3)

            # d_nom: (P,)
            d = np.linalg.norm(x_t[:, None, :] - pts[None, :, :], axis=-1)
            d_nom = np.min(d, axis=1)

            # U(x_t)
            U_vec = np.array([self.evaluate_U(x_t[i], t) for i in range(P)], dtype=np.float32)

            d_lower = np.maximum(d_nom - U_vec, 0.0)
            unsafe |= (d_lower < self.safe_rad)

        return unsafe

    def filter_unsafe_paths(
        self,
        paths: np.ndarray,
        vels: np.ndarray,
        boxes_3d: List[Any],
        pred_xyz: np.ndarray,     # (H,M,3)
        pred_mask: np.ndarray,    # (H,M)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        static_unsafe = self.filter_static_unsafe_paths(paths, boxes_3d)
        if np.all(static_unsafe):
            return None, None

        dyn_unsafe = self.filter_dynamic_unsafe_paths(paths, pred_xyz, pred_mask)
        safe = ~(static_unsafe | dyn_unsafe)
        if not np.any(safe):
            return None, None
        return paths[safe], vels[safe]

    # -----------------------------
    # scoring (goal in 3D, yaw ignored)
    # -----------------------------
    def score_paths(self, paths: np.ndarray, vels: np.ndarray, goal_xyz: np.ndarray) -> Tuple[int, float]:
        goal_xyz = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)
        P, T1, _ = paths.shape
        intermediate = self.weights.w_intermediate * np.sum((paths[:, :-1, :] - goal_xyz) ** 2, axis=(-2, -1))
        terminal = self.weights.w_terminal * np.sum((paths[:, -1, :] - goal_xyz) ** 2, axis=-1)
        control = self.weights.w_control * np.sum(vels ** 2, axis=(-2, -1))
        total = intermediate + terminal + control
        idx = int(np.argmin(total))
        return idx, float(total[idx])

    # -----------------------------
    # public API
    # -----------------------------
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        pred_xyz: np.ndarray,     # (H,M,3)
        pred_mask: np.ndarray,    # (H,M)
        boxes_3d: Optional[List[Any]] = None,
    ):
        boxes_3d = boxes_3d or []

        t0 = time.perf_counter()
        paths, vels = self.generate_paths_random(np.asarray(robot_xyz, dtype=np.float32), float(robot_yaw))
        safe_paths, safe_vels = self.filter_unsafe_paths(paths, vels, boxes_3d, pred_xyz, pred_mask)

        if safe_paths is None:
            return None, {"feasible": False, "final_path": None, "timing_ms": (time.perf_counter()-t0)*1000.0}

        best_idx, best_cost = self.score_paths(safe_paths, safe_vels, np.asarray(goal_xyz, dtype=np.float32))
        self.last_best_vels = safe_vels[best_idx].copy()

        act = safe_vels[best_idx, 0].copy()  # (3,)
        info = {
            "feasible": True,
            "final_path": safe_paths[best_idx],
            "cost": best_cost,
            "timing_ms": (time.perf_counter()-t0)*1000.0,
            "n_safe": int(safe_paths.shape[0]),
            "n_paths": int(paths.shape[0]),
        }
        return act, info