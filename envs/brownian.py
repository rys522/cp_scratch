from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from utils import reflect_to_box, distance_field_single_obstacle

"""
Brownian motion environment utilities for safe-forecast experiments.

Provides:
  - 2D Brownian motion with reflecting boundaries
  - Distance field computation on uniform grid
  - Unified environment wrapper with motion + grid parameters
"""

# ============================================================
#  Brownian default parameters
# ============================================================

DEFAULT_ENV_PARAMS = dict(
    H=64,
    W=64,
    box=100.0,          # spatial domain size
    dt=1.0,             # time step size
    sigma_robot=4.0,    # diffusion scale for robot
    sigma_obst=4.0,     # diffusion scale for obstacle
    x0_robot=(70.0, 30.0),  # initial robot position
    x0_obst=(30.0, 70.0),   # initial obstacle position
)


# ============================================================
# Brownian motion simulator
# ============================================================

def simulate_brownian_2d(
    n_steps: int,
    dt: float,
    sigma: float,
    x0: Tuple[float, float],
    box: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Simulate a 2D Brownian trajectory with reflecting boundaries."""
    traj = np.zeros((n_steps, 2), dtype=float)
    x = np.array(x0, dtype=float)
    traj[0] = reflect_to_box(x.copy(), box)
    for t in range(1, n_steps):
        step = rng.normal(0.0, sigma * np.sqrt(dt), size=2)
        x = reflect_to_box(x + step, box)
        traj[t] = x
    return traj





# ============================================================
#  BrownianEnvironment class
# ============================================================

@dataclass
class BrownianEnvironment:
    """Unified environment for 2D Brownian motion + grid utilities."""
    H: int = DEFAULT_ENV_PARAMS["H"]
    W: int = DEFAULT_ENV_PARAMS["W"]
    box: float = DEFAULT_ENV_PARAMS["box"]
    dt: float = DEFAULT_ENV_PARAMS["dt"]
    sigma_robot: float = DEFAULT_ENV_PARAMS["sigma_robot"]
    sigma_obst: float = DEFAULT_ENV_PARAMS["sigma_obst"]
    x0_robot: Tuple[float, float] = DEFAULT_ENV_PARAMS["x0_robot"]
    x0_obst: Tuple[float, float] = DEFAULT_ENV_PARAMS["x0_obst"]

    def build_grid(self):
        """Creates a uniform (x, y) mesh grid."""
        xs = np.linspace(0, self.box, self.W)
        ys = np.linspace(0, self.box, self.H)
        Xg, Yg = np.meshgrid(xs, ys)
        return xs, ys, Xg, Yg

    def make_rng(self, seed: Optional[int] = None):
        """Create a reproducible RNG."""
        return np.random.default_rng(seed)

    def simulate_pair(self, Tsteps: int, rng: np.random.Generator):
        """Simulate both robot and obstacle trajectories."""
        r_traj = simulate_brownian_2d(Tsteps, self.dt, self.sigma_robot, self.x0_robot, self.box, rng)
        o_traj = simulate_brownian_2d(Tsteps, self.dt, self.sigma_obst,  self.x0_obst,  self.box, rng)
        return r_traj.astype(np.float32), o_traj.astype(np.float32)

    def distance_fields_for_traj(self, traj, Xg, Yg, dtype=np.float32):
        """Compute distance field for all steps in a trajectory."""
        T = traj.shape[0]
        fields = np.zeros((T, self.H, self.W), dtype=dtype)
        for t in range(T):
            fields[t] = distance_field_single_obstacle(traj[t], Xg, Yg)
        return fields

    def simulate_dataset(
        self,
        N: int,
        Tsteps: int,
        seed: Optional[int] = 2023,
        return_fields: bool = True,
        dtype=np.float32
    ):
        """Simulate N episodes (optionally include full distance fields)."""
        rng = self.make_rng(seed)
        _, _, Xg, Yg = self.build_grid()

        robot_trajs = np.zeros((N, Tsteps, 2), dtype=dtype)
        obst_trajs  = np.zeros((N, Tsteps, 2), dtype=dtype)
        fields = np.zeros((N, Tsteps, self.H, self.W), dtype=dtype) if return_fields else None

        for i in range(N):
            r_traj, o_traj = self.simulate_pair(Tsteps, rng)
            robot_trajs[i], obst_trajs[i] = r_traj, o_traj
            if return_fields:
                fields[i] = self.distance_fields_for_traj(o_traj, Xg, Yg, dtype=dtype)

        return (robot_trajs, obst_trajs, fields) if return_fields else (robot_trajs, obst_trajs)

    def simulate_test_episode(self, Tsteps: int, seed: Optional[int] = 999, with_fields=True, dtype=np.float32):
        """Simulate a single test episode for visualization/evaluation."""
        rng = self.make_rng(seed)
        r_traj, o_traj = self.simulate_pair(Tsteps, rng)
        if not with_fields:
            return r_traj, o_traj
        _, _, Xg, Yg = self.build_grid()
        fields = self.distance_fields_for_traj(o_traj, Xg, Yg, dtype=dtype)
        return r_traj, o_traj, fields

    def set_params(self, **kwargs):
        """Dynamically update grid or motion parameters."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown parameter '{k}'")
            setattr(self, k, v)
        return self