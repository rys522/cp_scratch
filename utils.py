from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Sequence


def build_grid(ego_box: float, H: int, W: int):
    xs = np.linspace(-ego_box/2, ego_box/2, W)
    ys = np.linspace(-ego_box/2, ego_box/2, H)
    Xg, Yg = np.meshgrid(xs, ys)
    return xs, ys, Xg, Yg

# ============================================================
#  Reflection utility
# ============================================================

def reflect_to_box(pos: np.ndarray, box: float) -> np.ndarray:
    """Reflects a 2D point back into [0, box]^2 domain (reflective BC)."""
    pos = np.asarray(pos, dtype=float).copy()
    for i in range(2):
        if pos[i] < 0:
            pos[i] = -pos[i]
        if pos[i] > box:
            pos[i] = 2 * box - pos[i]
        while pos[i] < 0 or pos[i] > box:
            if pos[i] < 0:
                pos[i] = -pos[i]
            if pos[i] > box:
                pos[i] = 2 * box - pos[i]
    return pos

# ============================================================
# Distance field computation
# ============================================================

def distance_field_single_obstacle(point: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance field to a single obstacle point."""
    px, py = point
    return np.sqrt((grid_x - px) ** 2 + (grid_y - py) ** 2).astype(np.float32)

# ---------------------------
# Global matplotlib defaults
# ---------------------------

def set_mpl_defaults(fontsize: int = 12, tight_layout: bool = True) -> None:
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
    })
    if tight_layout:
        plt.rcParams["figure.autolayout"] = True


# ---------------------------
# Small plotting helpers
# ---------------------------

def _set_box_axes(ax: plt.Axes, box: float) -> None:
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect("equal")

def plot_distance_field(
    ax: plt.Axes,
    field: np.ndarray,           # (H, W)
    box: float,
    robot_traj: Optional[np.ndarray] = None,  # (T, 2)
    obst_traj: Optional[np.ndarray] = None,   # (T, 2)
    t: Optional[int] = None,                  # time index for highlighting
    title: Optional[str] = None
) -> None:
    im = ax.imshow(field, extent=[0, box, 0, box], origin="lower")
    if robot_traj is not None:
        Tuse = (t + 1) if (t is not None) else robot_traj.shape[0]
        ax.plot(robot_traj[:Tuse, 0], robot_traj[:Tuse, 1], "b-", lw=1)
        if t is not None:
            ax.plot(robot_traj[t, 0], robot_traj[t, 1], "bo", ms=5)
    if obst_traj is not None:
        Tuse = (t + 1) if (t is not None) else obst_traj.shape[0]
        ax.plot(obst_traj[:Tuse, 0], obst_traj[:Tuse, 1], "r-", lw=1)
        if t is not None:
            ax.plot(obst_traj[t, 0], obst_traj[t, 1], "rx", ms=7, mew=2)
    if title:
        ax.set_title(title)
    _set_box_axes(ax, box)


def distance_field_points(points_rel: np.ndarray, Xg_rel: np.ndarray, Yg_rel: np.ndarray) -> np.ndarray:
    """Min Euclidean distance to one or multiple points, in EGO frame."""
    if points_rel.ndim == 1:  # (2,)
        px, py = points_rel
        d2 = (Xg_rel - px) ** 2 + (Yg_rel - py) ** 2
        return np.sqrt(d2).astype(np.float32)
    # (M,2)
    dmin = None
    for p in points_rel:
        px, py = p
        d2 = (Xg_rel - px) ** 2 + (Yg_rel - py) ** 2
        d = np.sqrt(d2)
        dmin = d if dmin is None else np.minimum(dmin, d)
    return dmin.astype(np.float32)

def plot_unsafe_mask(
    ax: plt.Axes,
    field: np.ndarray,         # (H, W)
    box: float,
    safe_threshold: float,
    title: Optional[str] = None
) -> None:
    mask = (field < safe_threshold).astype(float)
    cmap = plt.cm.colors.ListedColormap(['white', 'red'])
    ax.imshow(mask, extent=[0, box, 0, box], origin="lower", cmap=cmap, vmin=0, vmax=1)
    if title:
        ax.set_title(title)
    _set_box_axes(ax, box)

def plot_forecast_vs_true(
    ax: plt.Axes,
    lower_field: np.ndarray,   # (H, W) predicted lower-bound field
    true_field: np.ndarray,    # (H, W)
    box: float,
    safe_threshold: float,
    title: Optional[str] = None
) -> None:
    # forecasted unsafe if lower_bound < threshold
    mask_forecast = (lower_field < safe_threshold).astype(float)
    mask_true = (true_field < safe_threshold).astype(float)

    ax.imshow(mask_forecast, extent=[0, box, 0, box], origin="lower",
              cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    ax.imshow(mask_true, extent=[0, box, 0, box], origin="lower",
              cmap="Blues", alpha=0.5, vmin=0, vmax=1)
    if title:
        ax.set_title(title)
    _set_box_axes(ax, box)


def world_to_ego(points: np.ndarray, agent_xy: np.ndarray, align_heading=False, heading_rad=0.0):
    """Translate (and optionally rotate) world coords to ego coords centered at agent_xy.
    points: (...,2); agent_xy: (2,)
    returns (...,2)
    """
    rel = points - agent_xy[None, ...] if points.ndim == 2 else points - agent_xy
    if align_heading:
        c, s = np.cos(-heading_rad), np.sin(-heading_rad)
        R = np.array([[c, -s],[s, c]], dtype=np.float32)
        if rel.ndim == 2:
            rel = rel @ R.T
        else:
            rel = rel @ R.T
    return rel


# ---------------------------
# Animation helper
# ---------------------------

def animate_cp_comparison(
    robot_traj: np.ndarray,                  # (T, 2)
    obst_traj: np.ndarray,                   # (T, 2)
    true_fields: np.ndarray,                 # (T, H, W) (Not used directly, but kept for signature)
    lower_forecast_fields: np.ndarray,       # (T, H, W)
    box: float,                              # Spatial boundary
    safe_threshold: float,                   # Distance threshold for unsafe region
    h: int,                                  # Horizon for forecast (t+h)
    Xg: np.ndarray,                          # (H, W) X coordinates of the grid
    Yg: np.ndarray,                          # (H, W) Y coordinates of the grid
    distance_field_single_obstacle: Callable, # Function to compute true distance field
    frames: Optional[Sequence[int]] = None,  # list/range of decision times t; default = 1..T-h-1
    interval: int = 150,                     # Delay between frames in ms
    figsize=(12, 10),
    titles: Optional[Sequence[str]] = None,  # Custom titles (optional)
) -> FuncAnimation:
    """
    Make a 2x2 animation comparing true field and forecasted lower bound at (t+h).
    Returns a matplotlib.animation.FuncAnimation (caller may .save(...) if needed).

    Convention:
        - frame variable is decision time t.
        - we display at tpH = t + h.
        - valid t in [0 .. T - h - 1].
    """
    # Fix 1: Map input names to internal names used in the original code
    Tsteps = robot_traj.shape[0]
    robot_test = robot_traj
    obst_test = obst_traj
    lower_forecast_at_tplusH = lower_forecast_fields

    if frames is None:
        # Valid frames: t from 0 up to Tsteps - h - 1
        # The original code used 1..T-h-1. I'll use 0..T-h-1 for 0-indexing safety,
        # but keep Tsteps - h - 1 as the max valid frame index 't'.
        frames_range = range(0, Tsteps - h)
    else:
        frames_range = frames
        
    # Overwrite the default titles if custom ones are not provided
    if titles is None:
        titles = [
            f"True distance field at t+h",
            f"Forecasted CP lower bound at t+h",
            f"True unsafe mask (d < {safe_threshold})",
            "Unsafe mask: Forecast (red) + True overlay (blue)"
        ]

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    def update(frame: int):
        """
        Animation frame is 't' (decision time). We visualize at time (t+h).
        Valid frames: t ∈ [0 .. Tsteps - h - 1].
        """
        # Ensure frame is within valid range.
        if frame < 0 or frame >= Tsteps - h:
             print(f"Warning: Frame {frame} is out of valid range [0, {Tsteps - h - 1}]. Skipping.")
             return

        for ax in axs.flat:
            ax.cla()

        tpH = frame + h  # time where we compare truth vs forecast

        # --- (1,1) True field at t+h
        # Fix 2: Use the passed-in distance_field_single_obstacle function
        F_true = distance_field_single_obstacle(obst_test[tpH], Xg, Yg)
        axs[0, 0].imshow(F_true, extent=[0, box, 0, box], origin="lower", cmap='viridis') # Added cmap
        axs[0, 0].plot(robot_test[:tpH + 1, 0], robot_test[:tpH + 1, 1], 'b-', lw=1, label='Robot Traj')
        axs[0, 0].plot(obst_test[:tpH + 1, 0],  obst_test[:tpH + 1, 1],  'r-', lw=1, label='Obstacle Traj')
        axs[0, 0].plot(robot_test[tpH, 0], robot_test[tpH, 1], 'bo', ms=5, label='Robot @ t+h')
        axs[0, 0].plot(obst_test[tpH, 0],  obst_test[tpH, 1],  'rx', ms=7, mew=2, label='Obstacle @ t+h')
        axs[0, 0].set_title(f"{titles[0]} (t={frame}, h={h} → {tpH})")
        axs[0, 0].set_xlim([0, box]); axs[0, 0].set_ylim([0, box]); axs[0, 0].set_aspect('equal')

        # --- (1,2) Forecasted CP lower bound field at t+h
        lower_field = lower_forecast_at_tplusH[tpH]
        axs[0, 1].imshow(lower_field, extent=[0, box, 0, box], origin="lower", cmap='viridis') # Added cmap
        axs[0, 1].plot(robot_test[:tpH + 1, 0], robot_test[:tpH + 1, 1], 'b-', lw=1)
        axs[0, 1].plot(obst_test[:tpH + 1, 0],  obst_test[:tpH + 1, 1],  'r-', lw=1)
        axs[0, 1].plot(robot_test[tpH, 0], robot_test[tpH, 1], 'bo', ms=5)
        axs[0, 1].plot(obst_test[tpH, 0],  obst_test[tpH, 1],  'rx', ms=7, mew=2)
        axs[0, 1].set_title(f"{titles[1]} (t={frame}, h={h} → {tpH})")
        axs[0, 1].set_xlim([0, box]); axs[0, 1].set_ylim([0, box]); axs[0, 1].set_aspect('equal')

        # --- (2,1) True unsafe mask at t+h
        cmap = plt.cm.colors.ListedColormap(['white', 'red'])
        mask_true = (F_true < safe_threshold).astype(float)
        axs[1, 0].imshow(mask_true, extent=[0, box, 0, box], origin="lower", cmap=cmap, vmin=0, vmax=1)
        axs[1, 0].set_title(f"{titles[2]} (t={frame}, h={h} → {tpH})")
        axs[1, 0].set_xlim([0, box]); axs[1, 0].set_ylim([0, box]); axs[1, 0].set_aspect('equal')

        # --- (2,2) Forecast vs True unsafe overlay at t+h
        mask_forecast = (lower_field < safe_threshold).astype(float)

        # Using Reds for forecast (lower bound) and Blues for true
        axs[1, 1].imshow(mask_forecast, extent=[0, box, 0, box],
                        origin="lower", cmap=plt.get_cmap("Reds", 2), alpha=0.5, vmin=0, vmax=1)
        # Overlay the true mask (blue region)
        axs[1, 1].imshow(mask_true, extent=[0, box, 0, box],
                        origin="lower", cmap=plt.get_cmap("Blues", 2), alpha=0.5, vmin=0, vmax=1)
        axs[1, 1].set_title(f"{titles[3]} (t={frame}, h={h} → {tpH})")
        axs[1, 1].set_xlim([0, box]); axs[1, 1].set_ylim([0, box]); axs[1, 1].set_aspect('equal')

    # Run animation
    # Fix 3: Use the calculated frames_range for the animation
    ani = FuncAnimation(fig, update, frames=frames_range, blit=False, interval=interval)
    plt.tight_layout() # Added to prevent title overlap
    return ani
