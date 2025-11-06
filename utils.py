from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Sequence

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


# ---------------------------
# Animation helper
# ---------------------------

def animate_cp_comparison(
    robot_traj: np.ndarray,                  # (T, 2)
    obst_traj: np.ndarray,                   # (T, 2)
    true_fields: np.ndarray,                 # (T, H, W)
    lower_forecast_fields: np.ndarray,       # (T, H, W)
    box: float,
    safe_threshold: float,
    h: int,
    frames: Optional[Sequence[int]] = None,  # list/range of decision times t; default = 1..T-h-1
    interval: int = 150,
    figsize=(12, 10),
    titles: Optional[Sequence[str]] = None,
) -> FuncAnimation:
    """
    Make a 2x2 animation comparing true field and forecasted lower bound at (t+h).
    Returns a matplotlib.animation.FuncAnimation (caller may .save(...) if needed).

    Convention:
        - frame variable is decision time t.
        - we display at tpH = t + h.
        - valid t in [1 .. T - h - 1].
    """
    T = true_fields.shape[0]
    if frames is None:
        frames = range(1, T - h)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    if titles is None:
        titles = [
            f"True distance field at t+h",
            f"Forecasted CP lower bound at t+h",
            f"True unsafe mask (d < {safe_threshold})",
            "Unsafe forecast (red) vs True (blue)"
        ]

    def _update(t):
        if t < 1 or t >= T - h:
            return
        tpH = t + h

        for ax in axs.flat:
            ax.cla()

        # (1,1) True field
        plot_distance_field(
            axs[0, 0], true_fields[tpH], box,
            robot_traj=robot_traj, obst_traj=obst_traj, t=tpH,
            title=f"{titles[0]} (t={t}, h={h} → {tpH})"
        )

        # (1,2) Forecasted CP lower-bound field
        plot_distance_field(
            axs[0, 1], lower_forecast_fields[tpH], box,
            robot_traj=robot_traj, obst_traj=obst_traj, t=tpH,
            title=f"{titles[1]} (t={t}, h={h} → {tpH})"
        )

        # (2,1) True unsafe mask
        plot_unsafe_mask(
            axs[1, 0], true_fields[tpH], box, safe_threshold,
            title=f"{titles[2]} (t={t}, h={h} → {tpH})"
        )

        # (2,2) Overlay
        plot_forecast_vs_true(
            axs[1, 1], lower_forecast_fields[tpH], true_fields[tpH], box, safe_threshold,
            title=f"{titles[3]} (t={t}, h={h} → {tpH})"
        )

    ani = FuncAnimation(fig, _update, frames=frames, blit=False, interval=interval)
    return ani