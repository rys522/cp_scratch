from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional
from matplotlib.transforms import Affine2D



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

def reflect_vectorized(pts: np.ndarray, box: float) -> np.ndarray:

    pts = np.abs(pts)
    
    mask_upper = pts > box
    if np.any(mask_upper):
        pts[mask_upper] = 2.0 * box - pts[mask_upper]
        
    return pts

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
    agent_traj: np.ndarray,            # (T,2) robot in WORLD
    obst_traj:  np.ndarray,            # (T,2) obstacle in WORLD
    true_fields_ego: np.ndarray,       # (T,hH,hW) TRUE distance field on EGO grid at time t
    lower_fields_ego: np.ndarray,      # (T,hH,hW) CP lower-bound field on EGO grid at time t
    box: float,                        # WORLD domain [0,box]^2 (for axes limits)
    ego_box: float,                    # EGO window size [-L/2,L/2]^2 placed at robot pose
    safe_threshold: float,
    h: int,
    headings: np.ndarray = None,       # (T,) yaw [rad]; if None, no rotation
    interval: int = 150
):
    """
    Visualize ego-centric TRUE/CP rasters by painting them into the WORLD axes.
    Frame = t (decision time), we compare/display at tph = t + h.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = np.array(axs)

    T = min(agent_traj.shape[0], obst_traj.shape[0],
            true_fields_ego.shape[0], lower_fields_ego.shape[0])
    if T <= h + 1:
        raise ValueError(f"Not enough time steps for horizon h={h}. Got T={T}.")

    world_extent = [0, box, 0, box]
    cmap_warn = plt.cm.colors.ListedColormap(['white', 'red'])

    def draw_ego_raster(ax, raster_ego, x, y, heading_rad=0.0, alpha=1.0,
                        cmap=None, vmin=None, vmax=None):
        L = float(ego_box)
        ego_extent = [-L/2, L/2, -L/2, L/2]  # ego coords
        im = ax.imshow(raster_ego, extent=ego_extent, origin="lower",
                       cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        trans = Affine2D().rotate(heading_rad).translate(x, y) + ax.transData
        im.set_transform(trans)
        im.set_clip_on(True)
        return im

    def update(t):
        if t < 1 or t >= T - h:
            return
        for ax in axs.flat:
            ax.cla()

        tph = t + h
        x, y = agent_traj[tph]
        yaw = float(headings[tph]) if headings is not None else 0.0

        # ---------- (1,1) TRUE distance field (EGO) placed in WORLD ----------
        F_true_ego = true_fields_ego[tph]
        draw_ego_raster(axs[0, 0], F_true_ego, x, y, heading_rad=yaw, alpha=1.0)
        axs[0, 0].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[0, 0].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[0, 0].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[0, 0].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 0].set_title(f"TRUE distance (ego → world) at t+h (t={t} → {tph})")
        axs[0, 0].set_xlim([0, box])
        axs[0, 0].set_ylim([0, box])
        axs[0, 0].set_aspect('equal')

        # ---------- (1,2) CP lower field (EGO) placed in WORLD ----------
        lower_ego = lower_fields_ego[tph]
        draw_ego_raster(axs[0, 1], lower_ego, x, y, heading_rad=yaw, alpha=1.0)
        axs[0, 1].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[0, 1].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[0, 1].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[0, 1].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[0, 1].set_title(f"CP lower (ego → world) at t+h (t={t} → {tph})")
        axs[0, 1].set_xlim([0, box])
        axs[0, 1].set_ylim([0, box])
        axs[0, 1].set_aspect('equal')

        # ---------- (2,1) TRUE unsafe mask (EGO) placed in WORLD ----------
        mask_true = (F_true_ego < safe_threshold).astype(float)
        draw_ego_raster(axs[1, 0], mask_true, x, y, heading_rad=yaw,
                        alpha=1.0, cmap=cmap_warn, vmin=0, vmax=1)
        axs[1, 0].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[1, 0].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[1, 0].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[1, 0].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        axs[1, 0].set_title("TRUE unsafe mask (ego → world)")
        axs[1, 0].set_xlim([0, box])
        axs[1, 0].set_ylim([0, box])
        axs[1, 0].set_aspect('equal')

        # ---------- (2,2) Overlay: CP unsafe (ego) vs TRUE unsafe (ego), both in WORLD ----------
        axs[1, 1].plot(agent_traj[:tph+1, 0], agent_traj[:tph+1, 1], 'b-', lw=1)
        axs[1, 1].plot(obst_traj[:tph+1, 0],  obst_traj[:tph+1, 1],  'r-', lw=1)
        axs[1, 1].plot(agent_traj[tph, 0], agent_traj[tph, 1], 'bo', ms=5)
        axs[1, 1].plot(obst_traj[tph, 0],  obst_traj[tph, 1],  'rx', ms=7, mew=2)
        # TRUE unsafe (blue)
        draw_ego_raster(axs[1, 1], mask_true, x, y, heading_rad=yaw,
                        alpha=0.5, cmap="Blues", vmin=0, vmax=1)
        # CP unsafe (red)
        mask_forecast = (lower_ego < safe_threshold).astype(float)
        draw_ego_raster(axs[1, 1], mask_forecast, x, y, heading_rad=yaw,
                        alpha=0.5, cmap="Reds", vmin=0, vmax=1)
        axs[1, 1].set_title("Unsafe: CP (red, ego→world) vs TRUE (blue, ego→world)")
        axs[1, 1].set_xlim([0, box])
        axs[1, 1].set_ylim([0, box])
        axs[1, 1].set_aspect('equal')

    ani = FuncAnimation(fig, update, frames=range(1, T - h), blit=False, interval=interval)
    return ani

def animate_cp_comparison_multi(
    agent_traj: np.ndarray,            # (T, 2)
    obst_traj:  np.ndarray,            # (T, M, 2) TRUE obstacle traj at target times
    obst_pred_traj: np.ndarray,        # (T, M, 2) PRED obstacle traj at same target times
    true_fields_ego: np.ndarray,       # (T, H, W) TRUE distance field at target times
    lower_fields_ego: np.ndarray,      # (T, H, W) CP lower field at target times
    box: float,
    ego_box: float,
    safe_threshold: float,
    h: int,                            # time horizon label only (for title)
    headings: np.ndarray = None,
    interval: int = 150
):

    # Ensure 3D for obstacle trajectories
    if obst_traj.ndim == 2:
        obst_traj = obst_traj[:, np.newaxis, :]
    if obst_pred_traj.ndim == 2:
        obst_pred_traj = obst_pred_traj[:, np.newaxis, :]

    T_obs, M, _ = obst_traj.shape
    T_pred, M2, _ = obst_pred_traj.shape
    assert M == M2, "M must match between true and pred traj"

    T = min(
        agent_traj.shape[0],
        T_obs,
        T_pred,
        true_fields_ego.shape[0],
        lower_fields_ego.shape[0],
    )

    if T < 2:
        raise ValueError(f"Not enough time steps for animation. Got T={T}.")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = np.array(axs)

    # Color palette for each obstacle m
    cmap = plt.cm.get_cmap("tab10", M)
    colors = [cmap(i) for i in range(M)]

    cmap_warn = plt.cm.colors.ListedColormap(['white', 'red'])

    def plot_obstacles(ax, t_idx):
        """True and predicted traj for all obstacles at target time index t_idx."""
        for m in range(M):
            col = colors[m]

            # TRUE history
            ax.plot(
                obst_traj[:t_idx+1, m, 0],
                obst_traj[:t_idx+1, m, 1],
                '-', color=col, lw=1.5, alpha=0.9
            )
            # TRUE current
            ax.plot(
                obst_traj[t_idx, m, 0],
                obst_traj[t_idx, m, 1],
                'x', color=col, ms=7, mew=2
            )

            # PRED history (same color, dashed)
            ax.plot(
                obst_pred_traj[:t_idx+1, m, 0],
                obst_pred_traj[:t_idx+1, m, 1],
                '--', color=col, lw=1.2, alpha=0.7
            )
            # PRED current (circle marker)
            ax.plot(
                obst_pred_traj[t_idx, m, 0],
                obst_pred_traj[t_idx, m, 1],
                'o', color=col, ms=5, alpha=0.8
            )

    def common_setup(ax, t_idx):
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_aspect('equal')
        # agent history & current
        ax.plot(agent_traj[:t_idx+1, 0], agent_traj[:t_idx+1, 1],
                'b-', lw=2.0)
        ax.plot(agent_traj[t_idx, 0], agent_traj[t_idx, 1],
                'bo', ms=6)
        plot_obstacles(ax, t_idx)

    def update(t):
        # t는 target time index (0,...,T-1)
        for ax in axs.flat:
            ax.cla()

        # ---------------------
        # (0,0) TRUE distance
        # ---------------------
        ax1 = axs[0, 0]
        common_setup(ax1, t)
        ax1.imshow(
            true_fields_ego[t],
            extent=[0, box, 0, box],
            origin='lower'
        )
        ax1.set_title(f"TRUE Dist @ target time (H={h})")

        # ---------------------
        # (0,1) CP lower
        # ---------------------
        ax2 = axs[0, 1]
        common_setup(ax2, t)
        ax2.imshow(
            lower_fields_ego[t],
            extent=[0, box, 0, box],
            origin='lower'
        )
        ax2.set_title(f"CP lower @ target time (H={h})")

        # ---------------------
        # (1,0) TRUE unsafe mask
        # ---------------------
        ax3 = axs[1, 0]
        common_setup(ax3, t)
        mask_true = (true_fields_ego[t] < safe_threshold).astype(float)
        ax3.imshow(
            mask_true,
            extent=[0, box, 0, box],
            origin='lower',
            cmap=cmap_warn,
            vmin=0, vmax=1,
            alpha=0.7
        )
        ax3.set_title("TRUE Unsafe (red)")

        # ---------------------
        # (1,1) TRUE vs CP unsafe
        # ---------------------
        ax4 = axs[1, 1]
        common_setup(ax4, t)
        # TRUE (blue)
        ax4.imshow(
            mask_true,
            extent=[0, box, 0, box],
            origin='lower',
            cmap="Blues",
            vmin=0, vmax=1,
            alpha=0.4
        )
        # CP (red)
        mask_cp = (lower_fields_ego[t] < safe_threshold).astype(float)
        ax4.imshow(
            mask_cp,
            extent=[0, box, 0, box],
            origin='lower',
            cmap="Reds",
            vmin=0, vmax=1,
            alpha=0.5
        )
        ax4.set_title("Unsafe: CP (red) vs TRUE (blue)")

    ani = FuncAnimation(fig, update, frames=range(T),
                        blit=False, interval=interval)
    return ani