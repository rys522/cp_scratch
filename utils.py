from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, Sequence
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
    agent_traj: np.ndarray,
    obst_traj:  np.ndarray,
    obst_pred_traj: np.ndarray,
    true_fields_ego: np.ndarray,
    lower_fields_ego: np.ndarray,
    box: float,
    ego_box: float,           # (사용은 안 하지만 시그니처 유지)
    safe_threshold: float,
    h: int,
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

    T = min(agent_traj.shape[0], T_obs, T_pred,
            true_fields_ego.shape[0], lower_fields_ego.shape[0])
    if T < 2:
        raise ValueError(f"Not enough time steps for animation. Got T={T}.")

    set_mpl_defaults(fontsize=12, tight_layout=False)

    fig, axs = plt.subplots(
        2, 2,
        figsize=(9.5, 8.0),   # (가로를 좀 더 줘야 2x2가 커 보임)
        dpi=180,
        constrained_layout=True
    )
    fig.patch.set_facecolor("white")
    axs = np.array(axs)

    cmap_warn = plt.cm.colors.ListedColormap(['white', 'red'])
    cmap = plt.cm.get_cmap("tab10", M)
    colors = [cmap(i) for i in range(M)]

    def _setup_ax(ax: plt.Axes, title: str):
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_aspect("equal", adjustable="box")  # equal 유지하되 subplot 안에서 최대한 채움
        ax.set_title(title, fontsize=12, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    _setup_ax(axs[0, 0], f"TRUE Dist @ target time (H={h})")
    _setup_ax(axs[0, 1], f"CP lower @ target time (H={h})")
    _setup_ax(axs[1, 0], "TRUE Unsafe (red)")
    _setup_ax(axs[1, 1], "Unsafe: CP (light) vs TRUE (dark)")

    # --- images created once ---
    world_extent = [0, box, 0, box]
    im_true = axs[0, 0].imshow(true_fields_ego[0], extent=world_extent, origin="lower",
                               interpolation="nearest", zorder=0)
    im_low  = axs[0, 1].imshow(lower_fields_ego[0], extent=world_extent, origin="lower",
                               interpolation="nearest", zorder=0)

    im_mask_true = axs[1, 0].imshow((true_fields_ego[0] < safe_threshold).astype(np.float32),
                                    extent=world_extent, origin="lower",
                                    cmap=cmap_warn, vmin=0, vmax=1,
                                    alpha=0.75, interpolation="nearest", zorder=0)

    im_overlay_true = axs[1, 1].imshow((true_fields_ego[0] < safe_threshold).astype(np.float32),
                                       extent=world_extent, origin="lower",
                                       cmap="Blues", vmin=0, vmax=1,
                                       alpha=0.40, interpolation="nearest", zorder=0)
    im_overlay_cp   = axs[1, 1].imshow((lower_fields_ego[0] < safe_threshold).astype(np.float32),
                                       extent=world_extent, origin="lower",
                                       cmap="Reds", vmin=0, vmax=1,
                                       alpha=0.50, interpolation="nearest", zorder=1)

    # --- agent lines/point per subplot ---
    agent_lines = []
    agent_pts = []
    for ax in axs.flat:
        ln, = ax.plot([], [], "b-", lw=2.2, zorder=2)
        pt, = ax.plot([], [], "bo", ms=6, zorder=3)
        agent_lines.append(ln)
        agent_pts.append(pt)

    # --- obstacle true/pred lines & markers per subplot and per obstacle ---
    obst_true_lines = [[None]*M for _ in range(4)]
    obst_pred_lines = [[None]*M for _ in range(4)]
    obst_true_pts   = [[None]*M for _ in range(4)]
    obst_pred_pts   = [[None]*M for _ in range(4)]

    for k, ax in enumerate(axs.flat):
        for m in range(M):
            col = colors[m]
            lt, = ax.plot([], [], "-",  color=col, lw=1.6, alpha=0.95, zorder=2)
            lp, = ax.plot([], [], "--", color=col, lw=1.3, alpha=0.75, zorder=2)
            pt_t, = ax.plot([], [], "x", color=col, ms=8, mew=2.2, zorder=3)
            pt_p, = ax.plot([], [], "o", color=col, ms=5, alpha=0.85, zorder=3)
            obst_true_lines[k][m] = lt
            obst_pred_lines[k][m] = lp
            obst_true_pts[k][m]   = pt_t
            obst_pred_pts[k][m]   = pt_p

    def update(t):
        t = int(t)

        # --- update images ---
        im_true.set_data(true_fields_ego[t])
        im_low.set_data(lower_fields_ego[t])

        mask_true = (true_fields_ego[t] < safe_threshold).astype(np.float32)
        mask_cp   = (lower_fields_ego[t] < safe_threshold).astype(np.float32)
        im_mask_true.set_data(mask_true)
        im_overlay_true.set_data(mask_true)
        im_overlay_cp.set_data(mask_cp)

        # --- update agent ---
        ax_x = agent_traj[:t+1, 0]
        ax_y = agent_traj[:t+1, 1]
        for k in range(4):
            agent_lines[k].set_data(ax_x, ax_y)
            agent_pts[k].set_data([agent_traj[t, 0]], [agent_traj[t, 1]])

        # --- update obstacles ---
        for k in range(4):
            for m in range(M):
                obst_true_lines[k][m].set_data(obst_traj[:t+1, m, 0], obst_traj[:t+1, m, 1])
                obst_pred_lines[k][m].set_data(obst_pred_traj[:t+1, m, 0], obst_pred_traj[:t+1, m, 1])

                obst_true_pts[k][m].set_data([obst_traj[t, m, 0]], [obst_traj[t, m, 1]])
                obst_pred_pts[k][m].set_data([obst_pred_traj[t, m, 0]], [obst_pred_traj[t, m, 1]])

        axs[0, 0].set_title(f"TRUE Dist @ target time (H={h}, t={t})", pad=8)
        axs[0, 1].set_title(f"CP lower @ target time (H={h}, t={t})", pad=8)

        # return list of artists (blit=False라 필수는 아니지만 깔끔)
        arts = [im_true, im_low, im_mask_true, im_overlay_true, im_overlay_cp]
        arts += agent_lines + agent_pts
        for k in range(4):
            for m in range(M):
                arts += [obst_true_lines[k][m], obst_pred_lines[k][m],
                         obst_true_pts[k][m], obst_pred_pts[k][m]]
        return arts

    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04,
                        wspace=0.07, hspace=0.10)

    ani = FuncAnimation(fig, update, frames=range(T), blit=False, interval=interval)
    return ani


def set_mpl_latex_style():
    """
    Matplotlib style that matches LaTeX (Computer Modern).
    Intended for paper-quality figures.
    """

    plt.rcParams.update({
        # Use LaTeX for all text rendering
        "text.usetex": True,

        # Font family
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],

        # Font sizes (paper-friendly)
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Line aesthetics
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,

        # Figure quality
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

def render_cp_comparison_paper_multi(
    agent_traj: np.ndarray,
    obst_traj: np.ndarray,
    obst_pred_traj: np.ndarray,
    true_fields_ego: np.ndarray,
    lower_fields_ego: np.ndarray,
    box: float,
    ego_box: float,                 # signature compatibility (not used here)
    safe_threshold: float,
    h: int,
    headings: np.ndarray = None,     # signature compatibility (optional)
    interval: int = 150,             # signature compatibility (not used here)
    *,
    t: Optional[int] = None,         # frame index to render (default: middle)
    use_target_time: bool = True,    # if True: render at t+h; else at t
    show_points: bool = True,        # robot/obstacle markers
    show_safe_contour: bool = True,  # plot D(x)=r_safe contour
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (6.8, 2.8),
    savepath: Optional[str] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """
    Paper-ready static figure (2 panels): True SDF vs CP lower SDF.

    This is a drop-in *call-style compatible* companion to
    `animate_cp_comparison_multi(...)`, but returns a static figure
    for papers instead of an animation.

    Parameters are intentionally similar to the animation function for convenience.
    """

    set_mpl_latex_style()

    # -----------------------------
    # Normalize obstacle shapes
    # -----------------------------
    if obst_traj.ndim == 2:
        obst_traj = obst_traj[:, np.newaxis, :]
    if obst_pred_traj.ndim == 2:
        obst_pred_traj = obst_pred_traj[:, np.newaxis, :]

    T = min(agent_traj.shape[0], obst_traj.shape[0], obst_pred_traj.shape[0],
            true_fields_ego.shape[0], lower_fields_ego.shape[0])

    if T <= 0:
        raise ValueError("Empty trajectories/fields (T <= 0).")

    # Choose t if not provided
    if t is None:
        t = max(0, (T // 2) - (h if use_target_time else 0))

    t = int(np.clip(t, 0, T - 1))
    tph = t + h if use_target_time else t
    if tph >= T:
        raise ValueError(f"Requested target time t+h={tph} exceeds available T={T}.")

    # Fields to render
    F_true = true_fields_ego[tph]
    F_cp   = lower_fields_ego[tph]

    # -----------------------------
    # Shared color scale
    # -----------------------------
    if vmin is None or vmax is None:
        stacked = np.stack([F_true, F_cp], axis=0)
        if vmin is None:
            vmin = float(np.nanpercentile(stacked, 1.0))
        if vmax is None:
            vmax = float(np.nanpercentile(stacked, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    # -----------------------------
    # Figure
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax0, ax1 = axes
    extent = [0.0, float(box), 0.0, float(box)]

    im0 = ax0.imshow(F_true, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = ax1.imshow(F_cp,   origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)

    # -----------------------------
    # Optional: safety contour D(x)=r_safe
    # -----------------------------
    if show_safe_contour and (safe_threshold is not None):
        Hh, Ww = F_true.shape
        xs = np.linspace(0.0, box, Ww)
        ys = np.linspace(0.0, box, Hh)
        X, Y = np.meshgrid(xs, ys)

        # Left: true SDF threshold
        ax0.contour(
            X, Y, F_true,
            levels=[safe_threshold],
            colors="white",
            linestyles="--",
            linewidths=1.4,
        )

        # Right: true SDF threshold (reference)
        ax1.contour(
            X, Y, F_true,
            levels=[safe_threshold],
            colors="white",
            linestyles="--",
            linewidths=1.2,
            alpha=0.8,
        )

        # Right: conformal lower-bound threshold
        ax1.contour(
            X, Y, F_cp,
            levels=[safe_threshold],
            colors="red",
            linestyles="--",
            linewidths=1.6,
        )
    # -----------------------------
    # Optional: markers (robot + obstacles)
    # -----------------------------
    if show_points:
        robot_xy = agent_traj[tph]
        obst_xy  = obst_traj[tph]  # (M,2)
        for ax in (ax0, ax1):
            ax.plot(robot_xy[0], robot_xy[1], "o", ms=3.5)
            ax.plot(obst_xy[:, 0], obst_xy[:, 1], "x", ms=4.0, mew=1.2)

    # -----------------------------
    # Formatting
    # -----------------------------
    ax0.set_title(r"\textbf{True SDF}", pad=3)
    ax1.set_title(r"\textbf{Conformal lower bound}", pad=3)

    for ax in (ax0, ax1):
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    # One shared colorbar
    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(r"Distance", fontsize=8)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.02)

    return fig, axes