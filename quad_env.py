from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 


def build_grid_3d(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    zs = np.linspace(zlim[0], zlim[1], nz, dtype=np.float32)

    # meshgrid: (nz, ny, nx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")  # gives (ny,nx,nz)
    X = np.transpose(X, (2, 0, 1))  # -> (nz,ny,nx)
    Y = np.transpose(Y, (2, 0, 1))
    Z = np.transpose(Z, (2, 0, 1))
    return xs, ys, zs, X, Y, Z


def distance_field_points_3d(points_xyz: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    points_xyz: (M,3)
    X,Y,Z: (nz,ny,nx)
    returns: D (nz,ny,nx) where D = min_m ||[X,Y,Z]-points[m]||
    """
    if points_xyz.size == 0:
        return np.full_like(X, np.inf, dtype=np.float32)

    pts = points_xyz.astype(np.float32)
    # broadcast: (nz,ny,nx,1) - (1,1,1,M)
    dx = X[..., None] - pts[:, 0][None, None, None, :]
    dy = Y[..., None] - pts[:, 1][None, None, None, :]
    dz = Z[..., None] - pts[:, 2][None, None, None, :]
    d2 = dx*dx + dy*dy + dz*dz
    return np.sqrt(np.min(d2, axis=-1)).astype(np.float32)

def build_training_residuals_from_env_3d(
    env: QuadWorldEnv3D,
    *,
    n_samples: int,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,   # (nz,ny,nx)
    time_horizon: int,
    episode_len: int = 60,
    group_by_episode: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    residuals[n, i] = D_pred_i - D_true_i on 3D grid
    returns:
      residuals: (N, H, nz, ny, nx)
      groups:    (N,)
    """
    Hh = int(time_horizon)
    nz, ny, nx = X.shape

    residuals: List[np.ndarray] = []
    groups: List[int] = []

    ep = 0
    obs = env.reset()
    step_in_ep = 0

    while len(residuals) < n_samples:
        p_dict = obs.get("prediction", {})  # {pid:(H,3)}
        f_dict = obs.get("future", {})      # {pid:(H,3)}

        # pid intersection
        pids = [pid for pid in p_dict.keys() if pid in f_dict]

        if len(pids) > 0:
            res_t = np.zeros((Hh, nz, ny, nx), dtype=np.float32)
            valid_sample = True

            for i in range(Hh):
                pred_pts, true_pts = [], []

                for pid in pids:
                    pr = np.asarray(p_dict[pid], dtype=np.float32)
                    gt = np.asarray(f_dict[pid], dtype=np.float32)

                    # i step exists in both
                    if i < pr.shape[0] and i < gt.shape[0]:
                        pred_pts.append(pr[i])
                        true_pts.append(gt[i])

                if len(pred_pts) == 0:
                    valid_sample = False
                    break

                pred_pts = np.asarray(pred_pts, dtype=np.float32)
                true_pts = np.asarray(true_pts, dtype=np.float32)

                sdf_pred = distance_field_points_3d(pred_pts, X, Y, Z)
                sdf_true = distance_field_points_3d(true_pts, X, Y, Z)
                res_t[i] = sdf_pred - sdf_true

            if valid_sample:
                residuals.append(res_t)
                groups.append(ep if group_by_episode else (len(residuals) - 1))

        # diversify motion (keep your original)
        v = float(np.clip(np.random.normal(0.0, 0.3), -0.6, 0.6))
        w = float(np.clip(np.random.normal(0.0, 0.4), -0.7, 0.7))
        vz = float(np.clip(np.random.normal(0.0, 0.15), -0.3, 0.3))
        obs = env.step(v, w, vz)

        step_in_ep += 1
        if step_in_ep >= episode_len:
            ep += 1
            step_in_ep = 0
            obs = env.reset()

    return np.stack(residuals, axis=0), np.asarray(groups, dtype=np.int64)

def render_env_3d(env: QuadWorldEnv3D, *, steps: int = 200, i_view: int = 7):
    obs = env.reset()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(env.xlim[0], env.xlim[1])
    ax.set_ylim(env.ylim[0], env.ylim[1])
    ax.set_zlim(env.zlim[0], env.zlim[1])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("QuadWorldEnv3D")

    goal = obs["goal_xyz"]
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="x", s=80)

    robot_dot, = ax.plot([], [], [], "bo", markersize=6)
    robot_line, = ax.plot([], [], [], "b-", linewidth=1)

    obs_sc = ax.scatter([], [], [], s=20)
    pred_sc = ax.scatter([], [], [], s=30, marker="x")

    robot_traj = []

    def upd(k):
        nonlocal obs
        robot = obs["robot_xyz"]
        robot_traj.append(robot.copy())
        tr = np.asarray(robot_traj)

        robot_dot.set_data([robot[0]], [robot[1]])
        robot_dot.set_3d_properties([robot[2]])
        robot_line.set_data(tr[:,0], tr[:,1])
        robot_line.set_3d_properties(tr[:,2])

        # obstacle current (history last)
        obs_now = []
        for pid, h in obs["history"].items():
            obs_now.append(h[-1])
        obs_now = np.asarray(obs_now, dtype=np.float32) if len(obs_now) else np.zeros((0,3), dtype=np.float32)

        # predicted points at i_view
        pred_now = []
        for pid, pr in obs["prediction"].items():
            if i_view < pr.shape[0]:
                pred_now.append(pr[i_view])
        pred_now = np.asarray(pred_now, dtype=np.float32) if len(pred_now) else np.zeros((0,3), dtype=np.float32)

        obs_sc._offsets3d = (obs_now[:,0], obs_now[:,1], obs_now[:,2]) if obs_now.size else ([],[],[])
        pred_sc._offsets3d = (pred_now[:,0], pred_now[:,1], pred_now[:,2]) if pred_now.size else ([],[],[])

        # random action just for animation
        v = float(np.clip(np.random.normal(0.2, 0.2), -0.6, 0.6))
        w = float(np.clip(np.random.normal(0.0, 0.3), -0.7, 0.7))
        vz = float(np.clip(np.random.normal(0.0, 0.15), -0.3, 0.3))
        obs = env.step(v, w, vz)
        return []

    ani = FuncAnimation(fig, upd, frames=steps, interval=60, blit=False)
    plt.show()

@dataclass
class ObstacleAgent:
    pid: int
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)
    mode: int = 0          # 0: CV-ish, 1: turn, 2: wander, 3: stop-go
    mode_ttl: int = 0   

class CVPredictor3D:
    def __init__(self, dt: float, horizon: int, process_noise_std: float = 0.0):
        self.dt = float(dt)
        self.H = int(horizon)
        self.noise = float(process_noise_std)

    def predict(self, pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        traj = np.zeros((self.H, 3), dtype=np.float32)
        p = np.asarray(pos, dtype=np.float32).copy()
        v = np.asarray(vel, dtype=np.float32).copy()
        for i in range(self.H):
            v_eff = v
            if self.noise > 0:
                v_eff = v + rng.normal(0, self.noise, size=(3,)).astype(np.float32)
            p = p + v_eff * self.dt
            traj[i] = p
        return traj

@dataclass
class QuadState:
    pos: np.ndarray  # (3,)
    yaw: float

def quad_step_3d_xyyaw_vz(
    st: QuadState,
    v_xy: float,
    yaw_rate: float,
    vz: float,
    dt: float,
    xlim: Tuple[float,float],
    ylim: Tuple[float,float],
    zlim: Tuple[float,float],
) -> QuadState:
    x, y, z = float(st.pos[0]), float(st.pos[1]), float(st.pos[2])
    th = float(st.yaw)

    th2 = th + float(yaw_rate) * dt
    x2 = x + float(v_xy) * np.cos(th2) * dt
    y2 = y + float(v_xy) * np.sin(th2) * dt
    z2 = z + float(vz) * dt

    # robot: clamp
    x2 = float(np.clip(x2, *xlim))
    y2 = float(np.clip(y2, *ylim))
    z2 = float(np.clip(z2, *zlim))
    return QuadState(pos=np.array([x2, y2, z2], dtype=np.float32), yaw=float(th2))

class QuadWorldEnv3D:
    """
    Fully 3D environment:
      - obstacles: 3D GT dynamics + reflect boundaries
      - predictor: CV with noise (model mismatch)
      - oracle future: uses separate RNG stream (reproducible, independent of predictor)
    """
    def __init__(
        self,
        *,
        dt: float,
        horizon: int,
        n_obs: int,
        world_bounds_xyz: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
        seed: int = 0,
        hist_len: int = 2,
        obs_process_noise: float = 0.05,
        pred_model_noise: float = 0.10,
        start_xyz_yaw: Optional[np.ndarray] = None,   # (4,)
        goal_xyz: Optional[np.ndarray] = None,        # (3,)
        gt_future_noise: float = 0.0,                 # oracle future noise
        oracle_seed_offset: int = 12345,              # to separate oracle rng
        mode_switch_p: float = 0.05,     # 매 step mode 바뀔 확률
        mode_min_ttl: int = 5,
        mode_max_ttl: int = 25,
        turn_rate_std: float = 1.2,     # turn 모드에서 방향 변동 강도
        stop_go_p: float = 0.10,            # stop-go 모드에서 멈췄다 출발할 확률
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.hist_len = int(hist_len)

        self.xlim, self.ylim, self.zlim = world_bounds_xyz
        self.obs_process_noise = float(obs_process_noise)
        self.gt_future_noise = float(gt_future_noise)

        # two RNG streams: one for env/predictor, one for oracle future
        self.rng = np.random.default_rng(int(seed))
        self.oracle_rng = np.random.default_rng(int(seed) + int(oracle_seed_offset))

        self.predictor = CVPredictor3D(dt=self.dt, horizon=self.H, process_noise_std=pred_model_noise)

        self.mode_switch_p = float(mode_switch_p)
        self.mode_min_ttl = int(mode_min_ttl)
        self.mode_max_ttl = int(mode_max_ttl)
        self.turn_rate_std = float(turn_rate_std)
        self.stop_go_p = float(stop_go_p)

        # fixed start/goal
        if start_xyz_yaw is None:
            sx = 0.2 * self.xlim[1] + 0.8 * self.xlim[0]
            sy = 0.2 * self.ylim[1] + 0.8 * self.ylim[0]
            sz = 0.2 * self.zlim[1] + 0.8 * self.zlim[0]
            syaw = 0.0
            start_xyz_yaw = np.array([sx, sy, sz, syaw], dtype=np.float32)
        self.start_xyz_yaw = np.asarray(start_xyz_yaw, dtype=np.float32)

        if goal_xyz is None:
            gx = 0.8 * self.xlim[1] + 0.2 * self.xlim[0]
            gy = 0.8 * self.ylim[1] + 0.2 * self.ylim[0]
            gz = 0.8 * self.zlim[1] + 0.2 * self.zlim[0]
            goal_xyz = np.array([gx, gy, gz], dtype=np.float32)
        self.goal_xyz = np.asarray(goal_xyz, dtype=np.float32)

        self.obstacles: List[ObstacleAgent] = []
        self.history_xyz: Dict[int, List[np.ndarray]] = {}

        self.robot: Optional[QuadState] = None
        self.t = 0

        self._init_obstacles(n_obs=n_obs)

    def _init_obstacles(self, n_obs: int):
        self.obstacles = []
        self.history_xyz = {}
        for pid in range(n_obs):
            pos = np.array([
                self.rng.uniform(*self.xlim),
                self.rng.uniform(*self.ylim),
                self.rng.uniform(*self.zlim),
            ], dtype=np.float32)

            vel = self.rng.normal(0, 1, size=(3,)).astype(np.float32)
            vel[2] *= 0.3
            self.obstacles.append(ObstacleAgent(pid=pid, pos=pos, vel=vel))
            self.history_xyz[pid] = [pos.copy()]

    def reset(self, *, reset_obstacles: bool = True) -> Dict:
        if reset_obstacles:
            self._init_obstacles(n_obs=len(self.obstacles))

        x, y, z, yaw = self.start_xyz_yaw
        self.robot = QuadState(pos=np.array([x, y, z], dtype=np.float32), yaw=float(yaw))
        self.t = 0
        return self._make_obs()

    def _reflect_bounds(self, p: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

        if x < self.xlim[0] or x > self.xlim[1]:
            vx *= -1.0
            x = float(np.clip(x, *self.xlim))
        if y < self.ylim[0] or y > self.ylim[1]:
            vy *= -1.0
            y = float(np.clip(y, *self.ylim))
        if z < self.zlim[0] or z > self.zlim[1]:
            vz *= -1.0
            z = float(np.clip(z, *self.zlim))

        return np.array([x, y, z], dtype=np.float32), np.array([vx, vy, vz], dtype=np.float32)

    def _step_obstacles(self):
        for ob in self.obstacles:
            # --- mode switching ---
            if ob.mode_ttl <= 0 or self.rng.random() < self.mode_switch_p:
                ob.mode = int(self.rng.integers(0, 4))
                ob.mode_ttl = int(self.rng.integers(self.mode_min_ttl, self.mode_max_ttl))
            ob.mode_ttl -= 1

            # base acceleration noise
            a = self.rng.normal(0, self.obs_process_noise, size=(3,)).astype(np.float32)
            a[2] *= 0.3

            v = ob.vel.copy()

            if ob.mode == 0:
                # CV-ish: 기존과 비슷
                v = v + a * self.dt

            elif ob.mode == 1:
                # turn: xy평면에서 방향이 계속 휘어짐 (CV가 특히 약함)
                ang = float(self.rng.normal(0, self.turn_rate_std)) * self.dt
                c, s = np.cos(ang), np.sin(ang)
                vx, vy = float(v[0]), float(v[1])
                v[0] = c * vx - s * vy
                v[1] = s * vx + c * vy
                v = v + a * self.dt

            elif ob.mode == 2:
                # wander: 속도 자체를 랜덤 타겟으로 끌어당김 (Ornstein-Uhlenbeck 느낌)
                v_target = self.rng.normal(0, 0.8, size=(3,)).astype(np.float32)
                v_target[2] *= 0.3
                tau = 1.5  # 작을수록 더 급격
                v = v + (v_target - v) * (self.dt / tau) + a * self.dt

            else:
                # stop-go: 확률적으로 거의 멈췄다가 다시 움직임
                if self.rng.random() < self.stop_go_p:
                    v *= 0.1
                else:
                    v = v + a * self.dt

            ob.vel = v
            ob.pos = ob.pos + ob.vel * self.dt
            ob.pos, ob.vel = self._reflect_bounds(ob.pos, ob.vel)

            hist = self.history_xyz[ob.pid]
            hist.append(ob.pos.copy())
            if len(hist) > self.hist_len:
                hist.pop(0)

    def _oracle_future(self, pos0: np.ndarray, vel0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = np.asarray(pos0, dtype=np.float32).copy()
        v = np.asarray(vel0, dtype=np.float32).copy()
        out = np.zeros((self.H, 3), dtype=np.float32)

        for i in range(self.H):
            if self.gt_future_noise > 0:
                a = rng.normal(0, self.gt_future_noise, size=(3,)).astype(np.float32)
                a[2] *= 0.3
                v = v + a * self.dt

            p = p + v * self.dt
            p, v = self._reflect_bounds(p, v)
            out[i] = p
        return out

    def _make_obs(self) -> Dict:
        h_dict = {pid: np.stack(hist, axis=0).astype(np.float32) for pid, hist in self.history_xyz.items()}

        p_dict: Dict[int, np.ndarray] = {}
        f_dict: Dict[int, np.ndarray] = {}

        # IMPORTANT: oracle is independent of predictor rng usage
        for ob in self.obstacles:
            p_dict[ob.pid] = self.predictor.predict(ob.pos, ob.vel, self.rng).astype(np.float32)

            # per-pid deterministic oracle stream
            pid_seed = int(self.oracle_rng.integers(0, 2**31 - 1))
            rng_pid = np.random.default_rng(pid_seed)
            f_dict[ob.pid] = self._oracle_future(ob.pos, ob.vel, rng_pid).astype(np.float32)

        return {
            "t": int(self.t),
            "robot_xyz": self.robot.pos.copy().astype(np.float32),
            "robot_yaw": float(self.robot.yaw),
            "goal_xyz": self.goal_xyz.copy().astype(np.float32),
            "history": h_dict,
            "prediction": p_dict,
            "future": f_dict,
        }

    def step(self, v_xy: float, yaw_rate: float, vz: float = 0.0) -> Dict:
        assert self.robot is not None
        self.robot = quad_step_3d_xyyaw_vz(
            self.robot,
            v_xy=float(v_xy),
            yaw_rate=float(yaw_rate),
            vz=float(vz),
            dt=self.dt,
            xlim=self.xlim,
            ylim=self.ylim,
            zlim=self.zlim,
        )
        self._step_obstacles()
        self.t += 1
        return self._make_obs()