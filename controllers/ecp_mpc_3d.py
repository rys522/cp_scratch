from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

DISTANCE_BOUND = 10000.0


# ============================================================
# 3D distance utilities (dependency-free)
# ============================================================
def compute_pairwise_distances_3d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x: (P,3)
    y: (M,3)
    return: (P,M) distances
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.full((x.shape[0], 0), np.inf, dtype=np.float32)
    d = x[:, None, :] - y[None, :, :]
    return np.linalg.norm(d, axis=-1)


def compute_pairwise_distances_along_axis_3d(paths: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    paths: (P,T,3)
    obs:   (M,T,3)  (deterministic)  or (M,S,T,3) (stochastic samples)
    return:
      deterministic -> (T,P,M)
      stochastic    -> (T,P,M,S)  (we'll min over (M,S) as needed)
    """
    paths = np.asarray(paths, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)

    P, T, _ = paths.shape

    if obs.ndim == 3:
        # obs: (M,T,3)
        M = obs.shape[0]
        out = np.zeros((T, P, M), dtype=np.float32)
        for t in range(T):
            d = paths[:, t, :][:, None, :] - obs[:, t, :][None, :, :]
            out[t] = np.linalg.norm(d, axis=-1)
        return out

    if obs.ndim == 4:
        # obs: (M,S,T,3)
        M, S, T2, _ = obs.shape
        assert T2 == T
        out = np.zeros((T, P, M, S), dtype=np.float32)
        for t in range(T):
            # (P,1,1,3) - (1,M,S,3) -> (P,M,S,3)
            d = paths[:, t, :][:, None, None, :] - obs[:, :, t, :][None, :, :, :]
            out[t] = np.linalg.norm(d, axis=-1)
        return out

    raise ValueError(f"obs must have ndim 3 or 4, got {obs.ndim}")


def quantile_higher(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    elementwise 'higher' quantile along last axis.
    x: (..., B)
    q: (...) in [0,1]
    return: (...)
    """
    x = np.asarray(x, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    B = x.shape[-1]
    xs = np.sort(x, axis=-1)
    k = np.ceil(q * B).astype(np.int32) - 1
    k = np.clip(k, 0, B - 1)
    # take along last axis
    return np.take_along_axis(xs, k[..., None], axis=-1)[..., 0]


# ============================================================
# Dict packing helpers (QuadWorldEnv3D-compatible)
# ============================================================
def process_obs_history_to_last(o: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    env.obs["history"]: dict(pid -> (hist_len,3))
    -> dict(pid -> (3,))  (most recent only)
    """
    return {pid: np.asarray(traj[-1], dtype=np.float32) for pid, traj in o.items()}


def pred_dict_to_numpy(pred: Dict[int, np.ndarray]) -> np.ndarray:
    """
    pred dict(pid -> (H,3)) -> (M,H,3)
    """
    if not pred:
        return np.zeros((0, 0, 3), dtype=np.float32)
    arrs = [np.asarray(v, dtype=np.float32) for v in pred.values()]
    return np.stack(arrs, axis=0).astype(np.float32)


def obs_dict_to_numpy_last(obs_last: Dict[int, np.ndarray]) -> np.ndarray:
    """
    obs_last dict(pid -> (3,)) -> (M,3)
    """
    if not obs_last:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([np.asarray(v, dtype=np.float32) for v in obs_last.values()], axis=0)


def compute_min_dist_3d(paths: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    paths: (P,T,3)
    obs:   (M,T,3) or (M,S,T,3)
    return: (P,T)  min distance to any obstacle (and sample if S exists)
    """
    D = compute_pairwise_distances_along_axis_3d(paths, obs)
    if D.ndim == 3:
        # (T,P,M) -> min over M -> (T,P) -> transpose
        return np.min(D, axis=-1).T
    else:
        # (T,P,M,S) -> min over (M,S) -> (T,P) -> transpose
        return np.min(D, axis=(-2, -1)).T


# ============================================================
# EgocentricCPMPC3D
# ============================================================
class EgocentricCPMPC3D:
    """
    3D version of Egocentric CP-MPC (ECP-MPC) compatible with QuadWorldEnv3D.

    - state: (x,y,z,yaw)
    - control: (v_xy, yaw_rate, vz)
    - predictions: dict(pid -> (H,3))  (deterministic)
      (stochastic version: dict(pid -> (S,H,3)) also supported if you pass that shape)
    """

    def __init__(
        self,
        n_steps: int = 12,
        dt: float = 0.4,
        min_linear_x: float = -0.8,
        max_linear_x: float = 0.8,
        min_angular_z: float = -0.7,
        max_angular_z: float = 0.7,
        vz_min: float = -0.3,
        vz_max: float = 0.3,
        n_skip: int = 4,
        robot_rad: float = 0.4,
        obstacle_rad: float = 1.0 / np.sqrt(2.0),
        calibration_set_size: int = 10,
        miscoverage_level: float = 0.1,
        step_size: float = 0.05,
        # discretization points (keep small like original)
        v_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
        w_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
        vz_points: Tuple[float, ...] = (-1.0, 0.0, 1.0),
    ):
        self._n_steps = int(n_steps)
        self._dt = float(dt)
        self._miscoverage_level = float(miscoverage_level)

        self.max_linear_x = float(max_linear_x)
        self.min_linear_x = float(min_linear_x)

        self.max_angular_z = float(max_angular_z)
        self.min_angular_z = float(min_angular_z)

        self.vz_min = float(vz_min)
        self.vz_max = float(vz_max)

        self.n_skip = int(n_skip)

        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.calibration_set_size = int(calibration_set_size)
        self._gamma = float(step_size)

        # discrete action grid (scaled to limits)
        self._v_grid = np.array(v_points, dtype=np.float32) * self.max_linear_x
        self._w_grid = np.array(w_points, dtype=np.float32) * self.max_angular_z
        self._vz_grid = np.array(vz_points, dtype=np.float32) * self.vz_max

        # count paths like original (grid^epochs)
        n_decision_epochs = self._n_steps // self.n_skip
        n_points = self._v_grid.size * self._w_grid.size * self._vz_grid.size
        self._n_paths = int(n_points ** n_decision_epochs)

        # alpha_t per-path/per-step (ACI-style)
        self.alpha_t = self._miscoverage_level * np.ones((self._n_paths, self._n_steps), dtype=np.float32)

        # history
        self.path_history: List[np.ndarray] = []
        self.quantile_history: List[np.ndarray] = []

        self._prediction_queue: List[Dict[int, np.ndarray]] = []
        self._track_queue: List[Dict[int, np.ndarray]] = []  # each is dict(pid->(3,)) at time t

    # -----------------------------
    # external loop hooks
    # -----------------------------
    def update_predictions(self, prediction_result: Dict[int, np.ndarray]) -> None:
        """
        prediction_result: dict(pid -> (H,3))  OR dict(pid -> (S,H,3))
        (we keep it as-is)
        """
        self._prediction_queue.append(prediction_result)

    def update_observations(self, obs_history: Dict[int, np.ndarray]) -> np.ndarray:
        """
        obs_history: env.obs["history"] dict(pid -> (hist_len,3))
        returns err matrix used in ACI update: (n_paths, n_steps)
        """
        n_paths = self.alpha_t.shape[0]

        if not obs_history:
            self._track_queue.append({})
            return np.zeros((n_paths, self._n_steps), dtype=np.float32)

        obs_last = process_obs_history_to_last(obs_history)

        quantiles = []
        min_dist_obs = []
        min_dist_pred = []

        n_data = len(self.quantile_history)  # stored for times where __call__ was run
        max_n_steps = min(self._n_steps + 1, n_data)

        for i in range(1, max_n_steps):
            pred = self._prediction_queue[-i]  # dict(pid->traj)

            obs_nodes = set(obs_last.keys())
            pred_nodes = set(pred.keys())
            common_nodes = list(obs_nodes & pred_nodes)

            if common_nodes:
                # obs at time t: (|V|,3)
                obs_np = np.stack([obs_last[n] for n in common_nodes], axis=0).astype(np.float32)

                # candidate paths used at time t-i: (n_paths, n_steps+1, 3)
                paths = self.path_history[-i]
                x_i = paths[:, i, :]  # (n_paths,3)

                min_dist_obs_i = np.min(compute_pairwise_distances_3d(x_i, obs_np), axis=-1)

                # pred at time t-i for step i: supports deterministic (H,3) and stochastic (S,H,3)
                pred_list = []
                for n in common_nodes:
                    arr = np.asarray(pred[n], dtype=np.float32)
                    pred_list.append(arr)
                # now pred_list elements are (H,3) or (S,H,3)
                arr0 = pred_list[0]
                if arr0.ndim == 2:
                    # deterministic -> take (i-1)
                    pred_i = np.stack([a[i - 1, :] for a in pred_list], axis=0)  # (|V|,3)
                    min_dist_pred_i = np.min(compute_pairwise_distances_3d(x_i, pred_i), axis=-1)
                elif arr0.ndim == 3:
                    # stochastic -> take samples at (i-1): (|V|,S,3)
                    pred_i = np.stack([a[:, i - 1, :] for a in pred_list], axis=0)  # (|V|,S,3)
                    # compute min over both node and sample
                    # x_i: (P,3), pred_i: (M,S,3)
                    d = x_i[:, None, None, :] - pred_i[None, :, :, :]
                    d = np.linalg.norm(d, axis=-1)  # (P,M,S)
                    min_dist_pred_i = np.min(d, axis=(1, 2))
                else:
                    raise ValueError("prediction array must be (H,3) or (S,H,3) per pid")

            else:
                min_dist_obs_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)
                min_dist_pred_i = np.full((n_paths,), DISTANCE_BOUND, dtype=np.float32)

            min_dist_obs.append(min_dist_obs_i)
            min_dist_pred.append(min_dist_pred_i)

            q_i = self.quantile_history[-i][:, i - 1]
            quantiles.append(q_i)

        if n_data > 1:
            min_dist_obs = np.stack(min_dist_obs, axis=-1)   # (P, <=N)
            min_dist_pred = np.stack(min_dist_pred, axis=-1) # (P, <=N)
            quantiles = np.stack(quantiles, axis=-1)         # (P, <=N)

            err = (quantiles < (min_dist_pred - min_dist_obs)).astype(np.float32)

            # ACI update
            self.alpha_t[:, :max_n_steps - 1] += self._gamma * (self._miscoverage_level - err)

            if n_data < self._n_steps + 1:
                pad_width = self._n_steps + 1 - n_data
                err = np.hstack((err, np.zeros((n_paths, pad_width), dtype=np.float32)))
        else:
            err = np.zeros((n_paths, self._n_steps), dtype=np.float32)

        self._track_queue.append(obs_last)
        return err

    # -----------------------------
    # main call (QuadWorldEnv3D compatible)
    # -----------------------------
    def __call__(
        self,
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        goal_xyz: np.ndarray,
        predictions: Dict[int, np.ndarray],   # dict(pid->(H,3)) or dict(pid->(S,H,3))
        boxes_3d: Optional[List[Any]] = None, # kept for signature compatibility (static constraints optional)
    ):
        """
        Returns:
          act: (3,) = (v_xy, yaw_rate, vz) or None if infeasible
          info: dict
        """
        boxes_3d = boxes_3d or []

        # generate discrete search space
        paths, vels = self.generate_paths_3d(robot_xyz, robot_yaw, n_skip=self.n_skip)
        self.path_history.append(paths)

        # compute per-path per-step quantiles
        quantiles = self.evaluate_scores_3d(paths=paths)
        self.quantile_history.append(quantiles)

        # filter unsafe paths
        safe_paths, safe_vels = self.filter_unsafe_paths_3d(
            paths=paths,
            vels=vels,
            boxes_3d=boxes_3d,
            predictions=predictions,
            quantiles=quantiles,
        )

        if safe_paths is None:
            return None, {"feasible": False, "quantiles": quantiles}

        best_path, best_vels, best_cost = self.score_paths_3d(safe_paths, safe_vels, goal_xyz)

        info = {
            "feasible": True,
            "candidate_paths": paths,
            "safe_paths": safe_paths,
            "final_path": best_path,
            "cost": best_cost,
            "quantiles": quantiles,
        }
        # act at first control
        return best_vels[0].copy(), info

    # -----------------------------
    # scoring
    # -----------------------------
    @staticmethod
    def score_paths_3d(paths: np.ndarray, vels: np.ndarray, goal_xyz: np.ndarray):
        goal = np.asarray(goal_xyz, dtype=np.float32).reshape(3,)
        intermediate_cost = np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        control_cost = 0.001 * np.sum(vels ** 2, axis=(-2, -1))
        terminal_cost = 10.0 * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        total = intermediate_cost + control_cost + terminal_cost
        idx = int(np.argmin(total))
        return paths[idx], vels[idx], float(total[idx])

    # -----------------------------
    # constraints
    # -----------------------------
    def filter_unsafe_paths_3d(
        self,
        *,
        paths: np.ndarray,                # (P, T+1, 3)
        vels: np.ndarray,                 # (P, T, 3)
        boxes_3d: List[Any],
        predictions: Dict[int, np.ndarray],
        quantiles: np.ndarray,            # (P, T)
    ):
        P, T1, _ = paths.shape
        T = T1 - 1

        # (A) static constraints (optional): if you have 3D boxes, implement here
        # baseline: none
        mask_unsafe_static = np.zeros((P,), dtype=bool)

        # (B) dynamic constraints
        if not predictions:
            mask_unsafe_dynamic = np.zeros((P,), dtype=bool)
        else:
            # pack predictions:
            # deterministic: (M,H,3)
            # stochastic:    (M,S,H,3)
            # We need "obs" shape as (#nodes, #steps, 3) or (#nodes, S, #steps, 3)
            any_arr = next(iter(predictions.values()))
            any_arr = np.asarray(any_arr)

            if any_arr.ndim == 2:
                pred_np = pred_dict_to_numpy(predictions)  # (M,H,3)
                # align horizon length
                H = pred_np.shape[1]
                T_use = min(T, H)
                pred_np_use = pred_np[:, :T_use, :]  # (M,T,3)
                min_dist = compute_min_dist_3d(paths=paths[:, 1:1+T_use, :], obs=pred_np_use)
            elif any_arr.ndim == 3:
                # dict(pid -> (S,H,3)) -> (M,S,H,3)
                pred_list = [np.asarray(v, dtype=np.float32) for v in predictions.values()]
                pred_np = np.stack(pred_list, axis=0)  # (M,S,H,3)
                H = pred_np.shape[2]
                T_use = min(T, H)
                pred_np_use = pred_np[:, :, :T_use, :]  # (M,S,T,3)
                min_dist = compute_min_dist_3d(paths=paths[:, 1:1+T_use, :], obs=pred_np_use)
            else:
                raise ValueError("predictions values must be (H,3) or (S,H,3)")

            # compare against safe_rad + quantile
            q_use = quantiles[:, :min_dist.shape[1]]
            mask_unsafe_dynamic = np.any(min_dist < (self.safe_rad + q_use), axis=-1)

        mask_safe = np.logical_and(~mask_unsafe_static, ~mask_unsafe_dynamic)
        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe]
        return None, None

    # -----------------------------
    # online score / quantiles
    # -----------------------------
    def load_recent_obs(self, batch_size: int) -> List[Dict[int, np.ndarray]]:
        return self._track_queue[-batch_size:]

    def load_recent_pred(self, batch_size: int, step: int) -> List[Dict[int, np.ndarray]]:
        """
        returns list of dicts aligned with obs_batch:
        pred_j[node] = predicted position for that node at 'step-1' (deterministic) OR samples at that step.
        """
        assert step >= 1
        idx_begin = -batch_size + 1 - step
        idx_end = len(self._prediction_queue) if step == 1 else 1 - step

        pred_batch = []
        for idx in range(idx_begin, idx_end):
            pred_t = self._prediction_queue[idx]
            # keep full (H,3) or (S,H,3); we will slice in evaluate
            pred_batch.append(pred_t)
        return pred_batch

    def evaluate_scores_3d(self, paths: np.ndarray) -> np.ndarray:
        """
        returns quantiles: (P, T)
        """
        n_data = len(self._track_queue)  # number of stored observations (time steps)
        P = paths.shape[0]
        T = self._n_steps

        if n_data >= self._n_steps + 1:
            batch_size = min(n_data - self._n_steps, self.calibration_set_size)

            obs_batch = self.load_recent_obs(batch_size=batch_size)

            min_dist_obs = np.zeros((P, T, batch_size), dtype=np.float32)
            min_dist_pred = np.zeros((P, T, batch_size), dtype=np.float32)

            for i in range(1, T + 1):
                x_i = paths[:, i, :]  # (P,3)
                pred_batch = self.load_recent_pred(batch_size=batch_size, step=i)

                for b_idx, (obs_j, pred_j) in enumerate(zip(obs_batch, pred_batch)):
                    obs_nodes = set(obs_j.keys())
                    pred_nodes = set(pred_j.keys())
                    common_nodes = list(obs_nodes & pred_nodes)

                    if common_nodes:
                        obs_np = np.stack([obs_j[n] for n in common_nodes], axis=0).astype(np.float32)

                        # obs distance
                        d_obs = compute_pairwise_distances_3d(x_i, obs_np)  # (P,|V|)
                        min_obs = np.min(d_obs, axis=-1)

                        # pred distance at step i
                        arr0 = np.asarray(pred_j[common_nodes[0]])
                        if arr0.ndim == 2:
                            pred_i = np.stack([np.asarray(pred_j[n], dtype=np.float32)[i - 1, :] for n in common_nodes], axis=0)
                            d_pred = compute_pairwise_distances_3d(x_i, pred_i)
                            min_pred = np.min(d_pred, axis=-1)
                        elif arr0.ndim == 3:
                            # (S,H,3)
                            pred_i = np.stack([np.asarray(pred_j[n], dtype=np.float32)[:, i - 1, :] for n in common_nodes], axis=0)  # (|V|,S,3)
                            d = x_i[:, None, None, :] - pred_i[None, :, :, :]
                            d = np.linalg.norm(d, axis=-1)  # (P,|V|,S)
                            min_pred = np.min(d, axis=(1, 2))
                        else:
                            raise ValueError("prediction must be (H,3) or (S,H,3) per node")
                    else:
                        min_obs = np.full((P,), DISTANCE_BOUND, dtype=np.float32)
                        min_pred = np.full((P,), DISTANCE_BOUND, dtype=np.float32)

                    min_dist_obs[:, i - 1, b_idx] = min_obs
                    min_dist_pred[:, i - 1, b_idx] = min_pred

            scores = np.clip(min_dist_pred - min_dist_obs, a_min=0.0, a_max=None)  # (P,T,B)

            # quantile levels = 1 - alpha_t
            levels = 1.0 - np.clip(self.alpha_t, 0.0, 1.0)
            quantiles = quantile_higher(scores, levels)  # (P,T)

            # cap inf
            max_scores = 0.5 * self._dt * np.arange(1, T + 1, dtype=np.float32)
            max_scores = np.tile(max_scores[None, :], (P, 1))
            quantiles = np.where(np.isposinf(quantiles), max_scores, quantiles)
            return quantiles.astype(np.float32)

        # warmup fallback
        max_scores = 1.5 * self._dt * np.arange(1, T + 1, dtype=np.float32)
        max_scores = np.tile(max_scores[None, :], (P, 1))
        return max_scores.astype(np.float32)

    # -----------------------------
    # path generation (3D)
    # -----------------------------
    def generate_paths_3d(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        *,
        n_skip: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns:
          paths: (P, T+1, 3)
          vels:  (P, T, 3)  where last dim is (v_xy, yaw_rate, vz)
        """
        dt = self._dt
        T = self._n_steps
        n_skip = int(n_skip)
        n_decision_epochs = T // n_skip

        # grid of controls per epoch
        v_grid, w_grid, vz_grid = np.meshgrid(self._v_grid, self._w_grid, self._vz_grid, indexing="xy")
        v_grid = v_grid.reshape(-1).astype(np.float32)
        w_grid = w_grid.reshape(-1).astype(np.float32)
        vz_grid = vz_grid.reshape(-1).astype(np.float32)
        n_points = v_grid.size  # 27 if 3x3x3

        # shapes like original: (n_points,)*epochs + (T+1,)
        state_shape = tuple(n_points for _ in range(n_decision_epochs)) + (T + 1,)
        x = np.zeros(state_shape, dtype=np.float32)
        y = np.zeros(state_shape, dtype=np.float32)
        z = np.zeros(state_shape, dtype=np.float32)
        yaw = np.zeros(state_shape, dtype=np.float32)

        x[..., 0] = float(robot_xyz[0])
        y[..., 0] = float(robot_xyz[1])
        z[..., 0] = float(robot_xyz[2])
        yaw[..., 0] = float(robot_yaw)

        control_shape = tuple(n_points for _ in range(n_decision_epochs)) + (T,)
        v_xy = np.zeros(control_shape, dtype=np.float32)
        w = np.zeros(control_shape, dtype=np.float32)
        vz = np.zeros(control_shape, dtype=np.float32)

        for e in range(n_decision_epochs):
            aug = [1] * n_decision_epochs
            aug[e] = -1
            v_e = v_grid.reshape(aug)
            w_e = w_grid.reshape(aug)
            vz_e = vz_grid.reshape(aug)

            for t in range(e * n_skip, (e + 1) * n_skip):
                v_xy[..., t] = v_e
                w[..., t] = w_e
                vz[..., t] = vz_e

                yaw[..., t + 1] = yaw[..., t] + dt * w_e
                x[..., t + 1] = x[..., t] + dt * v_e * np.cos(yaw[..., t + 1])
                y[..., t + 1] = y[..., t] + dt * v_e * np.sin(yaw[..., t + 1])
                z[..., t + 1] = z[..., t] + dt * vz_e

        # flatten
        x = x.reshape(-1, T + 1)
        y = y.reshape(-1, T + 1)
        z = z.reshape(-1, T + 1)

        v_xy = v_xy.reshape(-1, T)
        w = w.reshape(-1, T)
        vz = vz.reshape(-1, T)

        paths = np.stack([x, y, z], axis=-1).astype(np.float32)      # (P,T+1,3)
        vels = np.stack([v_xy, w, vz], axis=-1).astype(np.float32)   # (P,T,3)
        return paths, vels