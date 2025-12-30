from typing import Dict
import numpy as np
from utils import compute_pairwise_distances_along_axis, compute_pairwise_distances, compute_quantiles

DISTANCE_BOUND = 10000


def obs2numpy(o: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array(list(v[-1] for v in o.values()))  # most recent observation only


def process_obs(o: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {node: v[-1] for node, v in o.items()}


def to_numpy(o: Dict[str, np.ndarray], nodes) -> np.ndarray:
    """
    Take a subset of nodes and pack their data into a single numpy array; outputs an array of shape (subset size, data shape)
    For instance, when data shape = (1, N, 2), the overall shape is (subset size, 1, N, 2).
    """
    return np.array([o[node] for node in nodes])


def pred2numpy(o: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array(list(o.values()))


def compute_min_dist(paths, obs):
    """
    paths: shape (search space size, # steps, dim.)
    obs: shape (# nodes, sample size, # steps, dim.); If sample size = 1, then the shape is given as (# nodes, # steps, dim.)
    return: shape (# search space size, # steps)
    """

    D = compute_pairwise_distances_along_axis(paths, obs, axis=(1, 1))  # (# steps, # search space size, # nodes)
    return np.min(D, axis=-1).T


class EgocentricCPMPC:
    """
    An Implementation of ECP-MPC.
    See Shin, J., 2025. for details.
    """

    def __init__(
            self,
            n_steps=12,
            dt=0.4,
            min_linear_x=-0.8, max_linear_x=0.8,
            min_angular_z=-0.7, max_angular_z=0.7,
            n_skip=4,
            robot_rad=0.4,
            obstacle_rad=1./np.sqrt(2.),
            calibration_set_size=10,
            miscoverage_level=0.1,
            step_size=0.05
    ):
        """
        Parameters
        ----------
        n_steps: int
            planning horizon of the MPC

        dt: float
            sampling time of the MPC

        min_linear_x, max_linear_x: float
            lower/upper bound of the vehicle's linear velocity

        min_angular_z, max_angular_z: float
            lower/upper bound of the vehicle's angular velocity

        n_skip: int
            length of each subinterval: the control inputs are kept constant on each subinterval
            For instance, if n_steps = 12 and n_skip = 4, then MPC searches over velocity sequences of the form (v_0, v_1, v2)
            where v_0 is for the first subinterval, v_1 for the second, and v_2 for the final.

        robot_rad: float
            radius of the circle enclosing the robot

        calibration_set_size: int
            number of samples to use for online calibration

        miscoverage_level: float
            target miscoverage level to use; must be chosen from (0, 1)

        step_size: float
            step size parameter for ACI update
        """

        self._n_steps = n_steps
        self._dt = dt
        self._miscoverage_level = miscoverage_level

        self.max_linear_x = max_linear_x
        self.min_linear_x = min_linear_x

        self.max_angular_z = max_angular_z
        self.min_angular_z = min_angular_z

        n_decision_epochs = n_steps // n_skip

        n_points = 9

        n_paths = n_points ** n_decision_epochs

        self.alpha_t = miscoverage_level * np.ones((n_paths, n_steps))

        self.n_skip = n_skip

        self.robot_rad = robot_rad
        self.obstacle_rad = obstacle_rad
        self.safe_rad = self.robot_rad + self.obstacle_rad

        self.path_history = []
        self.quantile_history = []

        self.calibration_set_size = calibration_set_size

        self._gamma = step_size

        self._prediction_queue = []  # prediction results
        self._track_queue = []  # true configuration of dynamic obstacles

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):
        # Warning! The method can be invoked only when t >= N
        # Thus, the controller has to wait until at least N observations are collected.

        # update the observation queue & alpha^J_t's
        # The following line has been moved to the outer loop
        # self.update_observations(obs=tracking_res)

        # span a discrete search space (x^J_{...|t}: J in scr{J})
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)

        self.path_history.append(paths)

        quantiles = self.evaluate_scores(paths=paths)  # compute R^J_{t+i|t} for all J & i
        self.quantile_history.append(quantiles)

        # Solve MPC:
        # Find x^J_{...|t}'s within the constraints
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions, quantiles)

        # update the prediction queue
        # moved to the outer loop
        # self.update_predictions(predictions)

        if safe_paths is None:
            return None, {
                'feasible': False,
                'quantiles': quantiles}
        else:
            path, vel, cost = self.score_paths(safe_paths, vels, goal)

            info = {
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path,
                'cost': cost
            }

            return vel[1], info

    @staticmethod
    def score_paths(paths, vels, goal):
        intermediate_cost = np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        control_cost = .001 * np.sum(vels ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)
        total_cost = intermediate_cost + control_cost + terminal_cost
        minimum_cost = np.argmin(total_cost)
        return paths[minimum_cost], vels[minimum_cost], total_cost[minimum_cost]

    def filter_unsafe_paths(
            self,
            paths,
            vels,
            boxes,
            predictions,
            quantiles
    ):
        # static constraints
        masks = [np.zeros((paths.shape[0], paths.shape[1] - 1))]
        for box in boxes:
            center = box.pos
            sz = np.array([box.w, box.h])
            th = box.rad
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])  # rotate by -th w.r.t. the origin
            lb, ub = -.5 * sz - self.robot_rad, .5 * sz + self.robot_rad
            # robot's current coordinate frame -> rectangle's coordinate frame
            transformed_paths = (paths[:, 1:, :] - center) @ R  # first state: observed from the system
            # boolean array of shape (# paths, # steps)
            # True = collision
            mask = np.logical_and(np.all(transformed_paths <= ub, axis=-1), np.all(transformed_paths >= lb, axis=-1))
            masks.append(mask)
        masks = np.array(masks)

        mask_unsafe_static = np.sum(masks, axis=0, dtype=bool)
        mask_unsafe_static = np.sum(mask_unsafe_static, axis=-1)
        if not predictions:
            mask_unsafe_dynamic = np.zeros_like(mask_unsafe_static, dtype=bool)
        # shape = (search space size, prediction length)
        else:
            min_dist = compute_min_dist(paths=paths[:, 1:, :], obs=pred2numpy(predictions))
            mask_unsafe_dynamic = np.any(min_dist < self.safe_rad + quantiles, axis=-1)

        # True = no collision
        mask_safe = np.logical_and(np.logical_not(mask_unsafe_static), np.logical_not(mask_unsafe_dynamic))

        if np.any(mask_safe):
            return paths[mask_safe], vels[mask_safe]
        else:
            return None, None

    def update_predictions(self, prediction_result):
        self._prediction_queue.append(prediction_result)

    def load_recent_obs(self, batch_size):
        # list of length <= size whose elements are arrays of shape (|V|, 2) where |V|: # nodes
        # TODO: handle the case # observations < batch size
        # t' in [t-M+1, t]
        obs_batch = self._track_queue[-batch_size:]
        return obs_batch

    def load_recent_pred(self, batch_size, step):
        """
        If this method is called at time t, we assume that the queue only stores the prediction results up to t - 1:
        X_{...|t-1}, X{...|t-2}...
        """
        assert step >= 1
        # t' in [t-M+1-i, t-i]
        idx_begin = -batch_size + 1 - step
        idx_end = len(self._prediction_queue) if step == 1 else 1 - step

        # each element having of shape (|V(t')|, prediction length, 2) when deterministic;
        # shape (|V(t')|, sample size, prediction length, 2) when stochastic;

        pred_batch = []
        for idx in range(idx_begin, idx_end):
            pred_t = self._prediction_queue[idx]
            pred_i = {node: v[step - 1, :] for node, v in pred_t.items()}
            pred_batch.append(pred_i)
        return pred_batch

    def evaluate_scores(self, paths):
        n_data = len(self._track_queue)  # X(0), ...., X(t) stored in the queue; thus represents t + 1
        if n_data >= self._n_steps + 1:
            n_paths = paths.shape[0]
            batch_size = min(n_data - self._n_steps, self.calibration_set_size)

            obs_batch = self.load_recent_obs(batch_size=batch_size)
            min_dist_obs = np.zeros((n_paths, self._n_steps, batch_size))
            min_dist_pred = np.zeros((n_paths, self._n_steps, batch_size))

            for i in range(1, self._n_steps + 1):  # 1 <= i <= N
                x_i = paths[:, i, :]  # (search space size, 2)
                # (search space size, batch size, max # nodes, sample size)
                # TODO: check the order of insertions & loading
                pred_batch = self.load_recent_pred(batch_size=batch_size,
                                                   step=i)  # (batch size, max # nodes, sample size, 2)

                for batch_idx, (obs_j, pred_j) in enumerate(zip(obs_batch, pred_batch)):
                    obs_nodes = obs_j.keys()  # V(t)
                    pred_nodes = pred_j.keys()  # V(t - i)

                    common_nodes = list(set(obs_nodes) & set(pred_nodes))  # V = V(t) \cap V(t-i)

                    pred_numpy = np.array([pred_j[node] for node in common_nodes])
                    obs_numpy = np.array([obs_j[node] for node in common_nodes])

                    if common_nodes:
                        pdist_pred = compute_pairwise_distances(x_i, pred_numpy)  # (search space size, |V|)
                        min_dist_pred_i = np.min(pdist_pred, axis=-1)

                        pdist_obs = compute_pairwise_distances(x_i, obs_numpy)
                        min_dist_obs_i = np.min(pdist_obs, axis=-1)

                    else:
                        min_dist_pred_i = np.full((n_paths,), DISTANCE_BOUND)
                        min_dist_obs_i = np.full((n_paths,), DISTANCE_BOUND)

                    min_dist_obs[:, i - 1, batch_idx] = min_dist_obs_i
                    min_dist_pred[:, i - 1, batch_idx] = min_dist_pred_i

            scores = np.clip(min_dist_pred - min_dist_obs, a_min=0., a_max=None)
            # final shape: (search space size, prediction length)
            quantiles = compute_quantiles(scores, axis=-1, levels=1. - self.alpha_t)  # quantile along batch dim
            # maximum for Q_{1 - alpha_t} when alpha_t <= 0
            # The value is estimated from the reported FDEs of Trajectron++ on UCY-ETH datasets
            max_scores = .5 * self._dt * np.arange(1, self._n_steps + 1)
            max_scores = np.tile(max_scores, (n_paths, 1))
            quantiles = np.where(np.isposinf(quantiles), max_scores, quantiles)
            return quantiles
        else:
            n_paths = paths.shape[0]
            max_scores = 1.5 * self._dt * np.arange(1, self._n_steps + 1)
            max_scores = np.tile(max_scores, (n_paths, 1))
            return max_scores

    def update_observations(self, obs):
        """
        obs: dictionary containing the trajectories of the dynamic obstacles
        
        """
        n_paths = self.alpha_t.shape[0]
        if not obs:
            # no dynamic agents in the scene
            self._track_queue.append(obs)
            err = np.zeros((n_paths, self._n_steps))
            return err
        else:
            obs = process_obs(obs)

            quantiles = []
            min_dist_obs = []
            min_dist_pred = []

            n_data = len(self.quantile_history)  # R_{t_0}, ..., R_{t-1} stored; thus represents t - t_0 where t_0 >= N
            max_n_steps = min(self._n_steps + 1, n_data)
            for i in range(1, max_n_steps):
                # TODO: intersection of nodes

                pred = self._prediction_queue[-i]  # X_{...|t-i}; of shape (|V(t-i)|, sample size, prediction length, 2)

                obs_nodes = obs.keys()  # V(t)
                pred_nodes = pred.keys()  # V(t - i)

                common_nodes = list(set(obs_nodes) & set(pred_nodes))  # V = V(t) \cap V(t-i)

                if common_nodes:
                    obs_numpy = to_numpy(obs, common_nodes)
                    paths = self.path_history[-i]  # x^J_{...|t-i}
                    x_i = paths[:, i, :]  # x^J_{t|t-i}; of shape (search space size, 2)
                    min_dist_obs_i = np.min(compute_pairwise_distances(x_i, obs_numpy), axis=-1)  # (search space size,)
                    pred_numpy = to_numpy(pred,
                                          common_nodes)  # X_{...|t-i}; of shape (|V|, sample size, prediction length, 2)
                    pred_numpy_i = pred_numpy[..., i - 1, :]  # (|V|, sample size, 2)
                    min_dist_pred_i = np.min(compute_pairwise_distances(x_i, pred_numpy_i),
                                             axis=-1)  # (search space size,)


                else:
                    min_dist_obs_i = np.full((n_paths,), DISTANCE_BOUND)
                    min_dist_pred_i = np.full((n_paths,), DISTANCE_BOUND)

                min_dist_obs.append(min_dist_obs_i)
                min_dist_pred.append(min_dist_pred_i)

                q_i = self.quantile_history[-i][:, i - 1]
                quantiles.append(q_i)
                # x^J_{t-1}, x^J_{t-2}, ...

            if n_data > 1:
                min_dist_obs = np.stack(min_dist_obs, axis=-1)  # (search space size, prediction length)
                min_dist_pred = np.stack(min_dist_pred, axis=-1)

                quantiles = np.stack(quantiles, axis=-1)  #

                err = (quantiles < min_dist_pred - min_dist_obs)

                self.alpha_t[:, :max_n_steps - 1] += self._gamma * (self._miscoverage_level - err)
                if n_data < self._n_steps + 1:
                    pad_width = self._n_steps + 1 - n_data
                    err = np.hstack((err, np.zeros((n_paths, pad_width))))  # just for the consistent data size
            else:
                err = np.zeros((n_paths, self._n_steps))

            # online update of alpha^{J, i}_t's

            # self.alpha_t = np.clip(self.alpha_t, 0., 1.)
            # update data
            self._track_queue.append(obs)
            return err

    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            n_skip=5
    ):
        """
        Generate multiple paths starting at (x, y, theta) = (0, 0, 0)
        """

        # TODO: Employing pruning techniques would reduce the number of the paths, but would be also challenging to optimize...
        # TODO: use numba?
        # physical parameters
        dt = self._dt
        # velocity & acceleration ranges

        linear_xs = np.array([self.min_linear_x, .0, self.max_linear_x])
        angular_zs = np.array([self.min_angular_z, .0, self.max_angular_z])

        n_points = linear_xs.size * angular_zs.size

        linear_xs, angular_zs = np.meshgrid(linear_xs, angular_zs)

        linear_xs = np.reshape(linear_xs, newshape=(-1,))
        angular_zs = np.reshape(angular_zs, newshape=(-1,))

        # (# grid points, 2)
        # velocity_profile = np.stack((linear_xs, angular_zs), axis=0)

        n_decision_epochs = self._n_steps // n_skip

        # profiles = [velocity_profile for _ in range(n_decision_epochs)]

        # n_paths = n_points ** n_decision_epochs

        state_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps + 1,)
        x = np.zeros(state_shape)
        y = np.zeros(state_shape)
        th = np.zeros(state_shape)

        # state initialization
        x[..., 0] = pos_x
        y[..., 0] = pos_y
        th[..., 0] = orientation_z

        control_shape = tuple(n_points for _ in range(n_decision_epochs)) + (self._n_steps,)
        v = np.zeros(control_shape)
        w = np.zeros(control_shape)

        for e in range(n_decision_epochs):
            augmented_shape = [1] * n_decision_epochs
            augmented_shape[e] = -1
            v_epoch = linear_xs.reshape(augmented_shape)
            w_epoch = angular_zs.reshape(augmented_shape)
            for t in range(e * n_skip, (e + 1) * n_skip):
                v[..., t] = v_epoch
                w[..., t] = w_epoch

                x[..., t + 1] = x[..., t] + dt * v_epoch * np.cos(th[..., t])
                y[..., t + 1] = y[..., t] + dt * v_epoch * np.sin(th[..., t])
                th[..., t + 1] = th[..., t] + dt * w_epoch

        x = np.reshape(x, (-1, self._n_steps + 1))
        y = np.reshape(y, (-1, self._n_steps + 1))
        # th = np.reshape(th, (-1, self._n_steps))
        v = np.reshape(v, (-1, self._n_steps))
        w = np.reshape(w, (-1, self._n_steps))

        return np.stack((x, y), axis=-1), np.stack((v, w), axis=-1)
