import numpy as np
from typing import Dict, List


class AdaptiveConformalPredictionModule:
    """
    Implementation of Adaptive Conformal Prediction (ACP) for pedestrian trajectory forecasting.
    See Gibbs & Candes, 2021, or Zaffran et al., 2022 for details.
    Considering the lagged nature of prediction tasks, a lagged variant of ACP is implemented here; See Dixit et al., 2023 for details.
    """

    def __init__(self,
                 target_miscoverage_level,
                 step_size,
                 n_scores,
                 max_interval_lengths,
                 sample_size,
                 offline_calibration_set: Dict[int, List[float]]
                 ):

        self._n_scores = n_scores
        self._alpha = target_miscoverage_level

        # effective miscoverage level alpha_t
        # initialization: alpha_0 := alpha
        self._alpha_t = target_miscoverage_level * np.ones(self._n_scores)

        self._history = {
            'score': [],
            'interval': [],
            'coverage': [],
            'effective': []
        }

        # keeps the past K predictions, where K: prediction length
        # Each element of a queue is a dictionary representing a map track_id -> array of shape (K, 2)
        self._prediction_queue = []
        self._interval_queue = []

        # step size parameter
        # gamma = 0 corresponds to the standard split conformal prediction
        self._gamma = step_size
        self._max_interval_len = max_interval_lengths  # known upper bounds of the score functions

        # initialize each interval length to its maximum possible value
        # calibration set D^i_t containing the scores of i-th step prediction
        self._online_calibration_set = {
            i: [] for i in range(n_scores)
        }
        self._sample_size = sample_size
        # a set of pre-computed scores from offline data
        # corresponds to D^i_0
        self._offline_calibration_set = {i: [] for i in range(n_scores)}
        self._load_offline_calibration_set(offline_calibration_set)

        self._step = 0

    def _load_offline_calibration_set(self, offline_calibration_set: Dict[int, List[float]]):
        for i in range(self._n_scores):
            n_offline = len(offline_calibration_set[i])
            if n_offline < self._sample_size:
                # insufficient number of offline data (even smaller than calibration sample size)
                # fill the set with the known upper bound of the score
                self._offline_calibration_set[i] += [self._max_interval_len[i] for _ in
                                                     range(self._sample_size - n_offline)]

        return

    def update(self, obs, pred):
        """
        observe x_t -> generate prediction(s) -> adaptive update of D^i_{cal, t} & alpha^i_t
                    -> build C_{alpha^i_t}(x_t|D^i_{cal,t})
        """
        scores = self._compute_scores(obs)
        self._update_calibration_sets(scores)
        intervals = self._compute_interval_lengths()
        coverages = self._evaluate_coverage(scores)
        effective_levels = self._update_miscoverage_levels(coverages)

        self._prediction_queue.append(pred)  # PREDICTION(t)
        self._interval_queue.append(intervals)  # INTERVAL(t)
        self._step += 1  # t -> t + 1

        # store intermediate results for future evaluation
        self._history['score'].append(scores)
        self._history['interval'].append(intervals)
        self._history['coverage'].append(coverages)
        self._history['effective'].append(effective_levels)

        return intervals, coverages

    def _compute_scores(self, obs):
        """
        Compare the past predictions x(t|t-1), ..., x(t|t-K) with a new observation x(t) and compute the scores.

        return: list of length min(t, K) whose entry represents s^i_t
                If t = 0, then this returns the empty list [].
        """
        scores = []
        for i, pred in enumerate(reversed(self._prediction_queue)):
            # trace the diagonal of the prediction queue
            if i >= self._n_scores:
                # Recall: # score functions = prediction length
                break
            else:
                errors = []
                if not pred:
                    # When there is no pedestrian in the scene, then report zero error
                    scores.append(0.)
                else:
                    for track_id, xy_seq in pred.items():
                        if track_id in obs.keys():
                            # only if a tracked object appears both in the past prediction result & real observation
                            err = np.sum((xy_seq[i] - obs[track_id][-1]) ** 2) ** .5
                            errors.append(err)
                    if not errors:
                        scores.append(0.)
                    else:
                        scores.append(np.max(errors))
        return scores

    def _update_calibration_sets(self, scores):
        # D^i_{t+1] := D^i_t + [s^i_t]
        for i, score in enumerate(scores):
            self._online_calibration_set[i].append(score)
        return

    def _compute_interval_lengths(self):
        # R^i_t := Q_{1 - alpha_t} (D), where D ~ D^i_{t+1]
        intervals = []
        for i in range(self._n_scores):
            n_online = len(self._online_calibration_set[i])
            if n_online < self._sample_size:
                # insufficient online data
                # use all online data & sample a subset of offline data
                n_offline_samples = self._sample_size - n_online
                samples = self._online_calibration_set[i] + list(
                    np.random.choice(self._offline_calibration_set[i], size=n_offline_samples))
            else:
                samples = np.random.choice(self._online_calibration_set[i], size=self._sample_size, replace=True)
            alpha = self._alpha_t[i]
            if alpha <= 0.:
                interval = self._max_interval_len[i]
            elif alpha < 1.:
                interval = np.quantile(samples, q=1 - self._alpha_t[i])
            else:
                # alpha >= 1.
                interval = 0.
            intervals.append(interval)
        return np.array(intervals)

    def _evaluate_coverage(self, scores):
        coverages = np.array(scores) <= np.diag(np.flip(self._interval_queue, axis=0))
        assert coverages.size == min(self._n_scores, self._step)  # min(t, K)
        # check s^i_t <= R^i_{t-i}, which is equivalent to x(t) in C^i_{t-i}
        return coverages

    def _update_miscoverage_levels(self, coverages):
        if coverages.size > 0:
            # alpha^i_t = alpha^{i+1}_t + gamma * (alpha - 1 + I(x(t) in C^i_{t-i}))
            self._alpha_t[:coverages.size] += self._gamma * (self._alpha - 1. + coverages)
        return np.copy(self._alpha_t)

    def get_summary(self):
        return self._history.copy()


class AdaptiveCPMPC:
    def __init__(self,
                 n_steps=20,
                 dt=0.1,
                 min_linear_x=-0.8, max_linear_x=0.8,
                 min_angular_z=-0.7, max_angular_z=0.7,
                 n_skip=4,
                 robot_rad=0.4,
                 obstacle_rad=1./np.sqrt(2.),
                 target_miscoverage_level=0.2,
                 step_size=0.05
                 ):
        self._n_steps = n_steps
        self._dt = dt

        self.max_linear_x = max_linear_x
        self.min_linear_x = min_linear_x

        self.max_angular_z = max_angular_z
        self.min_angular_z = min_angular_z

        self.n_skip = n_skip

        self.robot_rad = robot_rad
        self.obstacle_rad = obstacle_rad
        self.safe_rad = self.robot_rad + self.obstacle_rad

        max_interval_lengths = 1.5 * dt * np.arange(1, n_steps + 1)
        offline_calibration_set = {i: [] for i in range(n_steps)}
        self.cp_module = AdaptiveConformalPredictionModule(
            target_miscoverage_level=target_miscoverage_level,
            step_size=step_size,
            n_scores=n_steps,
            max_interval_lengths=max_interval_lengths,
            sample_size=12,
            offline_calibration_set=offline_calibration_set
        )

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, confidence_intervals, goal):
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)
        # paths, vels = self.generate_paths_wheel_vel(pos_x, pos_y, orientation_z, linear_x, angular_z)
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes, predictions, confidence_intervals)
        if safe_paths is None:
            # print('MPC infeasible')
            return None, {'feasible': False}
        else:
            path, vel, cost = self.score_paths(safe_paths, vels, goal)
            info = {
                'cost': cost,
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path
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

    def filter_unsafe_paths(self, paths, vels, boxes, predictions, confidence_intervals):
        """
        Given a set of  xy-paths and a collection of rectangles, determine if the path intersects with one of the rectangles.
        :param paths: numpy array of shape (# paths, # steps, 2)
        :param boxes: list of rectangles, where each rectangle is defined as (center, size, angle)

        :return: safe paths of shape (# paths, # steps, 2), or None if all paths are unsafe
        """
        ROBOT_RAD = 0.4

        n_paths = paths.shape[0]

        masks = []
        for box in boxes:
            center = box.pos
            sz = np.array([box.w, box.h])
            th = box.rad
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s], [s, c]])  # rotate by -th w.r.t. the origin
            lb, ub = -.5 * sz - ROBOT_RAD, .5 * sz + ROBOT_RAD
            # robot's current coordinate frame -> rectangle's coordinate frame
            transformed_paths = (paths[:, 1:, :] - center) @ R  # first state: observed from the system
            # boolean array of shape (# paths, # steps)
            # True = collision
            mask = np.logical_and(np.all(transformed_paths <= ub, axis=-1), np.all(transformed_paths >= lb, axis=-1))
            masks.append(mask)
        masks = np.array(masks)

        mask_union_per_point = np.sum(masks, axis=0, dtype=bool)

        mask_union_per_path = np.sum(mask_union_per_point, axis=-1)

        mask_p_per_path = np.zeros((n_paths,), dtype=bool)
        for obj_id, prediction in predictions.items():
            obj_mask = np.any(
                np.sum((paths[:, 1:, :] - prediction) ** 2, axis=-1) < (self.safe_rad + confidence_intervals) ** 2,
                axis=-1)

            mask_p_per_path += obj_mask.squeeze()

        # True = no collision
        mask_final = np.logical_and(np.logical_not(mask_union_per_path), np.logical_not(mask_p_per_path))
        if np.any(mask_final):
            return paths[mask_final], vels[mask_final]
        else:
            # print('no safe paths found')
            return None, None

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
