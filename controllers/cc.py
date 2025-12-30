import numpy as np


class ConformalController:
    def __init__(
            self,
            n_steps=20,
            dt=0.1,
            min_linear_x=-0.8, max_linear_x=0.8,
            min_angular_z=-0.7, max_angular_z=0.7,
            n_skip=4,
            conformal_control_variable=1.,
            risk_level=-1.,
            step_size=5000.,
            robot_rad=0.4,
            obstacle_rad=1./np.sqrt(2.)
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
        self.safe_rad = self.robot_rad + obstacle_rad

        self._eta = step_size

        assert conformal_control_variable > 0.  # lambda > 0
        self._lambda = conformal_control_variable
        # assert .0 <= risk_level and risk_level <= 1.        # epsilon in [0, 1]
        self._epsilon = risk_level

    def __call__(self, pos_x, pos_y, orientation_z, boxes, predictions, goal):
        paths, vels = self.generate_paths(pos_x, pos_y, orientation_z, n_skip=self.n_skip)
        # paths, vels = self.generate_paths_wheel_vel(pos_x, pos_y, orientation_z, linear_x, angular_z)
        safe_paths, vels = self.filter_unsafe_paths(paths, vels, boxes)
        if safe_paths is None:
            # print('MPC infeasible')
            return None, {'feasible': False}
        else:
            path, vel, cost = self.score_paths(safe_paths, vels, predictions, goal)
            info = {
                'cost': cost,
                'feasible': True,
                'candidate_paths': paths,
                'safe_paths': safe_paths,
                'final_path': path
            }
            return vel[1], info

    def score_paths(self, paths, vels, predictions, goal):
        intermediate_cost = np.sum((paths[:, :-1, :] - goal) ** 2, axis=(-2, -1))
        control_cost = .001 * np.sum(vels ** 2, axis=(-2, -1))
        terminal_cost = 10. * np.sum((paths[:, -1, :] - goal) ** 2, axis=-1)

        distances = []
        for track_id, prediction in predictions.items():
            # shape = (# paths, # steps)
            distance = np.sum((paths[:, 1:, :] - prediction) ** 2, axis=-1) ** .5
            distances.append(distance)  # (# tracked, # paths, # steps)
        min_distances = np.min(distances, axis=0)
        avoidance_cost = -self._lambda * np.sum(min_distances, axis=-1)
        total_cost = intermediate_cost + control_cost + terminal_cost
        minimum_cost = np.argmin(total_cost + avoidance_cost)
        return paths[minimum_cost], vels[minimum_cost], total_cost[minimum_cost]

    def update_conformal_var(self, pos_x, pos_y, tracking_res):
        positions = np.array([xy[-1] for xy in tracking_res.values()])
        loss = -np.min(np.sum((np.array([pos_x, pos_y]) - positions) ** 2, axis=-1) ** .5)
        self._lambda -= self._eta * (self._epsilon - loss)
        self._lambda = max(0.01, self._lambda)
        return {'loss': loss, 'conformal_var': self._lambda}

    def filter_unsafe_paths(self, paths, vels, boxes):
        """
        Given a set of  xy-paths and a collection of rectangles, determine if the path intersects with one of the rectangles.
        :param paths: numpy array of shape (# paths, # steps, 2)
        :param boxes: list of rectangles, where each rectangle is defined as (center, size, angle)

        :return: safe paths of shape (# paths, # steps, 2), or None if all paths are unsafe
        """


        n_paths = paths.shape[0]

        masks = []
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

        mask_union_per_point = np.sum(masks, axis=0, dtype=bool)

        mask_union_per_path = np.sum(mask_union_per_point, axis=-1)

        mask_p_per_path = np.zeros((n_paths,), dtype=bool)

        # True = no collision
        mask_final = np.logical_and(np.logical_not(mask_union_per_path), np.logical_not(mask_p_per_path))
        if np.any(mask_final):
            return paths[mask_final], vels[mask_final]
        else:
            return None, None

    def generate_paths(
            self,
            pos_x,
            pos_y,
            orientation_z,
            n_skip=4
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
