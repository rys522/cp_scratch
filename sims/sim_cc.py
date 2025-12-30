import os
import numpy as np
import matplotlib.pyplot as plt

from visualization_utils import render
from func_cp.controllers.cc import ConformalController


def run_cc(
        dataset,
        scenarios,
        init_robot_pose,
        goal_pos,
        max_linear_x,
        min_linear_x,
        max_angular_z,
        min_angular_z,
        predictions,
        dt,  # sampling time
        init_frame,
        visualize,
        asset_dir,
        robot_img,
        max_n_steps,
        robot_rad,
        obstacle_rad,
        risk_level,
        step_size
):

    prediction_dict, histories_dict, futures_dict = predictions['prediction'], predictions['history'], predictions['future']

    # visualization (controller-specific)
    if visualize:
        assert asset_dir is not None
        print('dataset frames loaded from', asset_dir)

    stat_dir = os.path.join(os.path.dirname(__file__), 'stats', dataset, 'cc')

    os.makedirs(stat_dir, exist_ok=True)

    metric_dict = dict()

    trajectories = []
    for scene_idx, scenario_begin in enumerate(scenarios):
        xys = []
        eval_metrics = {
            'collisions': [],
            'costs': [],
            'exit_time': np.inf,
            'infeasible': []
        }

        prediction_len = 12
        controller = ConformalController(
            n_steps=prediction_len,
            dt=dt,
            min_linear_x=min_linear_x, max_linear_x=max_linear_x,
            min_angular_z=min_angular_z, max_angular_z=max_angular_z,
            n_skip=4,
            conformal_control_variable=1.,
            risk_level=risk_level,
            step_size=step_size,
            robot_rad=robot_rad,
            obstacle_rad=obstacle_rad
        )

        if len(prediction_dict.keys()) == 0:
            return

        position_x, position_y, orientation_z = init_robot_pose

        if visualize:
            video_dir = os.path.join(os.path.dirname(__file__), 'videos', dataset, str(scene_idx), 'cc')
            print('path to rendered scenes:', video_dir)
            os.makedirs(video_dir, exist_ok=True)

            print('results visualized at {}'.format(video_dir))

        count = 0
        done = False
        ts_key = scenario_begin
        while count < max_n_steps:
            if ts_key in prediction_dict:

                p_dict = prediction_dict[ts_key]
                h_dict = histories_dict[ts_key]
                f_dict = futures_dict[ts_key]

                if count < 15:

                    velocity = np.array([0., 0.])
                    info = {'feasible': True,
                            'candidate_paths': np.array([]),
                            'safe_paths': np.array([]),
                            'final_path': np.tile(np.array([position_x, position_y]), (12, 1))}

                else:
                    obs_pos = np.array([o[-1] for o in h_dict.values()])  # (|V|, 2)
                    robot_pos = np.array([position_x, position_y])
                    min_obs_dist = np.min(np.sum((obs_pos - robot_pos) ** 2, -1) ** .5)

                    min_goal_dist = np.sum((robot_pos - goal_pos) ** 2, -1) ** .5
                    if min_goal_dist <= 0.6:
                        eval_metrics['exit_time'] = count
                        done = True

                    collision = True if min_obs_dist < robot_rad + obstacle_rad else False
                    if not done:
                        eval_metrics['collisions'].append(collision)

                    velocity, info = controller(pos_x=position_x,
                                                pos_y=position_y,
                                                orientation_z=orientation_z,
                                                boxes=[],  # TODO
                                                predictions=p_dict,
                                                goal=goal_pos
                                                )
                conformal_info = controller.update_conformal_var(position_x, position_y, h_dict)

                if not info['feasible']:
                    velocity = np.array([0., 0.])

                else:
                    if count >= 15 and not done:
                        cost = info['cost']
                        eval_metrics['costs'].append(cost)

                infeasible = False if info['feasible'] else True
                eval_metrics['infeasible'].append(infeasible)


                linear_x, angular_z = velocity

                position_x += dt * linear_x * np.cos(orientation_z)
                position_y += dt * linear_x * np.sin(orientation_z)
                orientation_z += dt * angular_z

                xys.append(np.array([position_x, position_y]))

                if visualize:
                    # visualization (controller-specific)
                    render(
                        dataset, ts_key, init_frame, position_x, position_y, orientation_z, robot_img, goal_pos,
                        info, h_dict, f_dict, p_dict, video_dir, asset_dir, intervals=None
                    )
                count += 1

            ts_key += 1
        metric_dict[scene_idx] = eval_metrics
        trajectories.append(xys)

    plt.clf(), plt.cla()
    xmax = -np.inf
    for scene_idx, eval_metrics in metric_dict.items():
        collisions = np.array(eval_metrics['collisions'])
        xmax = max(xmax, len(collisions))
        collisions_cumul = np.cumsum(collisions)
        collisions_asymptotic = collisions_cumul / (1 + np.arange(collisions_cumul.size))
        plt.plot(collisions_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic collision rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(stat_dir, 'collision.png'))
    plt.close()

    plt.clf(), plt.cla()
    xmax = -np.inf
    for scene_idx, eval_metrics in metric_dict.items():
        infeas = np.array(eval_metrics['infeasible'])
        xmax = max(xmax, len(infeas))
        infeasible_cumul = np.cumsum(infeas)
        infeasible_asymptotic = infeasible_cumul / (1 + np.arange(infeasible_cumul.size))
        plt.plot(infeasible_asymptotic)
    plt.xlabel('simulation step')
    plt.ylabel('asymptotic infeasibility rate')
    plt.xlim(0., xmax)
    plt.ylim(0.)
    plt.grid()
    plt.savefig(os.path.join(stat_dir, 'infeasible.png'))
    plt.close()

    return metric_dict, trajectories
