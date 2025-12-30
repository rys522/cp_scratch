import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from visualization_utils import render
from controllers.func_cp_mpc import FunctionalCPMPC
from utils import build_grid, distance_field_points
from sim_utils import compute_lower_field_single_step, min_dist_robot_to_peds
from cp.functional_cp import CPStepParameters, get_envelopes_function


def run_func_cp_mpc(
    dataset,
    scenarios,
    init_robot_pose,
    goal_pos,
    max_linear_x,
    min_linear_x,
    max_angular_z,
    min_angular_z,
    predictions,
    dt,
    init_frame,
    visualize,
    asset_dir,
    robot_img,
    max_n_steps,
    robot_rad,
    obstacle_rad,
    alpha,
    box_size=100.0,
    grid_res=128
):
    """
    Run CP-MPC simulation over multiple scenarios.

    This function mirrors the interface of `run_ecp_mpc` for fair comparison.

    Args:
        dataset (str): Dataset name (e.g., 'zara1', 'univ')
        scenarios (list): Starting timestamps of evaluation scenarios
        init_robot_pose (tuple): (x, y, yaw) initial robot pose
        goal_pos (np.ndarray): Goal position in world coordinates
        predictions (dict): Pedestrian prediction / history / future dicts
        dt (float): Control sampling time
        visualize (bool): Whether to render videos
        max_n_steps (int): Maximum simulation steps
        robot_rad (float): Robot radius
        obstacle_rad (float): Pedestrian radius
        alpha (float): Target CP miscoverage level
    """

    # Unpack prediction dictionaries
    prediction_dict = predictions['prediction']
    histories_dict = predictions['history']
    futures_dict = predictions['future']

    # Directory for saving evaluation statistics
    stat_dir = os.path.join(os.path.dirname(__file__), 'stats', dataset, 'cp-mpc')
    os.makedirs(stat_dir, exist_ok=True)

    # Build spatial grid for distance field evaluation
    world_center = np.array([box_size / 2.0, box_size / 2.0], dtype=np.float32)
    xs, ys, Xg, Yg = build_grid(box_size, grid_res, grid_res)

    # ============================================================
    # CP upper envelope parameter
    # ============================================================
    # ⚠️ Placeholder:
    # In the full pipeline, g_upper should be obtained from
    # conformal calibration on a held-out dataset.
    horizon = 12
    g_upper = np.full((horizon,), 0.5, dtype=np.float32)

    metric_dict = {}
    trajectories = []

    for scene_idx, scenario_begin in enumerate(scenarios):

        eval_metrics = {
            'collisions': [],
            'costs': [],
            'exit_time': np.inf,
            'miscoverage': [],
            'infeasible': []
        }
        xys = []

        cp_params = get_envelopes_function(
            residuals_train=residuals_train,
            p_base=3, K=4, alpha=alpha,
            test_size=0.30,
            random_state=0,
            n_jobs=4,
            backend="loky",
            split_alpha=True,
            cov_jitter=1e-8
            )

        # ========================================================
        # Instantiate CP-MPC controller
        # ========================================================
        controller = FunctionalCPMPC(
            box=box_size,
            world_center=world_center,
            min_linear_x=-min_linear_x, max_linear_x=max_linear_x,
            min_angular_z=min_angular_z, max_angular_z=max_angular_z,
            g_upper=g_upper,
            grid_H=grid_res,
            grid_W=grid_res,
            n_steps=horizon,
            dt=dt,
            n_skip=4,
            n_paths=1200,
            robot_rad=robot_rad,
            obstacle_rad=obstacle_rad,
            hard_cp_constraint=True
        )

        if not prediction_dict:
            return

        position_x, position_y, orientation_z = init_robot_pose

        if visualize:
            video_dir = os.path.join(
                os.path.dirname(__file__), 'videos', dataset, str(scene_idx), 'cp-mpc'
            )
            os.makedirs(video_dir, exist_ok=True)
            print(f'Visualization saved to {video_dir}')

        count = 0
        done = False
        ts_key = scenario_begin

        # ========================================================
        # Main simulation loop
        # ========================================================
        while count < max_n_steps:

            if ts_key in prediction_dict:
                p_dict = prediction_dict[ts_key]
                h_dict = histories_dict[ts_key]
                f_dict = futures_dict[ts_key]

                # -----------------------------------------------
                # Format pedestrian predictions for MPC
                # -----------------------------------------------
                p_keys = list(p_dict.keys())
                num_peds = len(p_keys)

                pred_traj = np.zeros((horizon, num_peds, 2), dtype=np.float32)
                obst_mask = np.ones((horizon, num_peds), dtype=bool)

                for i, pid in enumerate(p_keys):
                    pred_traj[:, i, :] = p_dict[pid][:horizon]

                # -----------------------------------------------
                # Miscoverage evaluation (safety monitoring)
                # -----------------------------------------------
                if num_peds > 0:
                    lower_field = compute_lower_field_single_step(
                        pred_traj[0],
                        obst_mask[0],
                        Xg,
                        Yg,
                        g_upper[0],
                        world_center
                    )

                    actual_peds = np.array([
                        f_dict[pid][0] for pid in p_keys if pid in f_dict
                    ])

                    if actual_peds.size > 0:
                        d_true = distance_field_points(
                            actual_peds - world_center, Xg, Yg
                        )
                        violation = np.mean(d_true < (lower_field - 1e-3))
                        eval_metrics['miscoverage'].append(float(violation))

                # -----------------------------------------------
                # MPC execution
                # -----------------------------------------------
                if count < 15:
                    # Warm-up phase: no motion
                    velocity = np.array([0., 0.])
                    info = {'feasible': True, 'cost': 0.0}
                else:
                    robot_pos = np.array([position_x, position_y])

                    if h_dict:
                        obs_pos = np.array([h[-1] for h in h_dict.values()])
                        min_dist = np.min(
                            np.linalg.norm(obs_pos - robot_pos, axis=1)
                        )
                        collision = min_dist < (robot_rad + obstacle_rad)
                    else:
                        collision = False

                    if not done:
                        eval_metrics['collisions'].append(collision)

                    if np.linalg.norm(robot_pos - goal_pos) <= 0.6 and not done:
                        eval_metrics['exit_time'] = count
                        done = True

                    start = time.time()
                    velocity, info = controller(
                        pos_x=float(position_x),
                        pos_y=float(position_y),
                        orientation_z=float(orientation_z),
                        obst_pred_traj=pred_traj,
                        obst_mask=obst_mask,
                        goal=goal_pos
                    )

                    infeasible = 0. if info.get('feasible', False) else 1.
                    eval_metrics['infeasible'].append(infeasible)

                # -----------------------------------------------
                # State update
                # -----------------------------------------------
                if not info.get('feasible', False):
                    velocity = np.array([0., 0.])
                elif count >= 15 and not done:
                    eval_metrics['costs'].append(info.get('cost', 0.0))

                linear_x, angular_z = velocity
                position_x += dt * linear_x * np.cos(orientation_z)
                position_y += dt * linear_x * np.sin(orientation_z)
                orientation_z += dt * angular_z

                xys.append(np.array([position_x, position_y]))

                if visualize:
                    render(
                        dataset,
                        ts_key,
                        init_frame,
                        position_x,
                        position_y,
                        orientation_z,
                        robot_img,
                        goal_pos,
                        info,
                        h_dict,
                        f_dict,
                        p_dict,
                        video_dir,
                        asset_dir
                    )

                count += 1

            ts_key += 1

        trajectories.append(xys)
        metric_dict[scene_idx] = eval_metrics

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