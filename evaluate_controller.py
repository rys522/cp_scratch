import os
import argparse
import numpy as np
import pickle
import json

from sims.sim_acp_mpc import run_acp_mpc
from sims.sim_cc import run_cc
from sims.sim_ecp_mpc import run_ecp_mpc


seed = 0
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="id of dataset to evaluate on", type=str)
parser.add_argument("--controller", help="control method to use", type=str)
parser.add_argument("--visualize", help="control method to use", action='store_true')
parser.add_argument("--asset_dir", help="asset dirpath for visualization", type=str)
args = parser.parse_args()


def load_prediction_results(dataset):
    with open(os.path.join(os.path.dirname(__file__), 'predictions/{}.pkl'.format(dataset)), 'rb') as f:
        res = pickle.load(f)
    return res


if __name__ == "__main__":

    # simulation step
    dt = 0.4

    # robot parameters
    robot_rad = 0.4
    obstacle_rad = 1. / np.sqrt(2.)
    max_linear_x = 0.8
    min_linear_x = -0.8
    max_angular_z = 0.7
    min_angular_z = -0.7

    if args.visualize:
        if not os.path.isdir(args.asset_dir):
            raise OSError(
                "A valid asset directory path must be provided for visualization. "
                "If you do not have one, please run video_parser.py first."
            )
        else:
            from PIL import Image
            asset_dir = args.asset_dir
            print('The dataset frames will be loaded from', asset_dir)
            robot_img = Image.open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/robot.png"))
    else:
        asset_dir = None
        robot_img = None

    # controller-specific arguments
    controller_configs = {
        'acp-mpc': {'target_miscoverage_level': 0.2, 'step_size': 0.05},
        'ecp-mpc': {'target_miscoverage_level': 0.1, 'step_size': 0.02},
        'cc': {'risk_level': -2., 'step_size': 5000}
    }

    eval_functions = {
        'cc': run_cc,
        'acp-mpc': run_acp_mpc,
        'ecp-mpc': run_ecp_mpc
    }

    eval_task_configs = {
        # 'zara1': {'init_robot_pose': np.array([14., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
        'zara1': {'init_robot_pose': np.array([12., 5., np.pi]), 'goal_pos': np.array([3., 6.])},
        'zara2': {'init_robot_pose': np.array([1., 6., 0.]), 'goal_pos': np.array([14., 5.])},
        'hotel': {'init_robot_pose': np.array([-1.5, 0., -np.pi / 2]), 'goal_pos': np.array([2., -6.])},
        'eth': {'init_robot_pose': np.array([5., 1.0, np.pi / 2.]), 'goal_pos': np.array([3., 10.])},
        'univ': {'init_robot_pose': np.array([3.5, 2., np.pi / 4.]), 'goal_pos': np.array([11.5, 8.5])},
    }

    scenarios = {
        'zara1': [100, 200, 300],
        'zara2': [100, 200, 300],
        'eth': [100, 200, 300],
        'hotel': [100, 200, 300],
        'univ': [100]
    }

    init_frames = {
        'zara1': 0,
        'zara2': 1,
        'eth': 78,
        'hotel': 0,
        'univ': 0,
    }  # just for matching the correct frame

    max_n_steps = {
        'zara1': 100,
        'zara2': 100,
        'eth': 100,
        'hotel': 100,
        'univ': 300
    }

    task_kwargs = eval_task_configs[args.dataset]
    eval_func = eval_functions[args.controller]
    ctrl_kwargs = controller_configs[args.controller]

    predictions = load_prediction_results(args.dataset)

    metric_dict, trajectories = eval_func(
        dataset=args.dataset,
        scenarios=scenarios[args.dataset],
        max_linear_x=max_linear_x,
        min_linear_x=min_linear_x,
        max_angular_z=max_angular_z,
        min_angular_z=min_angular_z,
        predictions=predictions,
        dt=dt,
        init_frame=init_frames[args.dataset],
        visualize=args.visualize,
        asset_dir=asset_dir,
        robot_img=robot_img,
        max_n_steps=max_n_steps[args.dataset],
        robot_rad=robot_rad,
        obstacle_rad=obstacle_rad,
        **task_kwargs,
        **ctrl_kwargs
    )

    os.makedirs(os.path.join(os.path.dirname(__file__), 'traj'), exist_ok=True)
    np.save(os.path.join(os.path.dirname(__file__), 'traj/{}_{}.npy'.format(args.dataset, args.controller)), trajectories)

    dict_to_save = {
        'collision': [],
        'cost': [],
        'time': [],
        'infeasible': [],
        'miscoverage': []
    }

    print('dataset: {} / controller: {}'.format(args.dataset, args.controller))
    for scene_idx, eval_metric in metric_dict.items():
        print('-------- scene {} --------'.format(scene_idx))
        n_collisions = np.sum(eval_metric['collisions'])
        collision_ratio = n_collisions / len(eval_metric['collisions'])
        print('* collision_ratio={}'.format(collision_ratio))
        dict_to_save['collision'].append(collision_ratio)

        avg_cost = np.mean(eval_metric['costs'])
        print('* avg cost={:.4f}'.format(avg_cost))
        dict_to_save['cost'].append(avg_cost)

        exit_time = eval_metric['exit_time']
        exit_time = min(exit_time, max_n_steps[args.dataset])
        print('* exit time={}'.format(exit_time))
        dict_to_save['time'].append(exit_time)

        if 'infeasible' in eval_metric:
            n_infeasible = np.sum(eval_metric['infeasible'])
            infeasible_ratio = n_infeasible / len(eval_metric['infeasible'])
            print('* infeasible_ratio={}'.format(infeasible_ratio))
            dict_to_save['infeasible'].append(infeasible_ratio)

        if 'miscoverage' in eval_metric:
            n_miscoverage = np.sum(eval_metric['miscoverage'])
            miscoverage_ratio = n_miscoverage / len(eval_metric['miscoverage'])
            print('* asymptotic miscoverage={:.4f}'.format(miscoverage_ratio))
            dict_to_save['miscoverage'].append(miscoverage_ratio)

    save_dir = os.path.join(os.path.dirname(__file__), 'metric')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, '{}_{}.json'.format(args.dataset, args.controller)), 'w') as f:
        json.dump(dict_to_save, f, ensure_ascii=False, indent=4)
