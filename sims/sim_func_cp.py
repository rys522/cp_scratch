import os
import pickle
import numpy as np
from visualization_utils import render
from controllers.func_cp_mpc import FunctionalCPMPC, CPStepParameters

# You already import these in your codebase
from cp.functional_cp import get_envelopes_function  # should build envelope evaluator from offline params


def _stack_pred_traj_world(p_dict, horizon: int):
    """
    Convert PKL prediction dict into arrays WITHOUT any coordinate normalization.

    p_dict: {pid: (T,2)} in world coordinates
    Returns:
        pred_traj: (H, N, 2)
        obst_mask: (H, N)
        p_keys: list
    """
    p_keys = list(p_dict.keys())
    N = len(p_keys)
    pred_traj = np.zeros((horizon, N, 2), dtype=np.float32)
    obst_mask = np.zeros((horizon, N), dtype=bool)

    for j, pid in enumerate(p_keys):
        arr = np.asarray(p_dict[pid], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"p_dict[{pid}] must have shape (T,2), got {arr.shape}")
        take = min(horizon, arr.shape[0])
        if take > 0:
            pred_traj[:take, j, :] = arr[:take, :]
            obst_mask[:take, j] = True

    return pred_traj, obst_mask, p_keys


def build_calibration_residuals_from_pkl(
    prediction_dict: dict,
    future_dict: dict,
    horizon: int,
    calib_ts_keys,
    *,
    require_full_horizon: bool = True,
) -> np.ndarray:
    """
    Returns
    -------
    residuals_train : (N_eff, horizon, 1)
        residuals_train[n, i-1, 0] = || y_true(t+i) - y_pred(t+i|t) ||_2
    """
    H = int(horizon)
    episodes = []

    for ts_key in calib_ts_keys:
        if ts_key not in prediction_dict or ts_key not in future_dict:
            continue

        p_dict = prediction_dict[ts_key]
        f_dict = future_dict[ts_key]

        # Build episodes = (ts_key, pid)
        for pid, pred_traj in p_dict.items():
            if pid not in f_dict:
                continue

            pred_arr = np.asarray(pred_traj, dtype=np.float32)  # (Tp,2)
            true_arr = np.asarray(f_dict[pid], dtype=np.float32)  # (Tt,2)

            if pred_arr.ndim != 2 or pred_arr.shape[1] != 2:
                continue
            if true_arr.ndim != 2 or true_arr.shape[1] != 2:
                continue

            if require_full_horizon:
                if pred_arr.shape[0] < H or true_arr.shape[0] < H:
                    continue
                valid_H = H
            else:
                valid_H = min(H, pred_arr.shape[0], true_arr.shape[0])
                if valid_H <= 0:
                    continue

            r = np.zeros((H, 1), dtype=np.float32)
            for i in range(valid_H):
                r[i, 0] = float(np.linalg.norm(true_arr[i] - pred_arr[i], ord=2))

            # If not full horizon, you can either:
            # (a) drop this episode (require_full_horizon=True), OR
            # (b) keep it with zeros for missing i (not recommended), OR
            # (c) fill missing with NaN and handle NaNs inside CP code.
            episodes.append(r)

    if len(episodes) == 0:
        raise RuntimeError("No valid residual episodes collected (maybe require_full_horizon too strict?).")

    return np.stack(episodes, axis=0)  # (N_eff, H, 1)

def offline_calibrate_or_load(
    prediction_dict,
    futures_dict,
    horizon: int,
    alpha: float,
    cache_path: str | None,
    calib_ts_keys,
) -> CPStepParameters:
    """
    Offline calibration (Algorithm 1 lines 3-9) should be done once and cached.  [oai_citation:4‡functional_cp.pdf](sediment://file_00000000a28871fd956c3c17b9a50c8d)
    This function either loads cached parameters or builds them and caches.

    IMPORTANT:
    - This runner-level offline step does NOT normalize coordinates.
    - The exact calibration object depends on your cp.functional_cp implementation.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            offline_params = pickle.load(f)
        # offline_params could be Theta_i list or whatever your get_envelopes_function expects
        return offline_params

    # Build residual samples from PKL (minimal scalar residual example)
    residuals_train = build_calibration_residuals_from_pkl(
        prediction_dict=prediction_dict,
        future_dict=futures_dict,
        horizon=horizon,
        calib_ts_keys=calib_ts_keys,
        require_full_horizon=True,
)
    # Here you should replace this with your full functional CP calibration that produces Θ_i.
    # The PDF algorithm includes basis/PCA, GMM fit, threshold calibration, QCQP refinement.  [oai_citation:5‡functional_cp.pdf](sediment://file_00000000a28871fd956c3c17b9a50c8d)
    #
    # For now, we pass residuals into your project-level calibrator entrypoint:
    offline_params = get_envelopes_function(
        residuals_train=residuals_train,
        p_base=3,
        K=4,
        alpha=alpha,
        test_size=0.30,
        random_state=0,
        n_jobs=4,
        backend="loky",
        cov_jitter=1e-8,
    )

    return offline_params


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
    *,
    horizon: int = 12,
    n_skip: int = 4,
    n_paths: int = 1200,
    warmup_steps: int = 15,
    hard_cp_constraint: bool = True,
    n_calib_steps: int = 500,
):
    """
    Runner does:
      (1) Offline calibration once (Algorithm 1 lines 3-9) and cache it.  [oai_citation:6‡functional_cp.pdf](sediment://file_00000000a28871fd956c3c17b9a50c8d)
      (2) Online MPC loop: controller uses cached params to evaluate U_i(x) along sampled paths.  [oai_citation:7‡functional_cp.pdf](sediment://file_00000000a28871fd956c3c17b9a50c8d)

    Runner does NOT do any coordinate normalization.
    """

    prediction_dict = predictions["prediction"]
    histories_dict = predictions["history"]
    futures_dict = predictions["future"]

    # ----------------------------
    # Offline calibration (once)
    # ----------------------------
    # Pick calibration timestamps from available keys (exclude evaluation scenario windows if desired)
    all_keys = sorted(list(prediction_dict.keys()))
    calib_ts_keys = all_keys[: min(len(all_keys), n_calib_steps)]

    offline_params = offline_calibrate_or_load(
        prediction_dict=prediction_dict,
        futures_dict=futures_dict,
        horizon=horizon,
        alpha=alpha,
        calib_ts_keys=calib_ts_keys,
    )

    # Controller will only run Online using offline_params
    controller = FunctionalCPMPC(
        n_steps=horizon,
        dt=dt,
        min_linear_x=min_linear_x, max_linear_x=max_linear_x,
        min_angular_z=min_angular_z, max_angular_z=max_angular_z,
        n_skip=n_skip,
        n_paths=n_paths,
        robot_rad=robot_rad,
        obstacle_rad=obstacle_rad,
        hard_cp_constraint=hard_cp_constraint,
        cp_params=offline_params,  # <--- online uses this, runner does offline
        alpha=alpha,
    )

    # ----------------------------
    # Evaluation loop (same style as run_cc)
    # ----------------------------
    stat_dir = os.path.join(os.path.dirname(__file__), "stats", dataset, "cp-mpc")
    os.makedirs(stat_dir, exist_ok=True)

    metric_dict = {}
    trajectories = []

    for scene_idx, scenario_begin in enumerate(scenarios):
        xys = []
        eval_metrics = {
            "collisions": [],
            "costs": [],
            "exit_time": np.inf,
            "infeasible": [],
        }

        position_x, position_y, orientation_z = init_robot_pose

        if visualize:
            assert asset_dir is not None
            video_dir = os.path.join(os.path.dirname(__file__), "videos", dataset, str(scene_idx), "cp-mpc")
            os.makedirs(video_dir, exist_ok=True)
        else:
            video_dir = None

        count = 0
        done = False
        ts_key = scenario_begin

        while count < max_n_steps:
            if ts_key in prediction_dict:
                p_dict = prediction_dict[ts_key]
                h_dict = histories_dict[ts_key]
                f_dict = futures_dict[ts_key]

                pred_traj, obst_mask, _ = _stack_pred_traj_world(p_dict, horizon=horizon)

                if count < warmup_steps:
                    velocity = np.array([0.0, 0.0], dtype=np.float32)
                    info = {"feasible": True, "cost": 0.0}
                else:
                    robot_pos = np.array([position_x, position_y], dtype=np.float32)

                    if h_dict:
                        obs_pos = np.array([o[-1] for o in h_dict.values()], dtype=np.float32)
                        min_obs_dist = float(np.min(np.linalg.norm(obs_pos - robot_pos, axis=1)))
                    else:
                        min_obs_dist = np.inf

                    if float(np.linalg.norm(robot_pos - goal_pos)) <= 0.6:
                        eval_metrics["exit_time"] = count
                        done = True

                    collision = True if (min_obs_dist < (robot_rad + obstacle_rad)) else False
                    if not done:
                        eval_metrics["collisions"].append(collision)

                    # Online only
                    velocity, info = controller(
                        pos_x=float(position_x),
                        pos_y=float(position_y),
                        orientation_z=float(orientation_z),
                        obst_pred_traj=pred_traj,
                        obst_mask=obst_mask,
                        goal=goal_pos,
                    )

                infeasible = not bool(info.get("feasible", False))
                eval_metrics["infeasible"].append(bool(infeasible))

                if infeasible:
                    velocity = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    if count >= warmup_steps and (not done):
                        eval_metrics["costs"].append(float(info.get("cost", 0.0)))

                linear_x, angular_z = float(velocity[0]), float(velocity[1])
                position_x += dt * linear_x * np.cos(orientation_z)
                position_y += dt * linear_x * np.sin(orientation_z)
                orientation_z += dt * angular_z

                xys.append(np.array([position_x, position_y], dtype=np.float32))

                if visualize:
                    render(
                        dataset, ts_key, init_frame,
                        position_x, position_y, orientation_z,
                        robot_img, goal_pos,
                        info, h_dict, f_dict, p_dict,
                        video_dir, asset_dir, intervals=None,
                    )

                count += 1

            ts_key += 1

        metric_dict[scene_idx] = eval_metrics
        trajectories.append(xys)

    # Plots (same as run_cc)
    # ... (keep your plotting code)

    return metric_dict, trajectories