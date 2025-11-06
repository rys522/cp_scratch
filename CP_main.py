from __future__ import annotations
import os
import numpy as np
from joblib import Parallel, delayed

# ---- imports from your modules ----
from Config import load_experiment_config, build_environment, build_predictor
from envs.brownian import distance_field_single_obstacle  # field builder
from cp.functional_cp import compute_cp_upper_envelopes
from utils import set_mpl_defaults, animate_cp_comparison
import matplotlib.pyplot as plt



# ============================================================
# Helpers
# ============================================================

def build_training_residuals(
    obst_trajs: np.ndarray,          # (N, T, 2)
    Xg: np.ndarray, Yg: np.ndarray,  # meshgrid
    predictor,
    h: int
) -> np.ndarray:
    """
    Build residual tensor for all training episodes.

    residual[i, t-1] = d_pred(x; y_hat_{t+h|t}) - d_true(x; y_{t+h})
    where valid t in [1 .. T-h-1].
    """
    N, T, _ = obst_trajs.shape
    # residual time length
    T_res = T - h - 1
    if T_res <= 0:
        raise ValueError(f"h={h} too large for T={T}")

    # Derive grid dims from meshgrid
    H, W = Xg.shape[0], Xg.shape[1]
    residuals = np.zeros((N, T_res, H, W), dtype=np.float32)

    for i in range(N):
        traj = obst_trajs[i]
        for t in range(1, T - h):
            y_tm1, y_t, y_tph = traj[t - 1], traj[t], traj[t + h]
            y_hat = predictor.predict_h(y_t, y_tm1, h)

            F_true = distance_field_single_obstacle(y_tph, Xg, Yg).astype(np.float32)
            F_pred = distance_field_single_obstacle(y_hat,  Xg, Yg).astype(np.float32)

            residuals[i, t - 1] = F_pred - F_true  # index shift aligns to future t+h
    return residuals


def build_lower_forecast_fields(
    obst_traj_test: np.ndarray,      # (T, 2)
    Xg: np.ndarray, Yg: np.ndarray,
    predictor,
    g_upper_all_t: np.ndarray,       # (T_res, H, W)
    h: int
) -> np.ndarray:
    """
    Build lower-bound distance fields at times (t+h) for a held-out episode:
        lower[t+h] = max( d_pred(y_hat_{t+h|t}) - g_upper_t , 0 )
    """
    T = obst_traj_test.shape[0]
    H, W = Xg.shape
    lower_forecast = np.zeros((T, H, W), dtype=np.float32)

    for t in range(1, T - h):
        y_tm1, y_t = obst_traj_test[t - 1], obst_traj_test[t]
        y_hat = predictor.predict_h(y_t, y_tm1, h)
        d_pred = distance_field_single_obstacle(y_hat, Xg, Yg).astype(np.float32)

        g_up = g_upper_all_t[t - 1]  # residual index aligns to decision time t
        lower_forecast[t + h] = np.maximum(d_pred - g_up, 0.0)
    return lower_forecast


def evaluate_coverage(
    seeds: np.ndarray,
    env,
    predictor,
    Xg: np.ndarray, Yg: np.ndarray,
    g_upper_all_t: np.ndarray,
    h: int,
    safe_threshold: float,
    Tsteps: int
):
    """
    Evaluate mean coverage and missed pixels across seeds.
    Coverage metric:
        For each t, true-unsafe mask = 1{ d_true(t+h) < tau }
                    forecast-unsafe   = 1{ lower_bound(t+h) < tau }
        missed = true_unsafe & ~forecast_unsafe
        coverage_t = 1 - missed / (true_unsafe + eps)
    """
    results = []
    for seed in seeds:
        rng = env.make_rng(int(seed))
        _, obst = env.simulate_pair(Tsteps, rng)

        misses, totals = [], []
        for t in range(1, Tsteps - h):
            F_true = distance_field_single_obstacle(obst[t + h], Xg, Yg)
            y_tm1, y_t = obst[t - 1], obst[t]
            y_hat = predictor.predict_h(y_t, y_tm1, h)
            d_pred = distance_field_single_obstacle(y_hat, Xg, Yg)

            g_up = g_upper_all_t[t - 1]
            # lower < tau  <=>  d_pred - g_up < tau  <=>  d_pred < tau + g_up
            mask_true = (F_true < safe_threshold)
            mask_pred = (d_pred < safe_threshold + g_up)

            missed = np.sum(mask_true & ~mask_pred)
            total = np.sum(mask_true)
            misses.append(missed)
            totals.append(total)

        misses = np.array(misses, dtype=float)
        totals = np.array(totals, dtype=float) + 1e-8
        coverage = 1.0 - (misses / totals)
        results.append((coverage.mean(), misses.mean()))

    covs, missed_means = zip(*results)
    return float(np.mean(covs)), float(np.std(covs)), float(np.mean(missed_means)), float(np.std(missed_means))


# ============================================================
# Main
# ============================================================

def main(override_cfg: dict | None = None, do_viz: bool = False):
    # 1) Load config and instantiate environment + predictor
    cfg = load_experiment_config(override_cfg or {})
    env = build_environment(cfg)
    predictor = build_predictor(cfg)  # expects box to be resolved in config factory

    # 2) Build grid and training corpus
    _, _, Xg, Yg = env.build_grid()
    # Simulate training set (robot, obstacle, fields)
    robot_trajs, obst_trajs = env.simulate_dataset(
        N=cfg.train.N, Tsteps=cfg.train.Tsteps, seed=cfg.train.seed, return_fields=False
    )

    # 3) Residuals for CP (N, T_res, H, W)
    residuals_train = build_training_residuals(
        obst_trajs=obst_trajs, Xg=Xg, Yg=Yg, predictor=predictor, h=cfg.time_horizon
    )
    N, T_res, H, W = residuals_train.shape
    print(f"[Info] residuals_train: N={N}, T_res={T_res}, H={H}, W={W}")

    # 4) CP upper envelopes per residual time index
    #    functional_cp.compute_cp_upper_envelopes signature is assumed as:
    #    g_upper_all_t = compute_cp_upper_envelopes(
    #         residuals_train, p_base, K, alpha, test_size, random_state, n_jobs, backend
    #    )
    g_upper_all_t = compute_cp_upper_envelopes(
        residuals_train=residuals_train,
        p_base=cfg.cp.p_base,
        K=cfg.cp.K,
        alpha=cfg.cp.alpha,
        test_size=cfg.cp.test_size,
        random_state=cfg.cp.random_state,
        n_jobs=cfg.cp.n_jobs,
        backend=cfg.cp.backend,
    ).astype(np.float32)  # (T_res, H, W)
    print(f"[Info] g_upper_all_t: shape={g_upper_all_t.shape}")

    # 5) Held-out episode for visualization and lower-bound fields
    r_test, o_test = env.simulate_pair(cfg.train.Tsteps, env.make_rng(cfg.train.seed + 777))
    lower_forecast = build_lower_forecast_fields(
        obst_traj_test=o_test,
        Xg=Xg, Yg=Yg,
        predictor=predictor,
        g_upper_all_t=g_upper_all_t,
        h=cfg.time_horizon
    )

    # 6) Coverage evaluation over multiple seeds
    seeds = np.arange(cfg.train.seed + 100, cfg.train.seed + 110)
    mean_cov, std_cov, mean_miss, std_miss = evaluate_coverage(
        seeds=seeds,
        env=env,
        predictor=predictor,
        Xg=Xg, Yg=Yg,
        g_upper_all_t=g_upper_all_t,
        h=cfg.time_horizon,
        safe_threshold=cfg.safety.safe_threshold,
        Tsteps=cfg.train.Tsteps
    )
    print(f"[h={cfg.time_horizon}] Mean coverage: {mean_cov*100:.2f}% ± {std_cov*100:.2f}%")
    print(f"[h={cfg.time_horizon}] Avg missed pixels: {mean_miss:.1f} ± {std_miss:.1f}")

    # 7) Optional visualization
    if do_viz:
        set_mpl_defaults()
        # Build true distance fields for the held-out obstacle
        T = o_test.shape[0]
        true_fields = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            true_fields[t] = distance_field_single_obstacle(o_test[t], Xg, Yg)

        ani = animate_cp_comparison(
            robot_traj=r_test,
            obst_traj=o_test,
            true_fields=true_fields,
            lower_forecast_fields=lower_forecast,
            box=getattr(env, "box", 100.0),
            safe_threshold=cfg.safety.safe_threshold,
            h=cfg.time_horizon,
            interval=150
        )
        plt.show()

    return {
        "cfg": cfg,
        "residuals_train": residuals_train,
        "g_upper_all_t": g_upper_all_t,
        "lower_forecast": lower_forecast,
        "coverage": {
            "mean": mean_cov, "std": std_cov,
            "miss_mean": mean_miss, "miss_std": std_miss
        }
    }

if __name__ == "__main__":
    main(override_cfg=None, do_viz=True)