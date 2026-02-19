from __future__ import annotations

import os
import csv
import json
import time
import math
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Callable
import inspect


import numpy as np
from quad_env import QuadWorldEnv3D
from sim_func_3d import run_one_episode_visual_3d
from sim_cp_3d import run_one_episode_rerun_simple
from sim_ecp_3d import run_one_episode_ecp_3d_rerun



# ------------------------------------------------------------
# Helpers: timing stats
# ------------------------------------------------------------


def _stats_ms(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def _safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _short_err(s: str, limit: int = 2000) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    return s[:limit]


# ------------------------------------------------------------
# Standardized outcome labels
# ------------------------------------------------------------
@dataclass
class EpisodeResult:
    # identifiers
    method: str
    seed: int

    # environment summary
    dt: float
    horizon: int
    n_obs: int
    world_bounds_xyz: str

    # knobs (log everything you care about)
    alpha: float
    cp_enabled: bool
    backend: str
    n_paths: int
    n_skip: int
    time_horizon: int

    pred_model_noise: float
    obs_process_noise: float
    gt_future_noise: float

    mode_switch_p: float
    mode_min_ttl: int
    mode_max_ttl: int
    turn_rate_std: float
    stop_go_p: float

    # episode outcomes
    status: str                  # success | collision | timeout | compute_fail | error
    reached_goal: int            # 0/1
    steps: int
    collisions: int
    infeasible_steps: int

    # compute constraint stats
    ctrl_mean_ms: float
    ctrl_p50_ms: float
    ctrl_p90_ms: float
    ctrl_p99_ms: float
    ctrl_max_ms: float
    ctrl_over_dt_rate: float     # fraction of steps where ctrl_time > dt

    loop_mean_ms: float
    loop_p50_ms: float
    loop_p90_ms: float
    loop_p99_ms: float
    loop_max_ms: float
    loop_over_dt_rate: float     # (추가) loop_time > dt

    # misc
    runtime_sec: float
    error_msg: str = ""


# ------------------------------------------------------------
# CSV writer (append-safe)
# ------------------------------------------------------------

def append_csv(path: str, rows: List[EpisodeResult]) -> None:
    if not rows:
        return
    _safe_mkdir_for_file(path)
    write_header = (not os.path.exists(path))
    fieldnames = [f.name for f in fields(EpisodeResult)]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


# ------------------------------------------------------------
# Core: run a single episode (wrap your existing call)
# ------------------------------------------------------------
def run_one_episode_wrapper(
    *,
    QuadWorldEnv3D,                       # class
    run_fn: Callable[[Any], Dict[str, Any]],  # fn(env)->dict
    env_kwargs: Dict[str, Any],
    exp_kwargs: Dict[str, Any],
    method: str,
    seed: int,
    out_dt_fail_frac: float = 0.1,       # e.g., if >5% steps exceed dt -> compute_fail
    fail_on: str = "loop",                # "loop" or "ctrl"
) -> EpisodeResult:
    t0 = time.time()

    env_kwargs = dict(env_kwargs)
    env_kwargs["seed"] = seed
    env = QuadWorldEnv3D(**env_kwargs)

    result = None
    err = ""
    try:
        # exp_kwargs에는 visualize=False / save_rrd=False 권장(대규모용)
        result = run_fn(env, **exp_kwargs)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        result = None

    runtime = time.time() - t0

    # error path
    if result is None:
        return EpisodeResult(
            method=method,
            seed=seed,
            dt=float(env_kwargs["dt"]),
            horizon=int(env_kwargs["horizon"]),
            n_obs=int(env_kwargs["n_obs"]),
            world_bounds_xyz=str(env_kwargs.get("world_bounds_xyz", "")),
            alpha=float(exp_kwargs.get("alpha", math.nan)),
            cp_enabled=bool(exp_kwargs.get("CP", False)),
            backend=str(exp_kwargs.get("backend", "")),
            n_paths=int(exp_kwargs.get("n_paths", -1)),
            n_skip=int(exp_kwargs.get("n_skip", -1)),
            time_horizon=int(exp_kwargs.get("time_horizon", -1)),
            pred_model_noise=float(env_kwargs.get("pred_model_noise", math.nan)),
            obs_process_noise=float(env_kwargs.get("obs_process_noise", math.nan)),
            gt_future_noise=float(env_kwargs.get("gt_future_noise", math.nan)),
            mode_switch_p=float(env_kwargs.get("mode_switch_p", math.nan)),
            mode_min_ttl=int(env_kwargs.get("mode_min_ttl", -1)),
            mode_max_ttl=int(env_kwargs.get("mode_max_ttl", -1)),
            turn_rate_std=float(env_kwargs.get("turn_rate_std", math.nan)),
            stop_go_p=float(env_kwargs.get("stop_go_p", math.nan)),
            status="error",
            reached_goal=0,
            steps=0,
            collisions=0,
            infeasible_steps=0,
            ctrl_mean_ms=math.nan, ctrl_p50_ms=math.nan, ctrl_p90_ms=math.nan, ctrl_p99_ms=math.nan, ctrl_max_ms=math.nan,
            ctrl_over_dt_rate=math.nan,
            loop_mean_ms=math.nan, loop_p50_ms=math.nan, loop_p90_ms=math.nan, loop_p99_ms=math.nan, loop_max_ms=math.nan,
            loop_over_dt_rate=math.nan,
            runtime_sec=float(runtime),
            error_msg=_short_err(err),
        )

    # read expected keys (robust)
    reached_goal = bool(result.get("reached_goal", False))
    steps = int(result.get("steps", 0))
    collisions = int(result.get("collisions", 0))
    infeasible_steps = int(result.get("infeasible_steps", 0))

    ctrl_times_ms = list(result.get("ctrl_times_ms", []))
    loop_times_ms = list(result.get("loop_times_ms", []))

    ctrl_stats = _stats_ms(ctrl_times_ms)
    loop_stats = _stats_ms(loop_times_ms)

    dt_ms = float(env_kwargs["dt"]) * 1000.0
    ctrl_over_dt_rate = float(np.mean([x > dt_ms for x in ctrl_times_ms])) if ctrl_times_ms else math.nan
    loop_over_dt_rate = float(np.mean([x > dt_ms for x in loop_times_ms])) if loop_times_ms else math.nan

    if fail_on not in ("loop", "ctrl"):
        fail_on = "loop"

    miss_rate = loop_over_dt_rate if fail_on == "loop" else ctrl_over_dt_rate
    compute_fail = (not math.isnan(miss_rate)) and (miss_rate > out_dt_fail_frac)

    max_steps = int(exp_kwargs.get("max_steps", steps))

    if compute_fail:
        status = "compute_fail"
    elif collisions > 0:
        status = "collision"
    elif reached_goal:
        status = "success"
    else:
        status = "timeout" if steps >= max_steps else "timeout"

    return EpisodeResult(
        method=method,
        seed=seed,
        dt=float(env_kwargs["dt"]),
        horizon=int(env_kwargs["horizon"]),
        n_obs=int(env_kwargs["n_obs"]),
        world_bounds_xyz=str(env_kwargs.get("world_bounds_xyz", "")),
        alpha=float(exp_kwargs.get("alpha", math.nan)),
        cp_enabled=bool(exp_kwargs.get("CP", False)),
        backend=str(exp_kwargs.get("backend", "")),
        n_paths=int(exp_kwargs.get("n_paths", -1)),
        n_skip=int(exp_kwargs.get("n_skip", -1)),
        time_horizon=int(exp_kwargs.get("time_horizon", -1)),
        pred_model_noise=float(env_kwargs.get("pred_model_noise", math.nan)),
        obs_process_noise=float(env_kwargs.get("obs_process_noise", math.nan)),
        gt_future_noise=float(env_kwargs.get("gt_future_noise", math.nan)),
        mode_switch_p=float(env_kwargs.get("mode_switch_p", math.nan)),
        mode_min_ttl=int(env_kwargs.get("mode_min_ttl", -1)),
        mode_max_ttl=int(env_kwargs.get("mode_max_ttl", -1)),
        turn_rate_std=float(env_kwargs.get("turn_rate_std", math.nan)),
        stop_go_p=float(env_kwargs.get("stop_go_p", math.nan)),
        status=status,
        reached_goal=int(reached_goal),
        steps=steps,
        collisions=collisions,
        infeasible_steps=infeasible_steps,
        ctrl_mean_ms=ctrl_stats["mean"],
        ctrl_p50_ms=ctrl_stats["p50"],
        ctrl_p90_ms=ctrl_stats["p90"],
        ctrl_p99_ms=ctrl_stats["p99"],
        ctrl_max_ms=ctrl_stats["max"],
        ctrl_over_dt_rate=float(ctrl_over_dt_rate) if not math.isnan(ctrl_over_dt_rate) else math.nan,
        loop_mean_ms=loop_stats["mean"],
        loop_p50_ms=loop_stats["p50"],
        loop_p90_ms=loop_stats["p90"],
        loop_p99_ms=loop_stats["p99"],
        loop_max_ms=loop_stats["max"],
        loop_over_dt_rate=float(loop_over_dt_rate) if not math.isnan(loop_over_dt_rate) else math.nan,
        runtime_sec=float(runtime),
        error_msg="",
    )


# ------------------------------------------------------------
# Main suite runner
# ------------------------------------------------------------
def run_suite(
    *,
    QuadWorldEnv3D,
    run_one_episode_visual_3d,      
    env_base: Dict[str, Any],
    exp_base: Dict[str, Any],
    methods: Dict[str, Dict[str, Any]], 
    seeds: List[int],
    csv_path: str,
    out_dt_fail_frac: float = 0.05,
    fail_on: str = "loop",
    dump_json: bool = True,
) -> None:
    _safe_mkdir_for_file(csv_path)

    all_rows: List[EpisodeResult] = []

    # Make a run_fn wrapper to avoid re-binding each time

    def run_fn(env, **exp_kwargs):
        allowed = set(inspect.signature(run_one_episode_visual_3d).parameters.keys())
        clean = {k: v for k, v in exp_kwargs.items() if k in allowed}
        return run_one_episode_visual_3d(env, **clean)

    for method_name, exp_over in methods.items():
        for seed in seeds:
            env_kwargs = dict(env_base)
            exp_kwargs = dict(exp_base)
            exp_kwargs.update(exp_over)

            exp_kwargs.setdefault("visualize", False)
            exp_kwargs.setdefault("save_rrd", False)

            row = run_one_episode_wrapper(
                QuadWorldEnv3D=QuadWorldEnv3D,
                run_fn=run_fn,
                env_kwargs=env_kwargs,
                exp_kwargs=exp_kwargs,
                method=method_name,
                seed=seed,
                out_dt_fail_frac=out_dt_fail_frac,
                fail_on=fail_on,
            )

            all_rows.append(row)
            append_csv(csv_path, [row])

            miss = row.loop_over_dt_rate if fail_on == "loop" else row.ctrl_over_dt_rate
            print(
                f"[saved] {method_name} seed={seed} status={row.status} "
                f"coll={row.collisions} infeas={row.infeasible_steps} steps={row.steps} "
                f"ctrl_p99={row.ctrl_p99_ms:.1f}ms loop_p99={row.loop_p99_ms:.1f}ms "
                f"miss({fail_on})={miss:.3f}"
            )

    if dump_json:
        json_path = os.path.splitext(csv_path)[0] + ".json"
        _safe_mkdir_for_file(json_path)
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in all_rows], f, indent=2)
        print(f"[done] wrote {len(all_rows)} rows to {csv_path} and {json_path}")
    else:
        print(f"[done] wrote {len(all_rows)} rows to {csv_path}")




if __name__ == "__main__":
    methods = {
        #"nocp": {"CP": False},
        #"cp": {"CP": True, "alpha": 0.10},
        "ecp": {}
    }

    run_suite(
        QuadWorldEnv3D=QuadWorldEnv3D,
        run_one_episode_visual_3d=run_one_episode_ecp_3d_rerun,
        env_base=dict(
            dt=0.1, horizon=20, n_obs=280,
            world_bounds_xyz=((-3,7),(-3,7),(0,8)),
            pred_model_noise=0.20, obs_process_noise=0.22, gt_future_noise=0.20,
            mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6,
            turn_rate_std=3.0, stop_go_p=0.6, gui=False,
        ),
        exp_base=dict(
            nx=40, ny=40, nz=40,
            time_horizon=12,
            n_skip=4,
            n_paths=2000,
            max_steps=250,
            backend="loky",
            visualize=False,
            save_rrd=False,

        ),
        methods=methods,
        seeds=list(range(20, 40)),
        csv_path="results/ecp_mpc.csv",
        out_dt_fail_frac=0.1,
        fail_on="loop",
    )