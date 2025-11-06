# config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Callable
import importlib
import copy
import os

# -------------------------
# Atomic configs (no grid here)
# -------------------------

@dataclass
class TrainConfig:
    Tsteps: int = 120
    N: int = 1000
    seed: int = 2023
    n_workers: Optional[int] = None  # None => auto
    backend: str = "loky"  # for joblib

@dataclass
class SafetyConfig:
    safe_threshold: float = 8.0

@dataclass
class CPConfig:
    p_base: int = 3
    K: int = 4
    alpha: float = 0.05
    test_size: float = 0.30
    random_state: int = 0
    n_jobs: int = 1

@dataclass
class PredictorConfig:
    # registry key under predictions/
    name: str = "cv"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvironmentConfig:
    # registry key under envs/
    name: str = "brownian"
    # all environment-specific params (grid/motion/etc.) live here
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    cp: CPConfig = field(default_factory=CPConfig)

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    time_horizon: int = 3
    debug: bool = False

    def workers(self) -> int:
        if self.train.n_workers is not None:
            return max(1, int(self.train.n_workers))
        return max(1, (os.cpu_count() or 4) - 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -------------------------
# In-code overrides
# -------------------------

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _coerce_dataclass(cls, src: Dict[str, Any]):
    return cls(**{k: v for k, v in (src or {}).items() if k in cls.__dataclass_fields__})

def load_experiment_config(override: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    base = ExperimentConfig().to_dict()
    merged = _deep_update(base, override or {})

    train = _coerce_dataclass(TrainConfig, merged.get("train", {}))
    safety = _coerce_dataclass(SafetyConfig, merged.get("safety", {}))
    cp = _coerce_dataclass(CPConfig, merged.get("cp", {}))

    environment = _coerce_dataclass(EnvironmentConfig, merged.get("environment", {}))
    predictor = _coerce_dataclass(PredictorConfig, merged.get("predictor", {}))

    time_horizon = int(merged.get("time_horizon", 3))
    debug = bool(merged.get("debug", False))

    cfg = ExperimentConfig(
        train=train, safety=safety, cp=cp,
        environment=environment, predictor=predictor,
        time_horizon=time_horizon, debug=debug
    )
    validate_cfg(cfg)
    return cfg

# -------------------------
# Registry + factories
# -------------------------

_ENV_REGISTRY: Dict[str, Callable[[ExperimentConfig], Any]] = {}
_PRED_REGISTRY: Dict[str, Callable[[ExperimentConfig], Any]] = {}

def register_environment(name: str):
    def _wrap(fn: Callable[[ExperimentConfig], Any]):
        _ENV_REGISTRY[name] = fn
        return fn
    return _wrap

def register_predictor(name: str):
    def _wrap(fn: Callable[[ExperimentConfig], Any]):
        _PRED_REGISTRY[name] = fn
        return fn
    return _wrap

@register_environment("brownian")
def _build_brownian_env(cfg: ExperimentConfig):
    """
    Requires: envs/brownian.py -> class BrownianEnvironment(...)
    All env params (grid/motion/etc.) must be passed via cfg.environment.params.
    """
    mod = importlib.import_module("envs.brownian")
    BrownianEnvironment = getattr(mod, "BrownianEnvironment")
    env = BrownianEnvironment(**(cfg.environment.params or {}))
    return env

@register_predictor("cv")
def _build_cv_predictor(cfg: ExperimentConfig):

    # figure out box
    box = None
    if "box" in (cfg.predictor.params or {}):
        box = cfg.predictor.params["box"]
    elif "box" in (cfg.environment.params or {}):
        box = cfg.environment.params["box"]
    else:
        try:
            mod_env = importlib.import_module("envs.brownian")
            DEFAULTS = getattr(mod_env, "DEFAULT_ENV_PARAMS", None)
            if isinstance(DEFAULTS, dict) and "box" in DEFAULTS:
                box = DEFAULTS["box"]
        except Exception:
            pass
    if box is None:
        box = 100.0  # safe fallback

    mod = importlib.import_module("prediction.cv")
    ConstantVelocityPredictor = getattr(mod, "ConstantVelocityPredictor")
    # pass through remaining predictor params (without double-passing box)
    pred_kwargs = dict(cfg.predictor.params or {})
    pred_kwargs["box"] = box
    predictor = ConstantVelocityPredictor(**pred_kwargs)
    return predictor

def build_environment(cfg: ExperimentConfig):
    name = cfg.environment.name
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown environment '{name}'. Registered: {list(_ENV_REGISTRY.keys())}")
    return _ENV_REGISTRY[name](cfg)

def build_predictor(cfg: ExperimentConfig):
    name = cfg.predictor.name
    if name not in _PRED_REGISTRY:
        raise KeyError(f"Unknown predictor '{name}'. Registered: {list(_PRED_REGISTRY.keys())}")
    return _PRED_REGISTRY[name](cfg)

# -------------------------
# Validation
# -------------------------

def validate_cfg(cfg: ExperimentConfig) -> None:
    if cfg.train.Tsteps <= 2:
        raise ValueError("Tsteps must be > 2.")
    if cfg.time_horizon < 1:
        raise ValueError("time_horizon must be >= 1.")
    if cfg.time_horizon >= cfg.train.Tsteps - 1:
        raise ValueError("time_horizon too large relative to Tsteps.")
    if cfg.cp.p_base < 1:
        raise ValueError("cp.p_base must be >= 1.")
    if not (0.0 < cfg.cp.alpha < 1.0):
        raise ValueError("cp.alpha must be in (0, 1).")