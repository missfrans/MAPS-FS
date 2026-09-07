from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


def _resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    default_root = path.parent.parent if path.parent.name == "configs" else path.parent
    root_value = cfg.get("project_root")
    if root_value is None:
        root = default_root.resolve()
    else:
        rv = Path(root_value)
        root = rv.resolve() if rv.is_absolute() else (path.parent / rv).resolve()

    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(root)
    cfg["_config_hash"] = stable_hash(_strip_runtime_keys(cfg))

    for name, ds in cfg.get("datasets", {}).items():
        if "path" in ds:
            ds["path"] = str(_resolve_path(root, ds["path"]))
        if "paths" in ds:
            ds["paths"] = [str(_resolve_path(root, p)) for p in ds["paths"]]

    out = cfg.setdefault("paths", {}).get("output_dir", "outputs")
    cfg["paths"]["output_dir"] = str(_resolve_path(root, out))
    return cfg


def _strip_runtime_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for key in list(out):
        if str(key).startswith("_"):
            out.pop(key, None)
    return out


def training_protocol_hash(cfg: Dict[str, Any], dataset_name: str, fs_method: str, model_name: str) -> str:
    """Hash only settings capable of changing trained predictions.

    Analysis-only settings such as alpha or Pareto objective order are intentionally
    excluded so that changing an analysis threshold does not invalidate trained runs.
    """
    dataset_cfg = copy.deepcopy(cfg["datasets"][dataset_name])
    if "path" in dataset_cfg:
        dataset_cfg["path"] = Path(dataset_cfg["path"]).name
    if "paths" in dataset_cfg:
        dataset_cfg["paths"] = [Path(p).name for p in dataset_cfg["paths"]]
    payload = {
        "dataset": dataset_cfg,
        "splits": cfg.get("splits", {}),
        "preprocessing": cfg.get("preprocessing", {}),
        "feature_selection": {
            "method": fs_method,
            "params": cfg.get("feature_selection", {}).get("params", {}).get(fs_method, {}),
            "sample_size": cfg.get("feature_selection", {}).get("sample_size"),
        },
        "model": {
            "name": model_name,
            "params": cfg.get("models", {}).get("params", {}).get(model_name, {}),
        },
        "training": cfg.get("training", {}),
    }
    return stable_hash(payload)


def output_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["output_dir"]).resolve()


def validate_config(cfg: Dict[str, Any]) -> None:
    required = ["datasets", "experiment", "feature_selection", "models", "splits", "training", "screening", "statistics"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")

    seeds = cfg["experiment"].get("seeds", [])
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required; five or more are recommended for final inference.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds contains duplicates.")

    fs = cfg["feature_selection"].get("enabled", [])
    models = cfg["models"].get("enabled", [])
    if not fs or not models:
        raise ValueError("Enable at least one feature selector and one model.")
    if len(models) < 2:
        raise ValueError("MAPS-FS model-awareness requires at least two downstream models.")

    alpha = float(cfg["statistics"].get("alpha", 0.05))
    if not (0 < alpha < 1):
        raise ValueError("statistics.alpha must be between 0 and 1.")

    policy = cfg["training"].get("threshold_policy", "fixed")
    if policy not in {"fixed", "validation_f1", "validation_youden"}:
        raise ValueError("training.threshold_policy must be fixed, validation_f1, or validation_youden.")

    design_mode = cfg["screening"].get("design", "d_optimal_blocks")
    if design_mode not in {"d_optimal_blocks", "all"}:
        raise ValueError("screening.design must be d_optimal_blocks or all.")

    for name, ds in cfg["datasets"].items():
        if "label_col" not in ds:
            raise ValueError(f"Dataset {name} is missing label_col.")
        if not (ds.get("path") or ds.get("paths")):
            raise ValueError(f"Dataset {name} must define path or paths.")
        ks = ds.get("top_k_values", [])
        if not ks or any(int(k) <= 0 for k in ks):
            raise ValueError(f"Dataset {name} requires positive top_k_values.")
        if len(set(map(int, ks))) != len(ks):
            raise ValueError(f"Dataset {name} contains duplicate top_k_values.")
