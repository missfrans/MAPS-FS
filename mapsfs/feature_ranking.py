from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Register built-ins on import.
from . import feature_selectors as _builtins  # noqa: F401
from .config import output_dir
from .plugins import load_plugins, require_feature_selector
from .preprocessing import load_prepared, prepare_dataset_seed, validate_cross_seed_schema
from .utils import log, set_seed


def ranking_dir(cfg: Dict[str, Any], dataset: str, seed: int) -> Path:
    return output_dir(cfg) / "02_feature_rankings" / dataset / f"seed_{seed}"


def ranking_path(cfg: Dict[str, Any], dataset: str, seed: int, method: str) -> Path:
    return ranking_dir(cfg, dataset, seed) / f"{method}_ranking.csv"


def _sample(X, y, max_n, seed):
    if max_n is None or len(y) <= int(max_n):
        return np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=int(max_n), replace=False)
    return np.asarray(X)[idx], np.asarray(y)[idx]


def run_rankings_for_seed(cfg: Dict[str, Any], dataset: str, seed: int, force: bool = False):
    set_seed(seed)
    prepare_dataset_seed(cfg, dataset, seed, force=False)
    data = load_prepared(cfg, dataset, seed, mmap=True)
    X_fit, y_fit = data["X_fit"], data["y_fit"]
    feature_names = data["feature_names"]["feature_name"].astype(str).tolist()
    max_n = cfg.get("feature_selection", {}).get("sample_size")
    X_fs, y_fs = _sample(X_fit, y_fit, max_n, seed)

    load_plugins(cfg.get("plugins", []))
    rows = []
    for method in cfg["feature_selection"]["enabled"]:
        out = ranking_path(cfg, dataset, seed, method)
        if out.exists() and not force:
            rank = pd.read_csv(out)
            if len(rank) == len(feature_names) and set(rank["feature_name"].astype(str)) == set(feature_names):
                log(f"RANK SKIP compatible | {dataset} seed={seed} method={method}")
                rows.append(rank.assign(dataset=dataset, seed=seed, fs_method=method))
                continue
        func = require_feature_selector(method)
        params = cfg.get("feature_selection", {}).get("params", {}).get(method, {})
        rank = func(X_fs, y_fs, feature_names, seed, params)
        required = {"rank", "feature_index", "feature_name", "score"}
        if not required.issubset(rank.columns):
            raise ValueError(f"Feature selector '{method}' must return columns {sorted(required)}")
        if len(rank) != len(feature_names) or rank["feature_name"].nunique() != len(feature_names):
            raise ValueError(f"Selector '{method}' did not return a complete one-to-one ranking.")
        out.parent.mkdir(parents=True, exist_ok=True)
        rank.insert(0, "dataset", dataset)
        rank.insert(1, "seed", seed)
        rank.insert(2, "fs_method", method)
        rank.to_csv(out, index=False)
        log(f"RANK DONE | {dataset} seed={seed} method={method}")
        rows.append(rank)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_ranking(cfg: Dict[str, Any], dataset: str, seed: int, method: str) -> pd.DataFrame:
    path = ranking_path(cfg, dataset, seed, method)
    if not path.exists():
        raise FileNotFoundError(f"Ranking missing: {path}")
    return pd.read_csv(path).sort_values("rank")


def run_all_rankings(cfg: Dict[str, Any], datasets: List[str] | None = None, seeds: List[int] | None = None, force=False):
    datasets = datasets or list(cfg["datasets"])
    seeds = seeds or list(cfg["experiment"]["seeds"])
    all_rows = []
    for ds in datasets:
        for seed in seeds:
            all_rows.append(run_rankings_for_seed(cfg, ds, int(seed), force=force))
        if cfg.get("preprocessing", {}).get("require_same_schema_across_seeds", True):
            validate_cross_seed_schema(cfg, ds, [int(s) for s in seeds])
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out = output_dir(cfg) / "02_feature_rankings" / "feature_rankings_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    return combined
