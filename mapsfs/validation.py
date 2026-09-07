from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import output_dir, validate_config
from .plugins import FEATURE_SELECTORS, MODELS, load_plugins
from .preprocessing import read_table, resolve_dataset_paths
from .runner import collect_all_metrics
from .utils import log


def preflight(cfg: Dict[str, Any]) -> None:
    validate_config(cfg)
    # Register built-ins.
    from . import feature_selectors as _fs  # noqa
    from . import models as _models  # noqa
    load_plugins(cfg.get("plugins", []))
    missing_fs = sorted(set(cfg["feature_selection"]["enabled"]) - set(FEATURE_SELECTORS))
    missing_models = sorted(set(cfg["models"]["enabled"]) - set(MODELS))
    if missing_fs or missing_models:
        raise ValueError(f"Unregistered methods. Feature selectors={missing_fs}; models={missing_models}")
    for name, ds in cfg["datasets"].items():
        paths = resolve_dataset_paths(ds)
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
        p0 = paths[0]
        if p0.suffix.lower() == ".csv":
            sample = pd.read_csv(p0, nrows=5, low_memory=False)
        elif p0.suffix.lower() in {".xlsx", ".xls"}:
            sample = pd.read_excel(p0, nrows=5)
        else:
            sample = read_table(p0).head(5)
        if ds["label_col"] not in sample.columns:
            raise KeyError(f"{name}: label_col '{ds['label_col']}' not found in {paths[0]}")
        if set(ds.get("drop_cols", [])) & set(ds.get("identifier_cols", [])):
            log(f"PREFLIGHT NOTE | {name}: some columns appear in both drop_cols and identifier_cols; this is harmless.")
    log("PREFLIGHT PASSED")


def validate_run_store(cfg: Dict[str, Any]) -> pd.DataFrame:
    df = collect_all_metrics(cfg)
    if not len(df):
        return df
    key = ["dataset", "seed", "fs_method", "top_k", "dl_model"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate configuration keys detected in collected run metrics.")
    bad = df[(df.f1 < 0) | (df.f1 > 1) | (df.far < 0) | (df.far > 1)]
    if len(bad):
        raise ValueError(f"Out-of-range metrics found: {len(bad)} rows")
    return df
