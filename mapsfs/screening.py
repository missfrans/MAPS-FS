from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from patsy import dmatrix

from .config import output_dir
from .runner import run_one
from .utils import json_dump, log


def _candidate_representations(cfg: Dict[str, Any], dataset: str) -> pd.DataFrame:
    rows = []
    for fs in cfg["feature_selection"]["enabled"]:
        for k in cfg["datasets"][dataset]["top_k_values"]:
            rows.append({"fs_method": fs, "top_k": int(k)})
    return pd.DataFrame(rows)


def _full_design_matrix(reps: pd.DataFrame, models: List[str]):
    rows = []
    ks = reps["top_k"].astype(float)
    k_mean, k_std = float(ks.mean()), float(ks.std(ddof=0) or 1.0)
    for _, r in reps.iterrows():
        for m in models:
            rows.append({
                "fs_method": str(r.fs_method), "top_k": int(r.top_k),
                "k_scaled": (float(r.top_k) - k_mean) / k_std,
                "dl_model": m,
            })
    universe = pd.DataFrame(rows)
    X = dmatrix(
        "1 + C(fs_method) + k_scaled + C(dl_model) + C(fs_method):C(dl_model) + k_scaled:C(dl_model)",
        universe, return_type="dataframe"
    )
    universe["_row"] = np.arange(len(universe))
    return universe, np.asarray(X, dtype=float), list(X.design_info.column_names)


def select_d_optimal_blocks(cfg: Dict[str, Any], dataset: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    reps = _candidate_representations(cfg, dataset)
    models = list(cfg["models"]["enabled"])
    universe, X, columns = _full_design_matrix(reps, models)
    full_rank = int(np.linalg.matrix_rank(X))

    block_rows = {}
    for i, r in reps.reset_index(drop=True).iterrows():
        mask = (universe.fs_method == r.fs_method) & (universe.top_k == int(r.top_k))
        block_rows[i] = universe.loc[mask, "_row"].to_numpy(dtype=int)

    selected, remaining = [], set(block_rows)
    current_rows = np.array([], dtype=int)
    ridge = 1e-10
    target = cfg.get("screening", {}).get("representation_blocks", "auto_min_rank")
    target_n = None if str(target) == "auto_min_rank" else int(target)

    while remaining:
        best = None
        best_key = None
        for b in sorted(remaining):
            candidate_rows = np.concatenate([current_rows, block_rows[b]])
            XX = X[candidate_rows]
            rank = int(np.linalg.matrix_rank(XX))
            sign, logdet = np.linalg.slogdet(XX.T @ XX + ridge * np.eye(XX.shape[1]))
            key = (rank, float(logdet if sign > 0 else -np.inf))
            if best_key is None or key > best_key:
                best, best_key = b, key
        selected.append(best)
        remaining.remove(best)
        current_rows = np.concatenate([current_rows, block_rows[best]])
        rank_now = int(np.linalg.matrix_rank(X[current_rows]))
        if target_n is not None and len(selected) >= target_n:
            if rank_now < full_rank:
                raise ValueError(
                    f"screening.representation_blocks={target_n} is insufficient for the predefined screening effects. "
                    f"Current rank={rank_now}, required rank={full_rank}. Increase the budget or use auto_min_rank."
                )
            break
        if target_n is None and rank_now >= full_rank:
            break

    picked = reps.iloc[selected].copy().sort_values(["fs_method", "top_k"]).reset_index(drop=True)
    summary = {
        "dataset": dataset,
        "candidate_representation_blocks": int(len(reps)),
        "selected_representation_blocks": int(len(picked)),
        "models_per_block": int(len(models)),
        "design_matrix_rank": int(np.linalg.matrix_rank(X[current_rows])),
        "full_design_rank": full_rank,
        "fixed_effect_columns": columns,
        "selection_rule": "greedy blockwise D-optimality; rank first, then regularized log-determinant",
    }
    return picked, summary


def build_screen_design(cfg: Dict[str, Any], dataset: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    mode = cfg.get("screening", {}).get("design", "d_optimal_blocks")
    if mode == "all":
        reps = _candidate_representations(cfg, dataset)
        summary = {"dataset": dataset, "candidate_representation_blocks": len(reps), "selected_representation_blocks": len(reps), "selection_rule": "all"}
    else:
        reps, summary = select_d_optimal_blocks(cfg, dataset)
    rows = []
    for _, r in reps.iterrows():
        for model in cfg["models"]["enabled"]:
            rows.append({"dataset": dataset, "fs_method": r.fs_method, "top_k": int(r.top_k), "dl_model": model})
    return pd.DataFrame(rows), summary


def run_dependency_screen(cfg: Dict[str, Any], datasets: List[str] | None = None, resume=True, overwrite_stale=False):
    datasets = datasets or list(cfg["datasets"])
    out_root = output_dir(cfg) / "04_screening"
    pilot_seeds = [int(s) for s in cfg.get("screening", {}).get("pilot_seeds", cfg["experiment"]["seeds"])]
    all_designs = []
    for ds in datasets:
        design, summary = build_screen_design(cfg, ds)
        ds_dir = out_root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        design.to_csv(ds_dir / "screen_design.csv", index=False)
        json_dump(ds_dir / "screen_design_summary.json", summary)
        for _, row in design.iterrows():
            for seed in pilot_seeds:
                run_one(cfg, ds, seed, str(row.fs_method), int(row.top_k), str(row.dl_model), "screen", resume, overwrite_stale)
        all_designs.append(design)
        log(f"SCREEN DONE | {ds}: blocks={summary['selected_representation_blocks']} configs/seed={len(design)} seeds={pilot_seeds}")
    return pd.concat(all_designs, ignore_index=True) if all_designs else pd.DataFrame()
