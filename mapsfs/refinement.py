from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import output_dir
from .runner import run_one, fetch_metrics
from .utils import json_load, log


def _gate_paths(cfg, dataset):
    root = output_dir(cfg) / "05_gate" / dataset
    return root, root / "gate_decision.json", root / "prioritized_branches.csv", root / "common_decision.csv"


def _run_full_baselines(cfg, dataset: str, seeds: List[int], resume: bool, overwrite_stale: bool):
    for model in cfg["models"]["enabled"]:
        for seed in seeds:
            run_one(cfg, dataset, seed, "full", "all", model, "baseline_full", resume, overwrite_stale)


def _run_common(cfg, dataset: str, seeds: List[int], common_fs: str, common_k: int, resume: bool, overwrite_stale: bool):
    for model in cfg["models"]["enabled"]:
        for seed in seeds:
            run_one(cfg, dataset, seed, common_fs, common_k, model, "baseline_common", resume, overwrite_stale)



def choose_one_se_k(agg: pd.DataFrame) -> tuple[int, int, float, float]:
    """Return (best_k, selected_k, best_mean, one_se_threshold).

    `agg` must contain top_k_num, f1_mean, and f1_se. The smallest k whose
    mean remains within one standard error of the branch-specific best is kept.
    """
    best = agg.sort_values(["f1_mean", "top_k_num"], ascending=[False, True]).iloc[0]
    se = float(best.f1_se if not pd.isna(best.f1_se) else 0.0)
    band = float(best.f1_mean - se)
    chosen = agg[agg.f1_mean >= band].sort_values("top_k_num").iloc[0]
    return int(best.top_k_num), int(chosen.top_k_num), float(best.f1_mean), band


def run_refinement(cfg: Dict[str, Any], dataset: str, resume=True, overwrite_stale=False) -> pd.DataFrame:
    root, gate_path, branches_path, common_path = _gate_paths(cfg, dataset)
    gate = json_load(gate_path)
    common = pd.read_csv(common_path).iloc[0]
    common_fs, common_k = str(common.common_fs_method), int(common.common_top_k)
    seeds = [int(s) for s in cfg.get("refinement", {}).get("seeds", cfg["experiment"]["seeds"])]
    _run_full_baselines(cfg, dataset, seeds, resume, overwrite_stale)
    _run_common(cfg, dataset, seeds, common_fs, common_k, resume, overwrite_stale)

    out = output_dir(cfg) / "06_refinement" / dataset
    out.mkdir(parents=True, exist_ok=True)
    trajectory_rows, summary_rows = [], []

    if gate["gate"] == "NO":
        # No model-specific expansion. The common screened representation is retained.
        for model in cfg["models"]["enabled"]:
            design = pd.DataFrame([{"fs_method": common_fs, "top_k": common_k, "dl_model": model}])
            m = fetch_metrics(cfg, dataset, seeds, design)
            summary_rows.append({
                "dataset": dataset, "path": "model_independent", "dl_model": model,
                "fs_method": common_fs, "best_k": common_k, "one_se_k": common_k,
                "best_f1_mean": float(m.f1.mean()), "best_f1_se": float(m.f1.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else 0.0,
                "selected_f1_mean": float(m.f1.mean()), "selected_f1_std": float(m.f1.std(ddof=1)) if len(m) > 1 else 0.0,
            })
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(out / "refinement_summary.csv", index=False)
        pd.DataFrame(trajectory_rows).to_csv(out / "refinement_trajectories.csv", index=False)
        log(f"REFINE | {dataset}: gate NO, retained common representation {common_fs}-{common_k}")
        return summary

    branches = pd.read_csv(branches_path)
    top_k_values = [int(k) for k in cfg["datasets"][dataset]["top_k_values"]]
    for _, br in branches.iterrows():
        fs, model = str(br.fs_method), str(br.dl_model)
        for k in top_k_values:
            for seed in seeds:
                run_one(cfg, dataset, seed, fs, k, model, "refine", resume, overwrite_stale)
        design = pd.DataFrame([{"fs_method": fs, "top_k": k, "dl_model": model} for k in top_k_values])
        metrics = fetch_metrics(cfg, dataset, seeds, design)
        metrics["top_k_num"] = pd.to_numeric(metrics["top_k"]).astype(int)
        agg = metrics.groupby("top_k_num").agg(
            f1_mean=("f1", "mean"), f1_std=("f1", "std"), far_mean=("far", "mean"),
            latency_us_mean=("inference_time_per_flow_us", "mean"), n_features_mean=("n_features", "mean"), n=("seed", "nunique")
        ).reset_index()
        agg["f1_se"] = agg["f1_std"] / np.sqrt(agg["n"].clip(lower=1))
        agg.insert(0, "dataset", dataset); agg.insert(1, "dl_model", model); agg.insert(2, "fs_method", fs)
        trajectory_rows.extend(agg.to_dict("records"))

        best_k, selected_k, best_mean, band = choose_one_se_k(agg)
        best = agg[agg.top_k_num == best_k].iloc[0]
        chosen = agg[agg.top_k_num == selected_k].iloc[0]
        summary_rows.append({
            "dataset": dataset, "path": "model_aware", "dl_model": model, "fs_method": fs,
            "best_k": best_k, "one_se_k": selected_k,
            "best_f1_mean": best_mean, "best_f1_se": float(best.f1_se if not pd.isna(best.f1_se) else 0.0),
            "one_se_threshold": band, "selected_f1_mean": float(chosen.f1_mean),
            "selected_f1_std": float(chosen.f1_std if not pd.isna(chosen.f1_std) else 0.0),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "refinement_summary.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(out / "refinement_trajectories.csv", index=False)
    log(f"REFINE DONE | {dataset}: {len(summary)} retained FS-model branches")
    return summary


def run_all_refinements(cfg: Dict[str, Any], datasets: List[str] | None = None, resume=True, overwrite_stale=False):
    datasets = datasets or list(cfg["datasets"])
    return [run_refinement(cfg, d, resume, overwrite_stale) for d in datasets]
