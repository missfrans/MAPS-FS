from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import output_dir
from .pareto import pareto_mask
from .runner import run_one, fetch_metrics
from .utils import json_load, log


def oracle_design(cfg: Dict[str, Any], dataset: str) -> pd.DataFrame:
    rows = []
    for fs in cfg["feature_selection"]["enabled"]:
        for k in cfg["datasets"][dataset]["top_k_values"]:
            for model in cfg["models"]["enabled"]:
                rows.append({"dataset": dataset, "fs_method": fs, "top_k": int(k), "dl_model": model, "reference": False})
    for model in cfg["models"]["enabled"]:
        rows.append({"dataset": dataset, "fs_method": "full", "top_k": "all", "dl_model": model, "reference": True})
    return pd.DataFrame(rows)


def run_oracle(cfg: Dict[str, Any], dataset: str, resume=True, overwrite_stale=False):
    design = oracle_design(cfg, dataset)
    out = output_dir(cfg) / "08_oracle" / dataset
    out.mkdir(parents=True, exist_ok=True)
    design.to_csv(out / "oracle_design.csv", index=False)
    seeds = [int(s) for s in cfg.get("oracle", {}).get("seeds", cfg["experiment"]["seeds"])]
    for _, r in design.iterrows():
        for seed in seeds:
            run_one(cfg, dataset, seed, str(r.fs_method), str(r.top_k), str(r.dl_model), "oracle", resume, overwrite_stale)
    metrics = fetch_metrics(cfg, dataset, seeds, design[["fs_method", "top_k", "dl_model"]])
    metrics.to_csv(out / "oracle_metrics_raw.csv", index=False)
    log(f"ORACLE DONE | {dataset}: search configs={int((~design.reference).sum())} refs={int(design.reference.sum())} seeds={len(seeds)}")
    return design


def run_all_oracles(cfg: Dict[str, Any], datasets: List[str] | None = None, resume=True, overwrite_stale=False):
    datasets = datasets or list(cfg["datasets"])
    return [run_oracle(cfg, d, resume, overwrite_stale) for d in datasets]


def _practical_search_evaluations(cfg: Dict[str, Any], dataset: str) -> tuple[int, int, int]:
    """Reconstruct MAPS-FS search cost from the frozen experimental designs.

    Returns:
        (n_screening, n_additional_refinement, n_total_practical)

    Search cost counts unique (dataset, seed, FS, k, model) evaluations that belong
    to SCREEN or model-specific REFINE. Full/common baselines are references, not
    search configurations. Reconstructing from design/gate artifacts avoids relying
    on a request ledger that can be incomplete after distributed-run merges.
    """
    root = output_dir(cfg)
    key = ["dataset", "seed", "fs_method", "top_k", "dl_model"]

    screen_path = root / "04_screening" / dataset / "screen_design.csv"
    if not screen_path.exists():
        raise FileNotFoundError(
            f"Missing screening design for {dataset}: {screen_path}. "
            "Run Stage 2 dependency screening before Oracle validation."
        )
    screen_design = pd.read_csv(screen_path)
    required_screen = {"dataset", "fs_method", "top_k", "dl_model"}
    missing = required_screen.difference(screen_design.columns)
    if missing:
        raise ValueError(f"{screen_path} is missing columns: {sorted(missing)}")

    screen_seeds = [
        int(s)
        for s in cfg.get("screening", {}).get(
            "pilot_seeds", cfg["experiment"]["seeds"]
        )
    ]
    screen_rows = []
    for _, row in screen_design.iterrows():
        for seed in screen_seeds:
            screen_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fs_method": str(row.fs_method),
                    "top_k": int(row.top_k),
                    "dl_model": str(row.dl_model),
                }
            )
    screen_requests = pd.DataFrame(screen_rows, columns=key).drop_duplicates()

    gate_path = root / "05_gate" / dataset / "gate_decision.json"
    if not gate_path.exists():
        raise FileNotFoundError(
            f"Missing Gate decision for {dataset}: {gate_path}. "
            "Run Stage 3 MAPS Gate before Oracle validation."
        )
    gate = json_load(gate_path)

    refine_rows = []
    if str(gate.get("gate", "")).upper() == "YES":
        branches_path = root / "05_gate" / dataset / "prioritized_branches.csv"
        if not branches_path.exists():
            raise FileNotFoundError(
                f"Gate=YES but prioritized branches are missing for {dataset}: "
                f"{branches_path}"
            )
        branches = pd.read_csv(branches_path)
        required_branches = {"fs_method", "dl_model"}
        missing = required_branches.difference(branches.columns)
        if missing:
            raise ValueError(f"{branches_path} is missing columns: {sorted(missing)}")

        refine_seeds = [
            int(s)
            for s in cfg.get("refinement", {}).get(
                "seeds", cfg["experiment"]["seeds"]
            )
        ]
        top_k_values = [int(k) for k in cfg["datasets"][dataset]["top_k_values"]]
        for _, branch in branches.iterrows():
            for k in top_k_values:
                for seed in refine_seeds:
                    refine_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "fs_method": str(branch.fs_method),
                            "top_k": k,
                            "dl_model": str(branch.dl_model),
                        }
                    )

    refine_requests = pd.DataFrame(refine_rows, columns=key).drop_duplicates()
    if len(refine_requests):
        union = pd.concat([screen_requests, refine_requests], ignore_index=True).drop_duplicates()
        additional_refinement = int(len(union) - len(screen_requests))
    else:
        union = screen_requests
        additional_refinement = 0

    return int(len(screen_requests)), additional_refinement, int(len(union))


def validate_against_oracle(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    out_oracle = output_dir(cfg) / "08_oracle" / dataset
    design = pd.read_csv(out_oracle / "oracle_design.csv")
    raw = pd.read_csv(out_oracle / "oracle_metrics_raw.csv")
    raw["top_k"] = raw["top_k"].astype(str)
    search_design = design[~design.reference.astype(bool)].copy()
    n_seeds = raw.seed.nunique()
    n_oracle = int(len(search_design) * n_seeds)

    n_screen, n_refine_additional, n_maps = _practical_search_evaluations(cfg, dataset)
    saving = 100.0 * (1.0 - n_maps / n_oracle) if n_oracle else np.nan
    remaining_after_screen = max(n_oracle - n_screen, 0)
    post_gate_saving = (
        100.0 * (1.0 - n_refine_additional / remaining_after_screen)
        if remaining_after_screen
        else np.nan
    )

    agg = raw[raw.fs_method.isin(cfg["feature_selection"]["enabled"])].groupby(["fs_method", "top_k", "dl_model"]).agg(
        f1=("f1", "mean"), far=("far", "mean"), inference_time_per_flow_us=("inference_time_per_flow_us", "mean"), n_features=("n_features", "mean")
    ).reset_index()
    objectives = cfg.get("efficiency_selection", {}).get("objectives", {"f1":"max","far":"min","inference_time_per_flow_us":"min","n_features":"min"})
    agg["is_oracle_pareto"] = pareto_mask(agg, objectives)
    agg.to_csv(out_oracle / "oracle_aggregated_and_pareto.csv", index=False)

    final = pd.read_csv(output_dir(cfg) / "07_efficiency" / dataset / "refined_candidate_metrics.csv")
    final["top_k"] = final["top_k"].astype(str)
    key = ["fs_method", "top_k", "dl_model"]
    merged = final.merge(agg[key + ["is_oracle_pareto"]], on=key, how="left")
    maps_pareto = merged[merged.is_pareto.astype(bool)]
    recovered = int(maps_pareto.is_oracle_pareto.fillna(False).sum())
    oracle_pareto_n = int(agg.is_oracle_pareto.sum())
    maps_pareto_n = int(len(maps_pareto))

    perf_rows = []
    for model in cfg["models"]["enabled"]:
        o = agg[agg.dl_model == model]
        m = final[final.dl_model == model]
        if len(o) and len(m):
            perf_rows.append({"dl_model": model, "oracle_best_f1": float(o.f1.max()), "maps_best_f1": float(m.f1.max()), "f1_difference_maps_minus_oracle": float(m.f1.max() - o.f1.max())})
    pd.DataFrame(perf_rows).to_csv(out_oracle / "performance_differences.csv", index=False)

    summary = {
        "dataset": dataset, "n_oracle_search_evaluations": n_oracle,
        "n_maps_screening_evaluations": n_screen,
        "n_maps_additional_refinement_evaluations": n_refine_additional,
        "n_maps_practical_search_evaluations": n_maps,
        # Backward-compatible field: now explicitly means end-to-end MAPS-FS search saving.
        "search_saving_pct": saving,
        "overall_search_saving_pct": saving,
        "post_gate_refinement_saving_pct": post_gate_saving,
        "oracle_pareto_configurations": oracle_pareto_n, "maps_final_pareto_configurations": maps_pareto_n,
        "recovered_oracle_pareto_configurations": recovered,
        "oracle_pareto_recall": recovered / oracle_pareto_n if oracle_pareto_n else np.nan,
        "maps_pareto_precision": recovered / maps_pareto_n if maps_pareto_n else np.nan,
        "mean_f1_difference_maps_minus_oracle": float(pd.DataFrame(perf_rows).f1_difference_maps_minus_oracle.mean()) if perf_rows else np.nan,
    }
    pd.DataFrame([summary]).to_csv(out_oracle / "oracle_validation_summary.csv", index=False)
    merged.to_csv(out_oracle / "maps_candidates_oracle_membership.csv", index=False)
    log(
        f"ORACLE VALIDATION | {dataset}: screen={n_screen} "
        f"additional_refine={n_refine_additional} total_maps={n_maps} "
        f"saving={saving:.2f}% post_gate_saving={post_gate_saving:.2f}% "
        f"mean ΔF1={summary['mean_f1_difference_maps_minus_oracle']:.6f}"
    )
    return summary
