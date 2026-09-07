from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import t

from .config import output_dir
from .runner import fetch_metrics
from .utils import log


def pareto_mask(df: pd.DataFrame, objectives: Dict[str, str]) -> np.ndarray:
    values = df[list(objectives)].astype(float).to_numpy()
    transformed = values.copy()
    for j, col in enumerate(objectives):
        if objectives[col] == "max":
            transformed[:, j] *= -1.0
        elif objectives[col] != "min":
            raise ValueError(f"Objective direction for {col} must be min or max")
    n = len(df)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = np.all(transformed <= transformed[i], axis=1) & np.any(transformed < transformed[i], axis=1)
        if dominates_i.any():
            efficient[i] = False
    return efficient


def _candidate_metrics(cfg: Dict[str, Any], dataset: str, summary: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        design = pd.DataFrame([{"fs_method": r.fs_method, "top_k": int(r.one_se_k), "dl_model": r.dl_model}])
        m = fetch_metrics(cfg, dataset, seeds, design)
        row = {
            "dataset": dataset, "fs_method": r.fs_method, "top_k": int(r.one_se_k), "dl_model": r.dl_model,
            "n_features": float(m.n_features.mean()), "n_seeds": int(m.seed.nunique()),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "mcc", "far", "inference_time_per_flow_us", "throughput_flows_per_sec", "training_time_sec"]:
            if metric in m.columns:
                row[metric] = float(m[metric].mean())
                row[f"{metric}_std"] = float(m[metric].std(ddof=1)) if len(m) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_summary(candidate: pd.DataFrame, baseline: pd.DataFrame, metric: str, direction: str):
    from scipy.stats import ttest_rel
    c = candidate.set_index("seed")[metric]
    b = baseline.set_index("seed")[metric]
    idx = c.index.intersection(b.index)
    cv = c.loc[idx].astype(float).to_numpy(); bv = b.loc[idx].astype(float).to_numpy()
    diff = cv - bv
    n = len(diff)
    mean = float(np.mean(diff)) if n else np.nan
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else np.nan
    crit = float(t.ppf(0.975, n - 1)) if n > 1 else 0.0
    p_value = float(ttest_rel(cv, bv, nan_policy="omit").pvalue) if n > 1 and not np.allclose(diff, 0) else 1.0
    cohen_dz = mean / sd if n > 1 and sd > 0 else 0.0
    return mean, sd, mean - crit * se if n else np.nan, mean + crit * se if n else np.nan, p_value, cohen_dz


def run_efficiency_selection(cfg: Dict[str, Any], dataset: str) -> pd.DataFrame:
    ref_path = output_dir(cfg) / "06_refinement" / dataset / "refinement_summary.csv"
    summary = pd.read_csv(ref_path)
    seeds = [int(s) for s in cfg.get("refinement", {}).get("seeds", cfg["experiment"]["seeds"])]
    candidates = _candidate_metrics(cfg, dataset, summary, seeds)
    objectives = cfg.get("efficiency_selection", {}).get("objectives", {
        "f1": "max", "far": "min", "inference_time_per_flow_us": "min", "n_features": "min"
    })
    candidates["is_pareto"] = pareto_mask(candidates, objectives)
    out = output_dir(cfg) / "07_efficiency" / dataset
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "refined_candidate_metrics.csv", index=False)
    candidates[candidates.is_pareto].to_csv(out / "pareto_efficient_configurations.csv", index=False)

    common = pd.read_csv(output_dir(cfg) / "05_gate" / dataset / "common_decision.csv").iloc[0]
    baseline_rows = []
    for model in cfg["models"]["enabled"]:
        for bname, bfs, bk in [("full", "full", "all"), ("common", str(common.common_fs_method), int(common.common_top_k))]:
            bm = fetch_metrics(cfg, dataset, seeds, pd.DataFrame([{"fs_method": bfs, "top_k": bk, "dl_model": model}]))
            brow = {"dataset": dataset, "baseline": bname, "fs_method": bfs, "top_k": str(bk), "dl_model": model, "n_seeds": int(bm.seed.nunique())}
            for metric in ["accuracy","precision","recall","f1","roc_auc","average_precision","mcc","far","inference_time_per_flow_us","throughput_flows_per_sec","training_time_sec","n_features"]:
                if metric in bm.columns:
                    brow[metric] = float(bm[metric].mean()); brow[f"{metric}_std"] = float(bm[metric].std(ddof=1)) if len(bm)>1 else 0.0
            baseline_rows.append(brow)
    pd.DataFrame(baseline_rows).to_csv(out / "baseline_metrics.csv", index=False)
    comp_rows = []
    for _, cand in candidates.iterrows():
        design_c = pd.DataFrame([{"fs_method": cand.fs_method, "top_k": int(cand.top_k), "dl_model": cand.dl_model}])
        c = fetch_metrics(cfg, dataset, seeds, design_c)
        baselines = {
            "full": fetch_metrics(cfg, dataset, seeds, pd.DataFrame([{"fs_method": "full", "top_k": "all", "dl_model": cand.dl_model}])),
            "common": fetch_metrics(cfg, dataset, seeds, pd.DataFrame([{"fs_method": common.common_fs_method, "top_k": int(common.common_top_k), "dl_model": cand.dl_model}])),
        }
        for bname, b in baselines.items():
            row = {"dataset": dataset, "fs_method": cand.fs_method, "top_k": int(cand.top_k), "dl_model": cand.dl_model, "baseline": bname}
            for metric in ["f1", "far", "inference_time_per_flow_us"]:
                mean, sd, lo, hi, p_value, dz = _paired_summary(c, b, metric, "max" if metric == "f1" else "min")
                row[f"delta_{metric}_mean"] = mean; row[f"delta_{metric}_sd"] = sd
                row[f"delta_{metric}_ci95_low"] = lo; row[f"delta_{metric}_ci95_high"] = hi
                row[f"delta_{metric}_p_value"] = p_value; row[f"delta_{metric}_cohen_dz"] = dz
            row["feature_reduction_pct"] = float(100 * (b.n_features.mean() - c.n_features.mean()) / b.n_features.mean()) if b.n_features.mean() else np.nan
            comp_rows.append(row)
    comparisons = pd.DataFrame(comp_rows)
    if len(comparisons):
        from statsmodels.stats.multitest import multipletests
        for metric in ["f1", "far", "inference_time_per_flow_us"]:
            pcol = f"delta_{metric}_p_value"; acol = f"delta_{metric}_p_adjusted_holm"
            comparisons[acol] = np.nan
            for baseline_name, idx in comparisons.groupby("baseline").groups.items():
                vals = comparisons.loc[idx, pcol].astype(float).fillna(1.0).values
                comparisons.loc[idx, acol] = multipletests(vals, alpha=float(cfg.get("statistics", {}).get("alpha", 0.05)), method="holm")[1]
    comparisons.to_csv(out / "matched_baseline_comparisons.csv", index=False)
    log(f"EFFICIENCY | {dataset}: {int(candidates.is_pareto.sum())}/{len(candidates)} refined candidates Pareto-efficient")
    return candidates


def run_all_efficiency(cfg: Dict[str, Any], datasets: List[str] | None = None):
    datasets = datasets or list(cfg["datasets"])
    return [run_efficiency_selection(cfg, d) for d in datasets]
