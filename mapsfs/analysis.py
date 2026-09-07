from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import output_dir
from .feature_ranking import load_ranking
from .preprocessing import prepared_dir
from .utils import json_load, log


def data_integrity_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ds in cfg["datasets"]:
        for seed in cfg["experiment"]["seeds"]:
            m = json_load(prepared_dir(cfg, ds, int(seed)) / "manifest.json")
            rows.append({
                "dataset": ds, "seed": int(seed), "n_predictors": m["n_predictors"],
                "schema_hash": m["schema_hash"], "n_fit": m["n_fit"], "n_validation": m["n_validation"], "n_test": m["n_test"],
                "positive_rate_fit": m["positive_rate_fit"], "positive_rate_validation": m["positive_rate_validation"], "positive_rate_test": m["positive_rate_test"],
            })
    return pd.DataFrame(rows)


def feature_stability_tables(cfg: Dict[str, Any], top_n_values=(10, 20)):
    pair_rows, summary_rows, freq_rows = [], [], []
    seeds = [int(s) for s in cfg["experiment"]["seeds"]]
    for ds in cfg["datasets"]:
        for fs in cfg["feature_selection"]["enabled"]:
            ranks = {s: load_ranking(cfg, ds, s, fs) for s in seeds}
            names = ranks[seeds[0]]["feature_name"].astype(str).tolist()
            pos = {s: {n: int(r) for n, r in zip(ranks[s].feature_name.astype(str), ranks[s].rank)} for s in seeds}
            for i, a in enumerate(seeds):
                for b in seeds[i+1:]:
                    shared = sorted(set(pos[a]) & set(pos[b]))
                    rho = spearmanr([pos[a][n] for n in shared], [pos[b][n] for n in shared]).statistic if len(shared) > 1 else np.nan
                    row = {"dataset": ds, "fs_method": fs, "seed_a": a, "seed_b": b, "spearman_rho": rho}
                    for topn in top_n_values:
                        A = set(ranks[a].head(topn).feature_name.astype(str)); B = set(ranks[b].head(topn).feature_name.astype(str))
                        row[f"top{topn}_jaccard"] = len(A & B) / len(A | B) if (A | B) else np.nan
                    pair_rows.append(row)
            part = pd.DataFrame([r for r in pair_rows if r["dataset"] == ds and r["fs_method"] == fs])
            srow = {"dataset": ds, "fs_method": fs, "mean_spearman": float(part.spearman_rho.mean()), "std_spearman": float(part.spearman_rho.std(ddof=1))}
            for topn in top_n_values:
                srow[f"mean_top{topn}_jaccard"] = float(part[f"top{topn}_jaccard"].mean())
            summary_rows.append(srow)
            for topn in top_n_values:
                counts = {}
                for s in seeds:
                    for name in ranks[s].head(topn).feature_name.astype(str):
                        counts[name] = counts.get(name, 0) + 1
                for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                    freq_rows.append({"dataset": ds, "fs_method": fs, "top_n": topn, "feature_name": name, "selection_count": count, "total_seed_runs": len(seeds), "selection_frequency": count / len(seeds)})
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows), pd.DataFrame(freq_rows)


def build_results_package(cfg: Dict[str, Any]) -> Path:
    root = output_dir(cfg) / "09_results_package"
    tables = root / "tables"; figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    provenance = output_dir(cfg) / "00_provenance" / "protocol_manifest.json"
    if provenance.exists():
        import shutil
        shutil.copy2(provenance, root / "protocol_manifest.json")

    integrity = data_integrity_table(cfg); integrity.to_csv(tables / "R1_data_integrity.csv", index=False)
    pair, stab, freq = feature_stability_tables(cfg)
    pair.to_csv(tables / "R2_feature_stability_pairwise.csv", index=False)
    stab.to_csv(tables / "R2_feature_stability_summary.csv", index=False)
    freq.to_csv(tables / "R2_stable_feature_frequency.csv", index=False)

    gate_rows = []; refine_rows = []; eff_rows = []; base_rows = []; baseline_abs_rows = []; oracle_rows = []; perf_rows = []
    for ds in cfg["datasets"]:
        gate_rows.append(pd.read_csv(output_dir(cfg) / "05_gate" / ds / "gate_decision.csv"))
        refine_rows.append(pd.read_csv(output_dir(cfg) / "06_refinement" / ds / "refinement_summary.csv"))
        eff_rows.append(pd.read_csv(output_dir(cfg) / "07_efficiency" / ds / "refined_candidate_metrics.csv"))
        base_rows.append(pd.read_csv(output_dir(cfg) / "07_efficiency" / ds / "matched_baseline_comparisons.csv"))
        baseline_abs_rows.append(pd.read_csv(output_dir(cfg) / "07_efficiency" / ds / "baseline_metrics.csv"))
        oracle_summary = output_dir(cfg) / "08_oracle" / ds / "oracle_validation_summary.csv"
        if oracle_summary.exists():
            oracle_rows.append(pd.read_csv(oracle_summary))
            p = output_dir(cfg) / "08_oracle" / ds / "performance_differences.csv"
            if p.exists():
                tmp = pd.read_csv(p); tmp.insert(0, "dataset", ds); perf_rows.append(tmp)

    gates = pd.concat(gate_rows, ignore_index=True); gates.to_csv(tables / "R3_dependency_gate_results.csv", index=False)
    refine = pd.concat(refine_rows, ignore_index=True); refine.to_csv(tables / "R4_one_se_refinement.csv", index=False)
    eff = pd.concat(eff_rows, ignore_index=True); eff.to_csv(tables / "R5_efficiency_candidates.csv", index=False)
    base = pd.concat(base_rows, ignore_index=True); base.to_csv(tables / "R6_matched_baseline_comparisons.csv", index=False)
    pd.concat(baseline_abs_rows, ignore_index=True).to_csv(tables / "R6_baseline_absolute_metrics.csv", index=False)
    if oracle_rows:
        oracle = pd.concat(oracle_rows, ignore_index=True); oracle.to_csv(tables / "R7_oracle_search_validation.csv", index=False)
    else:
        oracle = pd.DataFrame()
    if perf_rows:
        pd.concat(perf_rows, ignore_index=True).to_csv(tables / "R8_oracle_performance_differences.csv", index=False)

    hypothesis = []
    for _, g in gates.iterrows():
        ds = g.dataset
        o = oracle[oracle.dataset == ds].iloc[0] if len(oracle) and (oracle.dataset == ds).any() else None
        hypothesis.append({
            "dataset": ds,
            "H1_gate": g.gate,
            "H1_p_interaction": g.p_interaction,
            "H1_interpretation": "model-specific refinement activated" if str(g.gate) == "YES" else "model-independent path retained",
            "H2_search_saving_pct": None if o is None else o.search_saving_pct,
            "H2_mean_f1_difference_maps_minus_oracle": None if o is None else o.mean_f1_difference_maps_minus_oracle,
            "H2_oracle_pareto_recall": None if o is None else o.oracle_pareto_recall,
            "H3_note": "Use R6 matched baseline comparisons; no automatic claim threshold is imposed.",
        })
    pd.DataFrame(hypothesis).to_csv(tables / "R9_hypothesis_evidence_summary.csv", index=False)

    runtime_rows = []
    for ds in cfg["datasets"]:
        rp = output_dir(cfg) / "10_runtime" / ds / "runtime_profile_summary.csv"
        if rp.exists():
            runtime_rows.append(pd.read_csv(rp))
    if runtime_rows:
        pd.concat(runtime_rows, ignore_index=True).to_csv(tables / "R10_runtime_profile_summary.csv", index=False)

    _make_figures(cfg, gates, refine, eff, stab, figures)
    _write_summary_md(cfg, gates, refine, eff, base, oracle, root / "RESULTS_SUMMARY.md")
    log(f"RESULTS PACKAGE | {root}")
    return root


def _make_figures(cfg, gates, refine, eff, stab, figures: Path):
    import matplotlib.pyplot as plt

    if len(stab):
        for ds, part in stab.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(part.fs_method, part.mean_spearman)
            ax.set_ylim(0, 1.05); ax.set_ylabel("Mean pairwise Spearman rho"); ax.set_xlabel("Feature selector")
            ax.set_title(f"Feature-ranking stability: {ds}"); fig.tight_layout()
            fig.savefig(figures / f"F1_feature_stability_{ds}.png", dpi=300); plt.close(fig)

    for ds in cfg["datasets"]:
        traj_path = output_dir(cfg) / "06_refinement" / ds / "refinement_trajectories.csv"
        if traj_path.exists():
            traj = pd.read_csv(traj_path)
            if len(traj):
                fig, ax = plt.subplots(figsize=(8, 5))
                for (model, fs), p in traj.groupby(["dl_model", "fs_method"]):
                    ax.plot(p.top_k_num, p.f1_mean, marker="o", label=f"{model}:{fs}")
                ax.set_xlabel("Retained features (k)"); ax.set_ylabel("Mean F1"); ax.set_title(f"Refined k-trajectories: {ds}"); ax.legend(fontsize=7)
                fig.tight_layout(); fig.savefig(figures / f"F2_refinement_trajectories_{ds}.png", dpi=300); plt.close(fig)
        p = eff[eff.dataset == ds]
        if len(p):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(p.inference_time_per_flow_us, p.f1, s=np.maximum(20, 300 / np.maximum(p.n_features, 1)))
            for _, r in p[p.is_pareto.astype(bool)].iterrows():
                ax.annotate(f"{r.fs_method}-{r.top_k}-{r.dl_model}", (r.inference_time_per_flow_us, r.f1), fontsize=6)
            ax.set_xlabel("Inference time (us/flow)"); ax.set_ylabel("Mean F1"); ax.set_title(f"Refined candidates and Pareto-efficient points: {ds}")
            fig.tight_layout(); fig.savefig(figures / f"F3_efficiency_tradeoff_{ds}.png", dpi=300); plt.close(fig)


def _write_summary_md(cfg, gates, refine, eff, base, oracle, path: Path):
    lines = ["# MAPS-FS Results Package", "", "This file is an analysis aid, not manuscript prose. Report uncertainty and avoid automatic causal/optimality claims.", ""]
    for ds in cfg["datasets"]:
        lines += [f"## {ds}"]
        g = gates[gates.dataset == ds].iloc[0]
        lines += [f"- Gate: **{g.gate}**; omnibus interaction p = {float(g.p_interaction):.6g}."]
        r = refine[refine.dataset == ds]
        lines += [f"- Refined/model-independent candidate rows: {len(r)}."]
        e = eff[eff.dataset == ds]
        lines += [f"- Pareto-efficient final candidates: {int(e.is_pareto.astype(bool).sum())}/{len(e)}."]
        if len(oracle) and (oracle.dataset == ds).any():
            o = oracle[oracle.dataset == ds].iloc[0]
            lines += [f"- Search saving vs exhaustive oracle: {float(o.search_saving_pct):.2f}%.", f"- Mean MAPS-minus-oracle best-F1 difference: {float(o.mean_f1_difference_maps_minus_oracle):.6f}.", f"- Oracle Pareto recall: {float(o.oracle_pareto_recall):.3f}."]
        lines += ["- Use `R6_matched_baseline_comparisons.csv` for F1/FAR/latency deltas vs full and common baselines.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
