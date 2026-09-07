from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import output_dir
from .utils import json_dump, json_load, log


def _fmt(x, digits: int = 6) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def _candidate_key(row: pd.Series) -> str:
    return f"{row.fs_method}-k{int(float(row.top_k))}-{row.dl_model}"


def _use_case_recommendations(pareto: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return objective-specific representatives without inventing a weighted winner."""
    if pareto.empty:
        return {}
    specs = {
        "highest_detection_f1": ("f1", False),
        "lowest_false_alarm_rate": ("far", True),
        "lowest_inference_latency": ("inference_time_per_flow_us", True),
        "smallest_retained_representation": ("n_features", True),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for label, (metric, ascending) in specs.items():
        if metric not in pareto.columns or pareto[metric].dropna().empty:
            continue
        r = pareto.sort_values([metric, "f1"], ascending=[ascending, False]).iloc[0]
        out[label] = {
            "configuration": _candidate_key(r),
            "fs_method": str(r.fs_method),
            "top_k": int(float(r.top_k)),
            "dl_model": str(r.dl_model),
            "f1": float(r.f1) if "f1" in r and pd.notna(r.f1) else None,
            "far": float(r.far) if "far" in r and pd.notna(r.far) else None,
            "inference_time_per_flow_us": float(r.inference_time_per_flow_us) if "inference_time_per_flow_us" in r and pd.notna(r.inference_time_per_flow_us) else None,
            "n_features": float(r.n_features) if "n_features" in r and pd.notna(r.n_features) else None,
        }
    return out


def _matched_vs_full(comparisons: pd.DataFrame, fs: str, k: int, model: str) -> Dict[str, Any]:
    if comparisons.empty:
        return {}
    p = comparisons[
        (comparisons.fs_method.astype(str) == str(fs))
        & (pd.to_numeric(comparisons.top_k, errors="coerce") == int(k))
        & (comparisons.dl_model.astype(str) == str(model))
        & (comparisons.baseline.astype(str) == "full")
    ]
    if p.empty:
        return {}
    r = p.iloc[0]
    keys = [
        "delta_f1_mean", "delta_f1_ci95_low", "delta_f1_ci95_high", "delta_f1_p_adjusted_holm",
        "delta_far_mean", "delta_far_ci95_low", "delta_far_ci95_high", "delta_far_p_adjusted_holm",
        "delta_inference_time_per_flow_us_mean", "delta_inference_time_per_flow_us_ci95_low",
        "delta_inference_time_per_flow_us_ci95_high", "delta_inference_time_per_flow_us_p_adjusted_holm",
        "feature_reduction_pct",
    ]
    result = {}
    for key in keys:
        if key in r.index and pd.notna(r[key]):
            result[key] = float(r[key])
    return result


def _gate_explanation(gate: Dict[str, Any]) -> tuple[str, str]:
    p = float(gate["p_interaction"])
    alpha = float(gate["alpha"])
    models = list(gate.get("decision_relevant_models", []))
    if gate["gate"] == "YES":
        why = (
            f"The model-related interaction block was statistically supported (p={p:.6g} < alpha={alpha:g}) "
            f"and the common representation was outside the multiplicity-adjusted comparable set for at least one model "
            f"({', '.join(models)})."
        )
        return "MODEL-SPECIFIC REFINEMENT ACTIVATED", why
    if p >= alpha:
        why = (
            f"The model-related interaction block was not statistically supported at the prespecified level "
            f"(p={p:.6g} >= alpha={alpha:g}). The framework therefore retained the common model-independent path."
        )
        return "MODEL-SPECIFIC REFINEMENT NOT ACTIVATED", why
    why = (
        f"The interaction block was statistically supported (p={p:.6g} < alpha={alpha:g}), but the common representation "
        "remained within every model's multiplicity-adjusted comparable set. The interaction did not change the representation decision, so further model-specific refinement was not activated."
    )
    return "MODEL-SPECIFIC REFINEMENT NOT ACTIVATED", why


def build_dataset_explanation(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    root = output_dir(cfg)
    gate = json_load(root / "05_gate" / dataset / "gate_decision.json")
    common = pd.read_csv(root / "05_gate" / dataset / "common_decision.csv").iloc[0]
    branches_path = root / "05_gate" / dataset / "prioritized_branches.csv"
    branches = pd.read_csv(branches_path) if branches_path.exists() else pd.DataFrame()
    refine = pd.read_csv(root / "06_refinement" / dataset / "refinement_summary.csv")
    candidates = pd.read_csv(root / "07_efficiency" / dataset / "refined_candidate_metrics.csv")
    comparisons_path = root / "07_efficiency" / dataset / "matched_baseline_comparisons.csv"
    comparisons = pd.read_csv(comparisons_path) if comparisons_path.exists() else pd.DataFrame()
    pareto = candidates[candidates.is_pareto.astype(bool)].copy() if len(candidates) else pd.DataFrame()
    decision_title, why = _gate_explanation(gate)

    use_cases = _use_case_recommendations(pareto)
    for item in use_cases.values():
        item["comparison_vs_full"] = _matched_vs_full(
            comparisons, item["fs_method"], item["top_k"], item["dl_model"]
        )

    refinement_rows = []
    for _, r in refine.iterrows():
        refinement_rows.append({
            "path": str(r.path), "fs_method": str(r.fs_method), "dl_model": str(r.dl_model),
            "best_k": int(r.best_k), "selected_one_se_k": int(r.one_se_k),
            "best_f1_mean": float(r.best_f1_mean), "selected_f1_mean": float(r.selected_f1_mean),
            "selected_f1_std": float(r.selected_f1_std) if "selected_f1_std" in r and pd.notna(r.selected_f1_std) else None,
        })

    oracle_summary_path = root / "08_oracle" / dataset / "oracle_validation_summary.csv"
    oracle = None
    if oracle_summary_path.exists():
        r = pd.read_csv(oracle_summary_path).iloc[0]
        oracle = {k: (None if pd.isna(v) else (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)) for k, v in r.items()}

    runtime_summary_path = root / "10_runtime" / dataset / "runtime_profile_summary.csv"
    runtime = []
    if runtime_summary_path.exists():
        runtime_df = pd.read_csv(runtime_summary_path)
        runtime = runtime_df.to_dict("records")

    if gate["gate"] == "YES":
        next_step = (
            "Use the branch-specific one-SE representations produced by MAPS-FS rather than forcing one common feature count. "
            "Choose an operating point from the Pareto-efficient set according to the deployment priority (detection, false alarms, latency, or compactness). "
            "If hardware changes, repeat runtime profiling; rerun the search when the data distribution, candidate FS methods, model set, or training protocol changes."
        )
    else:
        next_step = (
            f"Retain the common representation {common.common_fs_method}-k{int(common.common_top_k)} across the evaluated downstream models; "
            "the current evidence does not justify paying for model-specific feature refinement. Choose the downstream model from the Pareto-efficient "
            "operating points according to the deployment priority. Repeat dependency screening if the data distribution, candidate FS methods, model set, or training protocol changes."
        )

    return {
        "dataset": dataset,
        "decision": decision_title,
        "gate": gate["gate"],
        "p_interaction": float(gate["p_interaction"]),
        "alpha": float(gate["alpha"]),
        "why": why,
        "common_representation": {
            "fs_method": str(common.common_fs_method),
            "top_k": int(common.common_top_k),
            "mean_screening_response": float(common.common_mean_response),
        },
        "decision_relevant_models": list(gate.get("decision_relevant_models", [])),
        "prioritized_branches": branches.to_dict("records") if len(branches) else [],
        "refinement_results": refinement_rows,
        "n_pareto_efficient_candidates": int(len(pareto)),
        "pareto_efficient_candidates": pareto.to_dict("records") if len(pareto) else [],
        "use_case_recommendations": use_cases,
        "oracle_validation": oracle,
        "runtime_profile": runtime,
        "next_step": next_step,
        "interpretation_note": (
            "MAPS-FS returns an efficient set rather than a universal winner. Objective-specific representatives are reported without an arbitrary weighted score. "
            "A non-significant contrast is not interpreted as formal equivalence, and Pareto membership alone is not treated as proof of real-world deployment benefit."
        ),
    }


def _markdown_dataset(report: Dict[str, Any]) -> str:
    c = report["common_representation"]
    lines = [
        f"# MAPS-FS self-explanation — {report['dataset']}", "",
        f"## Final framework decision", f"**{report['decision']}**", "", report["why"], "",
        f"Common screened representation: **{c['fs_method']}-k{c['top_k']}**.", "",
    ]
    if report["gate"] == "YES":
        models = ", ".join(report["decision_relevant_models"]) or "none listed"
        lines += [
            "## What refinement produced",
            f"Decision-relevant model(s): **{models}**.",
            f"Prioritized FS-model branches: **{len(report['prioritized_branches'])}**.", "",
            "| Model | FS | Best k | One-SE selected k | Selected mean F1 |",
            "|---|---|---:|---:|---:|",
        ]
        for r in report["refinement_results"]:
            lines.append(f"| {r['dl_model']} | {r['fs_method']} | {r['best_k']} | {r['selected_one_se_k']} | {r['selected_f1_mean']:.6f} |")
        lines.append("")
    else:
        lines += [
            "## What the NO decision means",
            "The framework deliberately skipped model-specific feature-count expansion. This is not a failed run; it is the intended MAPS-FS outcome when additional refinement is not justified by the prespecified gate.", "",
        ]

    lines += ["## Pareto-efficient recommendations", f"MAPS-FS retained **{report['n_pareto_efficient_candidates']}** Pareto-efficient final candidate(s).", ""]
    if report["use_case_recommendations"]:
        lines += ["| Use case | Configuration | F1 | FAR | Latency (us/flow) | Features |", "|---|---|---:|---:|---:|---:|"]
        for label, r in report["use_case_recommendations"].items():
            lines.append(
                f"| {label.replace('_', ' ')} | {r['configuration']} | {_fmt(r.get('f1'))} | {_fmt(r.get('far'))} | {_fmt(r.get('inference_time_per_flow_us'))} | {_fmt(r.get('n_features'), 4)} |"
            )
        lines.append("")
        lines += [
            "These rows are objective-specific representatives, not a weighted overall ranking. If one configuration appears in several rows, it satisfies several deployment priorities simultaneously.", "",
        ]

    if report["oracle_validation"] is not None:
        o = report["oracle_validation"]
        lines += [
            "## Validation against exhaustive oracle",
            f"- Practical MAPS-FS search evaluations: **{int(o['n_maps_practical_search_evaluations'])}**.",
            f"- Exhaustive search evaluations: **{int(o['n_oracle_search_evaluations'])}**.",
            f"- Search saving: **{float(o['search_saving_pct']):.2f}%**.",
            f"- Mean MAPS-minus-oracle best-F1 difference: **{float(o['mean_f1_difference_maps_minus_oracle']):.6f}**.",
            f"- Oracle Pareto recall: **{float(o['oracle_pareto_recall']):.3f}**.",
            f"- MAPS Pareto precision: **{float(o['maps_pareto_precision']):.3f}**.", "",
            "No automatic success/failure threshold is imposed on these oracle metrics; report the values and interpret them in the experimental context.", "",
        ]

    if report["runtime_profile"]:
        lines += ["## Repeated runtime profile", "Repeated latency profiling is available under `outputs/10_runtime/`; use it for manuscript-quality latency reporting instead of relying only on a single pairing timing.", ""]

    lines += ["## Recommended next step", report["next_step"], "", "## Interpretation guardrail", report["interpretation_note"], ""]
    return "\n".join(lines)


def build_self_explanation(cfg: Dict[str, Any], datasets: Iterable[str] | None = None) -> Path:
    datasets = list(datasets or cfg["datasets"].keys())
    root = output_dir(cfg) / "09_results_package" / "self_explanation"
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for ds in datasets:
        report = build_dataset_explanation(cfg, ds)
        reports.append(report)
        json_dump(root / f"{ds}_FINAL_RECOMMENDATION.json", report)
        (root / f"{ds}_FINAL_RECOMMENDATION.md").write_text(_markdown_dataset(report), encoding="utf-8")

    lines = [
        "# MAPS-FS final self-explanation", "",
        "This report is generated automatically from the prespecified MAPS-FS decision path. It is intended as a factual interpretation aid for users and as an evidence map for manuscript Results/Discussion; it is not a substitute for domain-specific deployment validation.", "",
    ]
    for report in reports:
        c = report["common_representation"]
        lines += [
            f"## {report['dataset']}",
            f"- Decision: **{report['decision']}**.",
            f"- Common representation: **{c['fs_method']}-k{c['top_k']}**.",
            f"- Pareto-efficient final candidates: **{report['n_pareto_efficient_candidates']}**.",
            f"- Next step: {report['next_step']}", "",
        ]
    (root / "FINAL_SELF_EXPLANATION.md").write_text("\n".join(lines), encoding="utf-8")
    json_dump(root / "FINAL_SELF_EXPLANATION.json", {"datasets": reports})
    log(f"SELF-EXPLANATION | {root}")
    return root
