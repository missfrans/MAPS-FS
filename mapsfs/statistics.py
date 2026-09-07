from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from .config import output_dir
from .runner import fetch_metrics
from .utils import json_dump, log


def interaction_block_test(
    df: pd.DataFrame,
    response: str,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Test the predefined model-related interaction block.

    Final MAPS-FS Gate protocol:

    reduced:
        response ~ C(fs_method) + k_scaled + C(dl_model) + C(seed)

    full:
        reduced + C(fs_method):C(dl_model) + k_scaled:C(dl_model)

    The five predefined experimental seeds are treated as fixed blocking
    effects. The nested OLS models are compared with a partial F-test.

    MixedLM was removed as the primary Gate analysis after audit runs showed
    singular random-effect covariance and non-finite likelihoods when the seed
    random-effect variance approached zero. The fixed-seed blocking model is
    deterministic, finite, and directly matches the repeated experimental
    protocol used by MAPS-FS.
    """
    required = {"fs_method", "top_k", "dl_model", "seed", response}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for Gate analysis: {missing}")

    work = df.copy()
    work[response] = pd.to_numeric(work[response], errors="raise").astype(float)
    if not np.isfinite(work[response].to_numpy()).all():
        raise ValueError(f"Non-finite values found in screening response '{response}'.")

    work["top_k_num"] = pd.to_numeric(work["top_k"], errors="raise").astype(float)
    std = float(work["top_k_num"].std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    work["k_scaled"] = (
        work["top_k_num"] - float(work["top_k_num"].mean())
    ) / std

    reduced_formula = (
        f"{response} ~ C(fs_method) + k_scaled + C(dl_model) + C(seed)"
    )
    full_formula = (
        reduced_formula
        + " + C(fs_method):C(dl_model) + k_scaled:C(dl_model)"
    )

    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    reduced = smf.ols(reduced_formula, data=work).fit()
    full = smf.ols(full_formula, data=work).fit()
    comparison = anova_lm(reduced, full)
    row = comparison.iloc[-1]

    f_stat = float(row["F"])
    p = float(row["Pr(>F)"])
    df_diff = int(row["df_diff"])

    if df_diff <= 0:
        raise RuntimeError(
            f"Invalid nested-model degrees-of-freedom difference: {df_diff}."
        )
    if not np.isfinite(f_stat):
        raise RuntimeError("Non-finite partial F statistic in Gate interaction test.")
    if not np.isfinite(p):
        raise RuntimeError("Non-finite interaction p-value in Gate interaction test.")
    if not np.isfinite(float(reduced.llf)) or not np.isfinite(float(full.llf)):
        raise RuntimeError("Non-finite OLS log-likelihood in Gate interaction test.")

    result = {
        "analysis_model": "OLS with seed fixed blocking effects",
        "test_type": "nested-model partial F test",
        "screening_response": response,
        "seed_treatment": "fixed blocking effect",
        "reduced_formula": reduced_formula,
        "full_formula": full_formula,
        "reduced_llf": float(reduced.llf),
        "full_llf": float(full.llf),
        "lr_statistic": np.nan,
        "f_statistic": f_stat,
        "df": df_diff,
        "p_interaction": p,
    }
    return result, full, reduced


def mixed_interaction_test(
    df: pd.DataFrame,
    response: str,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Backward-compatible alias for MAPS-FS <=1.2 callers."""
    return interaction_block_test(df, response)


def _paired_p(x, y, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2 or np.allclose(x, y):
        return 1.0

    if method == "wilcoxon":
        try:
            p = float(wilcoxon(x, y, alternative="two-sided").pvalue)
        except Exception:
            return 1.0
    else:
        p = float(ttest_rel(x, y, nan_policy="omit").pvalue)

    return p if np.isfinite(p) else 1.0


def comparable_sets(
    screen_metrics: pd.DataFrame,
    alpha: float,
    method: str,
    response: str = "f1",
) -> tuple[pd.DataFrame, Dict[str, set]]:
    """Build Holm-adjusted within-model comparable representation sets.

    Each screened representation is compared with the screened best within the
    same downstream model using paired seed-level contrasts. The response is
    configurable so Gate sensitivity analyses remain internally consistent.
    """
    required = {"fs_method", "top_k", "dl_model", "seed", response}
    missing = sorted(required.difference(screen_metrics.columns))
    if missing:
        raise ValueError(f"Missing columns required for comparable sets: {missing}")

    rows: list[dict[str, Any]] = []
    sets: Dict[str, set] = {}

    for model, part in screen_metrics.groupby("dl_model"):
        means = (
            part.groupby(["fs_method", "top_k"])[response]
            .mean()
            .reset_index()
        )
        means["top_k_num"] = pd.to_numeric(means["top_k"], errors="raise")
        best = means.sort_values(
            [response, "top_k_num", "fs_method"],
            ascending=[False, True, True],
        ).iloc[0]
        best_key = (str(best.fs_method), str(best.top_k))

        best_by_seed = (
            part[
                (part.fs_method == best.fs_method)
                & (part.top_k.astype(str) == str(best.top_k))
            ]
            .set_index("seed")[response]
            .sort_index()
        )

        temp: list[dict[str, Any]] = []
        for (fs, k), g in part.groupby(["fs_method", "top_k"]):
            cur = g.set_index("seed")[response].sort_index()
            common = cur.index.intersection(best_by_seed.index)
            p = _paired_p(
                cur.loc[common].values,
                best_by_seed.loc[common].values,
                method,
            )
            row = {
                "dl_model": model,
                "fs_method": fs,
                "top_k": str(k),
                "response_metric": response,
                "best_fs_method": best_key[0],
                "best_top_k": best_key[1],
                "mean_response": float(cur.mean()),
                "best_mean_response": float(best_by_seed.mean()),
                "mean_difference_to_best": float(cur.mean() - best_by_seed.mean()),
                "p_raw": p,
            }
            # Preserve F1-specific columns used by existing analysis/reporting.
            if response == "f1":
                row["mean_f1"] = row["mean_response"]
                row["best_mean_f1"] = row["best_mean_response"]
            temp.append(row)

        pvals = [r["p_raw"] for r in temp]
        padj = (
            multipletests(pvals, alpha=alpha, method="holm")[1]
            if pvals
            else []
        )
        sets[str(model)] = set()
        for r, pa in zip(temp, padj):
            r["p_adjusted_holm"] = float(pa)
            r["statistically_comparable"] = bool(pa >= alpha)
            if r["statistically_comparable"]:
                sets[str(model)].add((str(r["fs_method"]), str(r["top_k"])))
            rows.append(r)

    return pd.DataFrame(rows), sets


def run_gate(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    screen_path = output_dir(cfg) / "04_screening" / dataset / "screen_design.csv"
    design = pd.read_csv(screen_path)
    seeds = [
        int(s)
        for s in cfg.get("screening", {}).get(
            "pilot_seeds", cfg["experiment"]["seeds"]
        )
    ]
    metrics = fetch_metrics(cfg, dataset, seeds, design)
    metrics["top_k"] = metrics["top_k"].astype(str)

    alpha = float(cfg["statistics"].get("alpha", 0.05))
    response = str(cfg["statistics"].get("screening_response", "f1"))
    contrast_test = str(cfg["statistics"].get("contrast_test", "paired_t"))

    test, full_model, _ = interaction_block_test(metrics, response)

    rep_mean = metrics.groupby(["fs_method", "top_k"])[response].mean().reset_index()
    rep_mean["top_k_num"] = pd.to_numeric(rep_mean["top_k"], errors="raise")
    d0row = rep_mean.sort_values(
        [response, "top_k_num", "fs_method"],
        ascending=[False, True, True],
    ).iloc[0]
    d0 = (str(d0row.fs_method), str(d0row.top_k))

    comp_df, comp_sets = comparable_sets(
        metrics,
        alpha,
        contrast_test,
        response=response,
    )
    decision_relevant_models = [
        model for model, comparable in comp_sets.items() if d0 not in comparable
    ]
    gate = bool(
        test["p_interaction"] < alpha
        and len(decision_relevant_models) > 0
    )

    out = output_dir(cfg) / "05_gate" / dataset
    out.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(out / "statistically_comparable_sets.csv", index=False)
    pd.DataFrame([test]).to_csv(out / "interaction_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "screening_response": response,
                "common_fs_method": d0[0],
                "common_top_k": d0[1],
                "common_mean_response": float(d0row[response]),
            }
        ]
    ).to_csv(out / "common_decision.csv", index=False)

    branches = []
    if gate:
        for model, comparable in comp_sets.items():
            for fs in sorted({fs for fs, _ in comparable}):
                branches.append(
                    {"dataset": dataset, "dl_model": model, "fs_method": fs}
                )
    pd.DataFrame(
        branches,
        columns=["dataset", "dl_model", "fs_method"],
    ).to_csv(out / "prioritized_branches.csv", index=False)

    result = {
        "dataset": dataset,
        "screening_response": response,
        "alpha": alpha,
        "p_interaction": float(test["p_interaction"]),
        "common_fs_method": d0[0],
        "common_top_k": d0[1],
        "decision_relevant_models": decision_relevant_models,
        "gate": "YES" if gate else "NO",
        "criterion": (
            "p_interaction < alpha AND common representation is outside "
            "at least one model's Holm-adjusted comparable set"
        ),
    }
    json_dump(out / "gate_decision.json", result)
    pd.DataFrame([result]).to_csv(out / "gate_decision.csv", index=False)

    params = full_model.params
    pd.DataFrame({"term": params.index, "estimate": params.values}).to_csv(
        out / "screening_model_fixed_effects.csv",
        index=False,
    )

    log(
        f"GATE | {dataset}: {result['gate']} response={response} "
        f"p_int={result['p_interaction']:.6g} d0={d0} "
        f"decision_relevant_models={decision_relevant_models}"
    )
    return result


def run_all_gates(cfg: Dict[str, Any], datasets: List[str] | None = None):
    datasets = datasets or list(cfg["datasets"])
    return [run_gate(cfg, dataset) for dataset in datasets]
