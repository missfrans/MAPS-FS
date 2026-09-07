from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .analysis import build_results_package
from .config import load_config, validate_config
from .feature_ranking import run_all_rankings
from .oracle import run_all_oracles, validate_against_oracle
from .pareto import run_all_efficiency
from .provenance import collect_provenance
from .refinement import run_all_refinements
from .reporting import build_self_explanation
from .runtime import profile_final_candidates
from .screening import run_dependency_screen
from .statistics import run_all_gates
from .utils import log
from .validation import preflight


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="One-command MAPS-FS execution: SCREEN -> GATE -> PRIORITIZE -> REFINE -> SELECT, with optional manuscript oracle validation."
    )
    p.add_argument("--config", type=Path, default=root / "configs" / "mapsfs_template.yaml")
    p.add_argument("--datasets", nargs="+", default=None, help="Optional subset of configured datasets.")
    p.add_argument(
        "--mode", choices=["practical", "manuscript"], default="practical",
        help="practical: MAPS-FS only. manuscript: also run exhaustive oracle, oracle validation, repeated runtime profiling, and results package."
    )
    p.add_argument("--force-rankings", action="store_true", help="Rebuild prepared data and feature rankings.")
    p.add_argument("--no-resume", action="store_true", help="Do not reuse compatible trained configurations.")
    p.add_argument("--overwrite-stale", action="store_true", help="Explicitly overwrite incompatible stale run outputs.")
    p.add_argument("--skip-runtime-profile", action="store_true", help="In manuscript mode, skip repeated final-candidate runtime profiling.")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)
    datasets = args.datasets or list(cfg["datasets"])
    unknown = set(datasets) - set(cfg["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    # Use a dataset-scoped config so downstream result builders do not expect datasets that were not requested.
    scoped = deepcopy(cfg)
    scoped["datasets"] = {d: cfg["datasets"][d] for d in datasets}
    resume = not args.no_resume

    log(f"MAPS-FS START | mode={args.mode} datasets={datasets}")
    preflight(scoped)
    collect_provenance(scoped)
    run_all_rankings(scoped, datasets=datasets, force=args.force_rankings)
    run_dependency_screen(scoped, datasets, resume=resume, overwrite_stale=args.overwrite_stale)
    gates = run_all_gates(scoped, datasets)
    for g in gates:
        log(f"AUTO-BRANCH | {g['dataset']}: gate={g['gate']} -> " + ("selective model-aware refinement" if g["gate"] == "YES" else "model-independent path"))
    run_all_refinements(scoped, datasets, resume=resume, overwrite_stale=args.overwrite_stale)
    run_all_efficiency(scoped, datasets)

    if args.mode == "manuscript":
        run_all_oracles(scoped, datasets, resume=resume, overwrite_stale=args.overwrite_stale)
        for ds in datasets:
            validate_against_oracle(scoped, ds)
        if not args.skip_runtime_profile:
            for ds in datasets:
                profile_final_candidates(scoped, ds)

    results_root = build_results_package(scoped)
    explanation_root = build_self_explanation(scoped, datasets)
    log(f"MAPS-FS COMPLETE | results={results_root} self_explanation={explanation_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
