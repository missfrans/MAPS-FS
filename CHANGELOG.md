# Changelog

## 1.3.2 — Single-machine reproducibility and Stage-7 accounting

- Hardened the public reader path: the README now lists the exact frozen dataset filenames, the launcher is working-directory independent, and expected preflight setup failures produce concise guidance instead of a Python traceback.

- Made one-computer execution the canonical public reproduction path.
- Added `scripts/reproduce_manuscript_single_machine.sh` and `REPRODUCE_SINGLE_MACHINE.md` for end-to-end manuscript reproduction with `configs/mapsfs_final.yaml`.
- Removed public worker/merge orchestration artifacts; seeds 42–46 run through the same resumable run store on one computer.
- Preserved the audited nested OLS Gate, fixed seed blocking, partial F-test, F1 primary response, paired seed-level comparisons with Holm correction, D-optimal screening, one-SE refinement, Pareto selection, and validation-only exhaustive Oracle.
- Corrected Stage-7 practical-search accounting so it is reconstructed from the frozen screening design and gate/refinement artifacts rather than relying on an incomplete request ledger.
- Frozen corrected search-saving evidence: UNSW-NB15 85.0% overall / 100.0% post-Gate; HIKARI2021 90.0% overall / 100.0% post-Gate.
- The release correction does not alter Gate decisions, common representations, trained metrics, performance regret, Pareto objectives, or Oracle configurations.

## 1.3.0 — audited OLS Gate release

- Replaced the random-intercept MixedLM Gate analysis with nested OLS models using seed as a fixed experimental blocking effect.
- Added finite-statistic checks for the Gate partial F-test.
- Made Holm-adjusted comparable sets follow the configured `screening_response` instead of being hard-coded to F1.
- Added positive-control and negative-control Gate regression tests (`NO` and `YES` expected behavior).
- Updated Gate documentation and method-to-code mapping to match the audited inferential protocol.
- Preserved the no-overlap policy, D-optimal `auto_min_rank` screening, F1 primary Gate response, paired seed-level contrasts, Holm correction, one-SE refinement, and validation-only exhaustive oracle.

## 1.2.0 — no-overlap MAPS-FS

- Removed the PAIR-IDS `overlap` representation from configuration, training selection, and exhaustive-oracle construction.
- `full` is now the only fixed reference representation.
- The common reduced representation is always the screening-derived `d0=(f0,k0)`.
- Added regression tests to prevent overlap support from being reintroduced accidentally.
- Updated README and manuscript-output documentation to match the final MAPS-FS method.
- Added `environment-gpu.yml` / `requirements-gpu.txt` for reproducible TensorFlow 2.21 GPU installation on WSL2.
- Pinned TensorFlow 2.21.0 in the reference package metadata.

## 1.0.0

Initial MAPS-FS research release generated from the earlier PAIR-IDS codebase.

- replaces exhaustive-first pairing with Screen -> Gate -> Prioritize -> Refine -> Select;
- isolates exhaustive search as validation-only oracle;
- adds blockwise D-optimal screening;
- added the original interaction gate and decision relevance check (the inferential implementation was superseded by the audited OLS Gate in v1.3.0);
- adds one-standard-error compact refinement;
- removes weighted composite score from core selection;
- uses four-objective Pareto efficiency;
- adds strict top-k and cross-seed schema checks;
- adds protocol/schema/feature hashes for safe resume;
- adds explicit fit/validation/test partitions;
- supports custom feature selectors and Keras-compatible DL models through plugins;
- generates H1-H3-oriented Results and Discussion tables.

## 1.1.0

- Added one-command execution through `python run_mapsfs.py` and installed CLI `mapsfs-run`.
- Added automatic gate-controlled branching: YES triggers selective refinement; NO retains the model-independent path without manual intervention.
- Added `--mode practical` and `--mode manuscript`.
- Manuscript mode automatically adds exhaustive-oracle validation and optional repeated runtime profiling.
- Added machine-readable and Markdown self-explanation reports with gate rationale, refinement results, Pareto-efficient recommendations, objective-specific operating points, oracle evidence, and recommended next steps.
- Added `R10_runtime_profile_summary.csv` when repeated runtime profiling is available.
- Added `SELF_EXPLANATION_GUIDE.md` and updated README/method-to-code/results documentation.
