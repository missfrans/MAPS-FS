# Migration from PAIR-IDS to MAPS-FS

MAPS-FS reuses the strong parts of the earlier codebase—leakage-aware preprocessing, ranked feature selectors, DL training, per-flow metrics, and repeated runtime profiling—but changes the experiment controller.

## Main revisions

1. **Exhaustive pairing is no longer the practical pipeline.** It is isolated in `06_run_exhaustive_oracle.py` and used only for validation.
2. **No weighted pairing score is used for final selection.** Four-objective Pareto efficiency uses F1 (maximize), FAR, inference time, and retained dimensionality (minimize).
3. **No arbitrary fixed F1 tolerance is used for compact refinement.** The smallest k within one standard error of the branch-specific best is selected.
4. **No default GNN.** Built-in final models are CNN, RNN, DNN, MLP, and LSTM; researchers can register any custom architecture.
5. **Strict top-k validation.** A top-k at or above the actual full predictor count is an error rather than a silently truncated subset.
6. **Explicit fit/validation/test partition.** The preprocessor and feature ranking are fitted on the fit split; validation is reserved for early stopping and, when selected, threshold calibration; test is evaluation-only.
7. **Stale-output protection.** Resume checks training-protocol, feature-schema, and selected-feature hashes before reusing a run.
8. **Cross-seed schema validation.** Final repeated runs must share the same predictor schema. This prevents a mixture such as one seed using 80 predictors and others using 79.
9. **The MAPS gate is explicit.** Mixed-effects interaction evidence must be statistically supported and the common representation must fall outside at least one model's multiplicity-adjusted comparable set.
10. **All outputs are organized around manuscript hypotheses H1-H3**, rather than an overall weighted leaderboard.

## One deliberate compatibility choice

The example YAML keeps a fixed probability threshold of 0.5 because this matches the earlier experimental protocol. The code also supports `validation_f1` and `validation_youden`. Changing threshold policy is a protocol change and requires rerunning every configuration that will be compared.

## Overlap subset removed in final MAPS-FS

The PAIR-IDS overlap subset is not part of MAPS-FS. MAPS-FS uses only the unreduced `full` baseline plus the data-derived common model-independent decision `d0=(f0,k0)`. This keeps the executable framework aligned with the methodology and avoids introducing a reference subset that is not used by the MAPS-FS gate, refinement rule, or hypotheses.
