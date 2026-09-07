# Results and Discussion output guide

`09_build_results_package.py` consolidates analysis into `outputs/09_results_package/`.

| Output | Manuscript role |
|---|---|
| `R1_data_integrity.csv` | dataset, split, predictor-count, and cross-seed integrity reporting |
| `R2_feature_stability_summary.csv` | robustness of feature rankings across seeds |
| `R2_stable_feature_frequency.csv` | recurring top-ranked features; supplementary robustness |
| `R3_dependency_gate_results.csv` | H1: omnibus interaction evidence and YES/NO MAPS decision |
| `R4_one_se_refinement.csv` | prioritized branches, branch-specific best k, and one-SE compact k* |
| `R5_efficiency_candidates.csv` | F1, FAR, inference time, dimensionality, and Pareto membership |
| `R6_matched_baseline_comparisons.csv` | H3: paired deltas versus full-feature and common model-independent baselines |
| `R7_oracle_search_validation.csv` | H2: search saving and efficient-front recovery |
| `R8_oracle_performance_differences.csv` | H2: model-wise MAPS-vs-exhaustive best F1 differences |
| `R9_hypothesis_evidence_summary.csv` | compact evidence map for H1-H3; not an automatic claim generator |
| `RESULTS_SUMMARY.md` | concise numerical checklist for manuscript drafting |

## Recommended Results structure

### 4.1 Data integrity and reference landscape
Use R1 and report the number of candidate configurations. Do not mix incompatible seeds.

### 4.2 Model-related dependency and MAPS gate (H1)
Use R3 together with `outputs/05_gate/<dataset>/interaction_test.csv` and `statistically_comparable_sets.csv`. Report the nested OLS partial F-test with seed as a fixed blocking effect. A different observed argmax by model is **not** sufficient by itself; report the omnibus interaction evidence and the decision relevance of the common representation.

### 4.3 Prioritized refinement
Use R4 and the trajectory CSVs. Make clear that one-SE is a parsimony rule, not a formal equivalence test.

### 4.4 Search efficiency against the exhaustive oracle (H2)
Use R7-R8. Report overall search saving (including screening cost), post-Gate refinement saving, performance differences, and recovery of oracle-efficient configurations. The exhaustive grid is validation-only and is not needed on a new dataset.

### 4.5 Detection-efficiency outcomes (H3)
Use R5-R6. Report F1, FAR, inference latency, and retained dimensionality versus matched baselines. Pareto membership alone does not prove deployment benefit.

### 4.6 Robustness
Use feature-ranking stability and repeated runtime profiling. Keep normalization/weight-sensitivity analyses, if desired, as supplementary evidence rather than MAPS-FS core logic.

## One-command manuscript output

For the complete evidence package used to write Results and Discussion, run:

```bash
python run_mapsfs.py --config configs/mapsfs_template.yaml --mode manuscript
```

No manual decision is required after the gate. `04_selective_refinement` is invoked automatically only when the MAPS gate returns YES; otherwise the model-independent path is retained and the pipeline continues to efficiency selection.

## Self-explanation files

The final interpretation aid is written under:

```text
outputs/09_results_package/self_explanation/
```

Key files:

- `FINAL_SELF_EXPLANATION.md`: concise cross-dataset decision summary;
- `FINAL_SELF_EXPLANATION.json`: machine-readable equivalent;
- `<DATASET>_FINAL_RECOMMENDATION.md`: detailed decision, refinement outcome, Pareto recommendations, oracle validation, and next step;
- `<DATASET>_FINAL_RECOMMENDATION.json`: machine-readable dataset report.

Use these reports as an evidence map for manuscript prose, not as text to copy without checking context. The generator intentionally avoids calling a Pareto point a universal winner and does not treat non-significance as formal equivalence.

## Baseline interpretation

MAPS-FS reports two manuscript-relevant baselines: (1) the unreduced `full` representation and (2) the data-derived common model-independent representation `d0`. There is no overlap-subset baseline in the final implementation.
