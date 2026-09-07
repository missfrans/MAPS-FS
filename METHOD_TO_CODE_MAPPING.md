# MAPS-FS method-to-code mapping

This document links the manuscript framework to executable components.

## Candidate notation

- feature selector: `fs_method`
- retained-feature count: `top_k`
- downstream model: `dl_model`
- repeated run: `seed`
- complete candidate: `(fs_method, top_k, dl_model)`

The feature selector owns the ranking rule. MAPS-FS does not modify that rule.

## Dependency screening — Eq. (1)

Implemented in:

- `mapsfs/screening.py`
- `mapsfs/statistics.py`

The blockwise D-optimal screen chooses informative `(FS, k)` representations and crosses every selected block with every enabled model. The screening model uses:

```text
response ~ C(FS) + k_scaled + C(Model)
         + C(FS):C(Model)
         + k_scaled:C(Model)
```

with `C(seed)` as a fixed blocking effect. The reduced model omits the two model-related interaction terms, while the full model adds `C(FS):C(Model)` and `k_scaled:C(Model)`. Their joint contribution is evaluated using a nested-model partial F-test. This is the final prespecified Gate inference used by the repository.

## MAPS gate — Eq. (2)

Implemented in `mapsfs/statistics.py`.

- `p_interaction`: omnibus test for the predefined interaction block.
- `d0`: screened `(FS, k)` representation with the highest response after averaging across downstream models and repeated seeds.
- `R_m`: screened representations not statistically distinguishable from the within-model screened best after Holm adjustment of paired seed-level contrasts.

The gate is YES only when:

```text
p_interaction < alpha
AND
d0 is outside R_m for at least one downstream model
```

## Candidate prioritization

Also implemented in `mapsfs/statistics.py`.

For a YES gate, an `(FS, Model)` branch is promoted if at least one screened retained-feature decision for that feature selector belongs to `R_m` for the model. Prioritization removes clearly inferior screened branches; it does not select the final subset.

## Model-aware compact refinement — Eq. (3)-(4)

Implemented in `mapsfs/refinement.py`.

For each promoted `(FS, Model)` branch:

1. evaluate the complete configured k trajectory across refinement seeds;
2. find `k_best`, the highest mean-F1 point;
3. calculate `mean_F1(k_best) - SE(k_best)`;
4. keep the smallest k whose mean F1 is at least that threshold.

The one-standard-error rule is a parsimony rule, not a formal equivalence test.

## Efficiency selection — Eq. (5)

Implemented in `mapsfs/pareto.py`.

Default objective vector:

```text
[-F1, FAR, inference_time_per_flow_us, n_features]
```

The code exposes directions in YAML rather than hard-coding a weighted score. Final candidates are compared with both the matched full-feature baseline and the common model-independent baseline under the same downstream model.

## Validation-only exhaustive oracle

Implemented in `mapsfs/oracle.py` and scripts 06-07.

The oracle is intentionally outside the practical MAPS-FS path. It reports:

- screening, additional-refinement, and total practical search evaluations versus exhaustive ranked-search evaluations;
- overall search saving and post-Gate refinement saving;
- model-wise F1 differences between MAPS-FS and exhaustive best;
- number and fraction of exhaustive Pareto-efficient configurations recovered by MAPS-FS.

These quantities support H2 but are not required when using MAPS-FS on a new dataset.

## Automatic orchestration and self-explanation

| Framework function | One-command implementation |
|---|---|
| Automatic execution | `run_mapsfs.py` / `mapsfs.cli` |
| Gate-controlled YES/NO branch | `mapsfs.cli` + `mapsfs.refinement.run_refinement` |
| Final decision explanation | `mapsfs.reporting.build_self_explanation` |
| Dataset-specific next-step recommendation | `mapsfs.reporting.build_dataset_explanation` |
| Objective-specific Pareto representatives | `mapsfs.reporting._use_case_recommendations` |
| Manuscript evidence package | `mapsfs.analysis.build_results_package` |

The command-line orchestrator never asks the user to manually decide whether refinement should run. That decision is produced by the prespecified MAPS gate.

## Reference representation policy

Only `full` is implemented as a fixed reference representation. The common reduced representation `d0` is estimated by the gate stage from screened FS–k evidence; no predefined overlap subset is used.
