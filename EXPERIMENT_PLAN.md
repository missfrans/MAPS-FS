# Default experiment plan

This file reports the size implied by `configs/mapsfs_template.yaml`. It is a planning aid; changing enabled feature selectors, models, retained-feature counts, or seeds changes the counts automatically.

## Practical screening stage

With five feature selectors, five downstream models, and the automatic minimum-rank blockwise D-optimal design:

- UNSW-NB15: 6 selected `(FS, k)` representation blocks x 5 models x 5 screening seeds = **150 screening evaluations**.
- HIKARI2021: 6 selected `(FS, k)` representation blocks x 5 models x 5 screening seeds = **150 screening evaluations**.

Further refinement cost depends on the MAPS gate and prioritized branches, so it is intentionally not known before screening.

## Validation-only exhaustive oracle

The strict top-k policy excludes retained-feature counts at or above the full dimensionality. The default template therefore uses k <= 40 for UNSW-NB15 and k <= 70 for HIKARI2021; the unreduced representation is evaluated through the explicit `full` reference.

- UNSW-NB15 search space: 5 FS x 8 k x 5 models x 5 seeds = **1,000 search evaluations**, plus 25 full-reference evaluations.
- HIKARI2021 search space: 5 FS x 12 k x 5 models x 5 seeds = **1,500 search evaluations**, plus 25 full-reference evaluations.
- Total validation archive under the default template: **2,550 evaluations** including full references.

The exhaustive archive is not required for practical use of MAPS-FS on a new dataset.

## Canonical execution policy

The public reproducibility protocol executes the complete experiment on **one computer**. Seeds 42–46 are sequential/reusable repetitions within the same run store on that computer. The exhaustive Oracle may take substantial wall-clock time, but multi-computer execution is not required for scientific reproduction. Rerunning the same command safely resumes compatible completed configurations.
