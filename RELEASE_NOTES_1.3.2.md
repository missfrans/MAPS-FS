# MAPS-FS v1.3.2 — Single-Machine Reproducibility Release

## Canonical reproduction model

The public MAPS-FS release is designed to reproduce the complete manuscript experiment on one computer. No multi-computer orchestration, seed-sharded configuration files, or merge step is required. The canonical command is:

```bash
bash scripts/reproduce_manuscript_single_machine.sh
```

The run store resumes compatible completed configurations automatically, so long Oracle runs can be continued after interruption on the same machine.

## Scientific protocol retained

- Datasets: UNSW-NB15 and HIKARI2021.
- Feature selectors: RFE, LASSO, Random Forest, mRMR, Autoencoder.
- Downstream models: CNN, RNN, DNN, MLP, LSTM.
- Seeds: 42–46.
- Screening: D-optimal, six representation blocks per dataset.
- Gate: nested OLS with seed as a fixed blocking effect; partial F-test; F1 primary response; paired seed-level comparisons with Holm correction.
- Frozen decisions: UNSW-NB15 Gate=NO, d0=RFE-40; HIKARI2021 Gate=NO, d0=LASSO-5.
- Exhaustive search remains validation-only and must not alter Gate or d0.

## Stage-7 accounting correction

Practical search cost is reconstructed from the frozen screening design and gate/refinement artifacts. For the frozen experiment:

- UNSW-NB15: 150 practical evaluations versus 1000 ranked Oracle evaluations = 85.0% overall search saving; post-Gate refinement saving = 100.0%.
- HIKARI2021: 150 practical evaluations versus 1500 ranked Oracle evaluations = 90.0% overall search saving; post-Gate refinement saving = 100.0%.

This is a reporting/reproducibility correction only. It does not change trained metrics, performance regret, Pareto objectives, Gate decisions, or common representations.

## Reader setup hardening

- The canonical launcher resolves the repository root automatically, so it can be invoked from any working directory.
- `README.md` and `REPRODUCE_SINGLE_MACHINE.md` use the same frozen dataset filenames required by `configs/mapsfs_final.yaml`.
- Missing dataset files now stop at preflight with a concise setup message instead of an expected Python traceback.

