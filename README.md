# MAPS-FS

**MAPS-FS — Model-Aware Prioritized Search for Feature Selection** is a reproducible research framework for deciding **whether model-specific feature refinement is worth the additional search cost** in deep-learning-based network intrusion detection.

The implementation follows the methodological flow:

```text
SCREEN -> GATE -> PRIORITIZE -> REFINE -> SELECT
                         \
                          -> validation-only exhaustive oracle
```

MAPS-FS is **not a new feature-ranking algorithm**. Existing or user-defined feature selectors generate ranked representations. The framework decides whether those representations need model-specific refinement, allocates additional evaluations only where justified, selects a compact branch-specific feature count using the one-standard-error rule, and retains Pareto-efficient operating points.

## Why this repository differs from the earlier PAIR-IDS workflow

The previous PAIR-IDS implementation primarily evaluated the full FS x retained-feature-count x DL-model matrix and ranked configurations after training. MAPS-FS instead treats exhaustive search as a **validation oracle**. Weighted composite scores and normalization-dependent leaderboards are not used as the practical selection rule. See `MIGRATION_FROM_PAIR_IDS.md`.

### Final reference policy

The final MAPS-FS implementation intentionally removes the PAIR-IDS `overlap` subset. The only external reference representation is `full`. A common reduced representation, denoted `d0=(f0,k0)`, is learned from the screening evidence and is therefore not a manually predefined overlap subset.

## Core decision logic

The screening response is analyzed with two nested OLS models. Both models include FS, standardized retained-feature count, downstream model, and seed as a fixed blocking effect; the full model additionally includes FS x Model and k x Model interactions. A partial F-test evaluates the joint interaction block. MAPS-specific refinement is activated only when both conditions are met:

1. the model-related interaction block is statistically supported; and
2. the common model-independent representation is no longer statistically comparable with the screened model-specific best for at least one downstream model.

The second condition prevents statistically detectable but decision-irrelevant interaction from automatically expanding the search.

**Why seed is fixed rather than random in the final protocol:** the five seeds are predefined experimental repetitions, not a sampled population of random-effect groups. During audit, the earlier random-intercept MixedLM produced singular covariance and non-finite likelihoods when the estimated seed variance approached zero. The final protocol therefore uses an explicit fixed blocking effect for seed and a deterministic nested-model partial F-test.

## Repository structure

```text
MAPS_FS_github_ready/
├── configs/
│   ├── mapsfs_final.yaml       # frozen manuscript configuration
│   └── mapsfs_template.yaml    # template for custom datasets
├── data/
│   └── README.md
├── mapsfs/
│   ├── preprocessing.py
│   ├── feature_selectors.py
│   ├── feature_ranking.py
│   ├── models.py
│   ├── runner.py
│   ├── screening.py
│   ├── statistics.py
│   ├── refinement.py
│   ├── pareto.py
│   ├── oracle.py
│   ├── runtime.py
│   └── analysis.py
├── plugins/
│   └── example_custom.py
├── run_mapsfs.py             # one-command entry point
├── scripts/
│   ├── 00_preflight_validate.py
│   ├── 01_prepare_feature_rankings.py
│   ├── 02_run_dependency_screen.py
│   ├── 03_run_maps_gate.py
│   ├── 04_run_selective_refinement.py
│   ├── 05_run_efficiency_selection.py
│   ├── 06_run_exhaustive_oracle.py
│   ├── 07_validate_against_oracle.py
│   ├── 08_runtime_profile.py
│   └── 09_build_results_package.py
├── tests/
├── MIGRATION_FROM_PAIR_IDS.md
├── SELF_EXPLANATION_GUIDE.md
└── RESULTS_OUTPUT_GUIDE.md
```

## Installation

Python 3.11 is recommended. Use one environment path consistently for the complete experiment.

### CPU / generic environment

```bash
conda env create -f environment.yml
conda activate maps-fs
pip install -e .
```

Or with `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### NVIDIA GPU on WSL2

For the manuscript-scale experiment on WSL2, use the GPU environment so the CUDA runtime libraries expected by TensorFlow are installed inside the Python environment:

```bash
conda env create -f environment-gpu.yml
conda activate maps-fs
pip install -e . --no-deps
```

Verify the Windows/WSL NVIDIA driver and TensorFlow GPU visibility before running MAPS-FS:

```bash
nvidia-smi
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

For exact manuscript reproduction, run the complete pipeline on one computer using one frozen environment, one GPU/CPU setup, and `configs/mapsfs_final.yaml`. Do not combine runtime measurements from different machines.

## Data setup

For the **frozen manuscript reproduction** with `configs/mapsfs_final.yaml`, place the raw datasets under `data/` using these exact filenames:

```text
data/UNSW_NB15_1.csv
data/UNSW_NB15_2.csv
data/UNSW_NB15_3.csv
data/UNSW_NB15_4.csv
data/HIKARI2021.csv
```

These raw datasets are **not redistributed** by this repository. The filenames above are the canonical data contract for reproducing the published UNSW-NB15 and HIKARI2021 experiment.

For a **new/custom dataset**, copy and edit `configs/mapsfs_template.yaml` instead of changing the frozen manuscript configuration. MAPS-FS remains dataset-agnostic as long as the configuration defines a valid binary target and compatible feature schema.

Before the first run, verify carefully:

- label column;
- label-to-binary mapping;
- columns removed before feature selection;
- identifiers that should not be available to the detector;
- retained-feature counts `top_k_values`.

**Important:** MAPS-FS does not silently truncate an invalid `top_k`. If the prepared representation has 43 predictors, `top_k: 45` is an error. The unreduced representation is the explicit `full` reference.

## Canonical single-machine reproduction

The public release uses **one computer as the canonical execution model**. No additional orchestration or multi-computer setup is required. From a fresh checkout with the two datasets placed under `data/`, reproduce the manuscript pipeline with:

```bash
bash scripts/reproduce_manuscript_single_machine.sh
```

The script performs preflight validation, prints the expected experiment size, and then runs the full manuscript workflow sequentially on the same machine:

```text
preflight -> feature ranking -> D-optimal screening -> Gate
          -> automatic refinement path -> Pareto selection
          -> exhaustive validation-only Oracle -> Oracle validation
          -> repeated runtime profiling -> results package
```

The run store is resumable by default. If execution is interrupted, rerun the same command; compatible completed configurations are reused rather than retrained. The exhaustive Oracle is computationally expensive, but it does not require multiple computers. See `REPRODUCE_SINGLE_MACHINE.md`.

## One-command execution

MAPS-FS is intended to be executed end-to-end without manual branch selection. The gate automatically decides whether `04_selective_refinement` is activated.

### Practical use on a new dataset

```bash
python run_mapsfs.py --config configs/mapsfs_template.yaml --mode practical
```

After installation with `pip install -e .`, the equivalent command is:

```bash
mapsfs-run --config configs/mapsfs_template.yaml --mode practical
```

The practical mode runs:

```text
preflight -> feature ranking -> D-optimal screening -> MAPS gate
          -> automatic YES/NO branch -> efficiency selection -> final explanation
```

If the gate is `YES`, only prioritized FS-model branches receive full `k` refinement. If the gate is `NO`, model-specific refinement is skipped automatically and the common model-independent representation is retained. The exhaustive oracle is **not required** for practical use.

### Manuscript / methodological validation mode

```bash
python run_mapsfs.py --config configs/mapsfs_final.yaml --mode manuscript
```

This performs the same practical MAPS-FS path and then automatically adds the validation-only exhaustive oracle, oracle comparison, repeated runtime profiling of final Pareto candidates, the Results package, and a self-explanation report. Use `--skip-runtime-profile` only when repeated final-candidate profiling must be postponed.

The run store is shared, so configurations already trained during screening/refinement are reused if their protocol, schema, and selected-feature hashes are compatible. The request ledger is retained as provenance. Stage-7 practical-search accounting is reconstructed deterministically from the frozen screening design together with the gate/refinement artifacts.

### Final self-explanation

Every completed run writes:

```text
outputs/09_results_package/self_explanation/FINAL_SELF_EXPLANATION.md
outputs/09_results_package/self_explanation/FINAL_SELF_EXPLANATION.json
```

plus one Markdown/JSON report per dataset. The report explains whether refinement was activated, why the gate returned YES/NO, the common representation, prioritized/refined branches when applicable, Pareto-efficient candidates, objective-specific recommendations, and the recommended next step. In manuscript mode it also reports search saving and oracle comparison. See `SELF_EXPLANATION_GUIDE.md`.

## Repeated runtime profiling

The main pairing run records batched inference time. For manuscript-quality latency estimates, profile only the final Pareto-efficient candidates with repeated warm-up and timed runs:

```bash
python scripts/08_runtime_profile.py --config configs/mapsfs_template.yaml
```

Defaults are 5 warm-up runs and 30 timed repetitions per seed and candidate.

## Built-in feature selectors

- `rfe`: RFE with logistic-regression estimator;
- `lasso`: absolute L1-logistic coefficient ranking;
- `rf`: random-forest feature importance;
- `mrmr`: greedy mutual-information relevance minus correlation redundancy;
- `autoencoder`: input-weight importance from an autoencoder.

`full` is the sole unreduced reference and is not a ranked search method. MAPS-FS does not use an overlap subset.

## Built-in DL models

- CNN
- RNN
- DNN
- MLP
- LSTM

CNN/RNN/LSTM process the **feature axis** as an ordered input axis in the default implementation. They should not be described as learning packet-flow temporal dynamics unless the supplied data truly contain a temporal sequence.

## Using a different feature selector

Create a plugin function returning one row per feature with columns:

```text
rank, feature_index, feature_name, score
```

Example:

```python
from mapsfs.plugins import register_feature_selector

@register_feature_selector("my_selector")
def my_selector(X, y, feature_names, seed, params):
    ...
    return ranking_dataframe
```

Then add the module and method to YAML:

```yaml
plugins: [plugins.my_methods]
feature_selection:
  enabled: [rfe, my_selector]
```

See `plugins/example_custom.py`.

## Using a different DL architecture

Register a model builder and an input adapter:

```python
from mapsfs.plugins import register_model

@register_model("transformer", my_adapter)
def build_transformer(n_features, seed, params):
    ...
    return compiled_keras_model
```

A custom built-in-compatible model must support Keras-style `fit`, `predict`, and `count_params`. Add the plugin module and model name to YAML.

## Screening design

`screening.design: d_optimal_blocks` selects representation blocks `(FS, k)` and evaluates every selected representation across all enabled models. This blockwise design preserves within-model comparison and the common model-independent decision.

With:

```yaml
representation_blocks: auto_min_rank
```

the greedy D-optimal procedure adds representation blocks until the predefined fixed-effect design reaches full matrix rank. An explicit integer can be supplied instead; the script refuses a budget that cannot estimate the screening effects.

## Statistical gate outputs

For each dataset, `outputs/05_gate/<dataset>/` contains:

- `interaction_test.csv` — nested OLS partial F-test for the model-related interaction block, with seed treated as a fixed blocking effect;
- `common_decision.csv` — screened common representation `d0`;
- `statistically_comparable_sets.csv` — within-model paired contrasts to the screened best with Holm multiplicity adjustment;
- `prioritized_branches.csv` — FS-model branches that receive refinement when the gate is YES;
- `gate_decision.csv/json` — final YES/NO decision.

A candidate called “statistically comparable” means that the screened difference from the model-specific best was **not statistically distinguishable under the specified multiplicity-aware procedure**. It is not a formal equivalence claim.

## One-standard-error refinement

For every prioritized branch, all configured `k` values are evaluated across the refinement seeds. If `k_best` is the largest mean-F1 point, MAPS-FS retains the smallest `k` satisfying:

```text
mean_F1(k) >= mean_F1(k_best) - SE(k_best)
```

This is a parsimony rule. It should not be described as statistical equivalence.

## Efficiency selection

The default Pareto objectives are:

```yaml
f1: max
far: min
inference_time_per_flow_us: min
n_features: min
```

No arbitrary weighted score is needed. The result is an efficient set, not a forced universal winner. The final operating point can be chosen according to application constraints.

## Resume safety

Each trained run stores:

- training-protocol hash;
- prepared feature-schema hash;
- selected-feature hash.

`--resume` reuses a run only when these hashes agree. Incompatible outputs cause an error instead of being silently mixed. `--overwrite-stale` is deliberately explicit.

## Threshold policy

The example configuration retains the earlier fixed threshold 0.5:

```yaml
threshold_policy: fixed
fixed_threshold: 0.5
```

Two validation-only calibration options are implemented:

- `validation_f1`
- `validation_youden`

If threshold policy is changed, rerun **all configurations being compared** under the same protocol.

## Outputs for Results and Discussion

Run:

```bash
python scripts/09_build_results_package.py --config configs/mapsfs_template.yaml
```

The generated tables directly support:

- data integrity;
- feature-ranking robustness;
- H1 dependency/gate evidence;
- candidate prioritization and one-SE refinement;
- four-objective Pareto results;
- matched full/common baseline comparisons for H3;
- H2 search saving, performance differences, and exhaustive-front recovery when the oracle is available.

See `RESULTS_OUTPUT_GUIDE.md` for the exact mapping to a manuscript Results section.

## Reproducibility checklist

Before reporting final numbers:

1. Commit the exact YAML used for the experiment.
2. Record Python, TensorFlow, CUDA, GPU/CPU, and OS versions.
3. Run all final repeated seeds with the same feature schema.
4. Do not mix fixed-threshold and calibrated-threshold runs.
5. Do not silently clip invalid `k` values.
6. Validate the run store before analysis.
7. Keep exhaustive-oracle results conceptually separate from practical MAPS-FS execution.
8. Report mean, dispersion/CI, and matched baseline differences rather than only the best seed.

## Tests

```bash
pytest -q
python -m compileall mapsfs scripts
```

## License and citation

A `CITATION.cff` stub and `LICENSE_TEMPLATE.txt` are included. Replace them with the final manuscript citation and institution-approved software license before public release.

### Provenance manifest

`00_preflight_validate.py` also records `outputs/00_provenance/protocol_manifest.json` with the frozen configuration hash, Git commit when available, Python/package versions, operating system, TensorFlow build information, and visible GPU devices. Keep this file with the manuscript archive.
