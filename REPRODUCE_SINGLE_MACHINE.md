# Reproducing MAPS-FS on One Computer

This is the canonical public reproduction path for MAPS-FS v1.3.2. The complete experiment runs on one computer using one resumable run store; no multi-computer orchestration is required.

## 1. Environment

Python 3.11 is recommended. For the GPU environment used for manuscript-scale execution:

```bash
conda env create -f environment-gpu.yml
conda activate maps-fs
pip install -e . --no-deps
```

For CPU/generic execution:

```bash
conda env create -f environment.yml
conda activate maps-fs
pip install -e .
```

Use the same environment and hardware for the complete run, especially for runtime comparisons.

## 2. Dataset files

Place the raw files under `data/` using the filenames expected by `configs/mapsfs_final.yaml`:

```text
data/UNSW_NB15_1.csv
data/UNSW_NB15_2.csv
data/UNSW_NB15_3.csv
data/UNSW_NB15_4.csv
data/HIKARI2021.csv
```

Raw datasets are not redistributed by this repository.

## 3. Preflight and experiment size

```bash
python scripts/00_preflight_validate.py --config configs/mapsfs_final.yaml
python scripts/plan_experiment.py --config configs/mapsfs_final.yaml
```

For the frozen configuration the planned counts are:

- UNSW-NB15 screening: 150 evaluations.
- HIKARI2021 screening: 150 evaluations.
- UNSW-NB15 exhaustive ranked Oracle: 1000 evaluations + 25 full-reference evaluations.
- HIKARI2021 exhaustive ranked Oracle: 1500 evaluations + 25 full-reference evaluations.
- Canonical Oracle total: 2550 evaluations including full references.

## 4. Run the complete manuscript pipeline

```bash
bash scripts/reproduce_manuscript_single_machine.sh
```

Equivalent direct command:

```bash
python run_mapsfs.py --config configs/mapsfs_final.yaml --mode manuscript
```

All seeds 42–46 are executed by the same process on the same computer. The pipeline automatically follows Gate=YES/NO decisions; the Oracle remains validation-only.

## 5. Resume after interruption

Resume is enabled by default. If the computer is restarted or the run is interrupted, activate the same environment and rerun:

```bash
bash scripts/reproduce_manuscript_single_machine.sh
```

Compatible completed runs are reused. Do not pass `--no-resume` unless a full retraining is intentionally required.

## 6. Expected frozen validation evidence

For the published/frozen experiment, the validated Stage-7 accounting is:

- UNSW-NB15: 150 practical evaluations versus 1000 ranked Oracle evaluations = 85.0% overall search saving; 100.0% post-Gate refinement saving.
- HIKARI2021: 150 practical evaluations versus 1500 ranked Oracle evaluations = 90.0% overall search saving; 100.0% post-Gate refinement saving.

These values are validation evidence. The Oracle must not be used to change the Gate decision or common representation.

## 7. Final outputs

Key outputs are written under:

```text
outputs/05_gate/
outputs/07_efficiency/
outputs/08_oracle/
outputs/09_results_package/
```

Validate the canonical run store after completion:

```bash
python scripts/validate_run_store.py --config configs/mapsfs_final.yaml
```
