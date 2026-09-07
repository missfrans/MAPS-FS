#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/mapsfs_final.yaml}"
python scripts/06_run_exhaustive_oracle.py --config "$CFG"
python scripts/07_validate_against_oracle.py --config "$CFG"
python scripts/09_build_results_package.py --config "$CFG"
