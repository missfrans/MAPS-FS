#!/usr/bin/env bash
set -euo pipefail
CFG="${1:-configs/mapsfs_template.yaml}"
MODE="${2:-practical}"
python run_mapsfs.py --config "$CFG" --mode "$MODE"
