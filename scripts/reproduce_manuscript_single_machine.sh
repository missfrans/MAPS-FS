#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root from this script location so the launcher works
# regardless of the caller's current working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CFG="${MAPSFS_CONFIG:-configs/mapsfs_final.yaml}"

echo "===== MAPS-FS SINGLE-MACHINE REPRODUCTION ====="
echo "Repository: ${REPO_ROOT}"
echo "Config: ${CFG}"
echo "Host: $(hostname)"
echo "Python: $(python --version 2>&1)"
echo

python scripts/00_preflight_validate.py --config "${CFG}"
python scripts/plan_experiment.py --config "${CFG}"
python run_mapsfs.py --config "${CFG}" --mode manuscript

echo
echo "===== MAPS-FS SINGLE-MACHINE REPRODUCTION COMPLETE ====="
