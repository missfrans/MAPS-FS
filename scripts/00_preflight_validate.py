import sys

from _common import base_parser, load
from mapsfs.validation import preflight
from mapsfs.provenance import collect_provenance


p = base_parser("Validate MAPS-FS configuration and dataset accessibility.")
args = p.parse_args()

try:
    cfg, _ = load(args)
    preflight(cfg)
    collect_provenance(cfg)

except FileNotFoundError as exc:
    missing = str(exc)
    print("", file=sys.stderr)
    print("===== MAPS-FS PREFLIGHT FAILED =====", file=sys.stderr)
    print("Missing required dataset file:", file=sys.stderr)
    print(f"  {missing}", file=sys.stderr)
    print("", file=sys.stderr)
    print("For the frozen manuscript configuration, place these files under data/:", file=sys.stderr)
    print("  data/UNSW_NB15_1.csv", file=sys.stderr)
    print("  data/UNSW_NB15_2.csv", file=sys.stderr)
    print("  data/UNSW_NB15_3.csv", file=sys.stderr)
    print("  data/UNSW_NB15_4.csv", file=sys.stderr)
    print("  data/HIKARI2021.csv", file=sys.stderr)
    print("", file=sys.stderr)
    print("See REPRODUCE_SINGLE_MACHINE.md for setup instructions.", file=sys.stderr)
    sys.exit(2)

except (KeyError, ValueError) as exc:
    print("", file=sys.stderr)
    print("===== MAPS-FS PREFLIGHT FAILED =====", file=sys.stderr)
    print(f"Configuration/dataset validation error: {exc}", file=sys.stderr)
    print("See REPRODUCE_SINGLE_MACHINE.md and configs/mapsfs_final.yaml.", file=sys.stderr)
    sys.exit(2)
