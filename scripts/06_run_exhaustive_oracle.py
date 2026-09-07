"""VALIDATION ONLY. This script is not required for practical MAPS-FS deployment."""
from _common import base_parser, load
from mapsfs.oracle import run_all_oracles

p = base_parser("Run the exhaustive validation-only oracle.")
p.add_argument("--no-resume", action="store_true")
p.add_argument("--overwrite-stale", action="store_true")
args = p.parse_args(); cfg, datasets = load(args)
run_all_oracles(cfg, datasets, resume=not args.no_resume, overwrite_stale=args.overwrite_stale)
