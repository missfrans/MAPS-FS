from _common import base_parser, load
from mapsfs.feature_ranking import run_all_rankings

p = base_parser("Prepare leakage-safe splits and feature rankings.")
p.add_argument("--seeds", nargs="+", type=int, default=None)
p.add_argument("--force", action="store_true")
args = p.parse_args(); cfg, datasets = load(args)
run_all_rankings(cfg, datasets=datasets, seeds=args.seeds, force=args.force)
