from _common import base_parser, load
from mapsfs.refinement import run_all_refinements

p = base_parser("Run branch prioritization and one-standard-error refinement.")
p.add_argument("--no-resume", action="store_true")
p.add_argument("--overwrite-stale", action="store_true")
args = p.parse_args(); cfg, datasets = load(args)
run_all_refinements(cfg, datasets, resume=not args.no_resume, overwrite_stale=args.overwrite_stale)
