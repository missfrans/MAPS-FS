from _common import base_parser, load
from mapsfs.screening import run_dependency_screen

p = base_parser("Run MAPS-FS D-optimal dependency screening.")
p.add_argument("--no-resume", action="store_true")
p.add_argument("--overwrite-stale", action="store_true")
args = p.parse_args(); cfg, datasets = load(args)
run_dependency_screen(cfg, datasets, resume=not args.no_resume, overwrite_stale=args.overwrite_stale)
