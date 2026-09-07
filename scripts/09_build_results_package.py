from _common import base_parser, load
from mapsfs.analysis import build_results_package

p = base_parser("Build manuscript-ready Results and Discussion tables/figures.")
args = p.parse_args(); cfg, _ = load(args)
build_results_package(cfg)
