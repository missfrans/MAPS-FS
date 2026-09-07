from _common import base_parser, load
from mapsfs.pareto import run_all_efficiency

p = base_parser("Apply four-objective Pareto efficiency selection and matched baseline comparisons.")
args = p.parse_args(); cfg, datasets = load(args)
run_all_efficiency(cfg, datasets)
