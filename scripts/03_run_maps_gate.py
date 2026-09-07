from _common import base_parser, load
from mapsfs.statistics import run_all_gates

p = base_parser(
    "Fit the nested OLS screening models with seed fixed blocking effects "
    "and evaluate the MAPS Gate."
)
args = p.parse_args()
cfg, datasets = load(args)
run_all_gates(cfg, datasets)
