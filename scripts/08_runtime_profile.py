from _common import base_parser, load
from mapsfs.runtime import profile_final_candidates

p = base_parser("Repeated runtime profiling for Pareto-efficient final candidates.")
args = p.parse_args(); cfg, datasets = load(args)
for ds in datasets:
    profile_final_candidates(cfg, ds)
