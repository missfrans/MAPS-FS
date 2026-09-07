from _common import base_parser, load
from mapsfs.oracle import validate_against_oracle

p = base_parser("Quantify MAPS-FS search saving, performance differences, and oracle-front recovery.")
args = p.parse_args(); cfg, datasets = load(args)
for ds in datasets:
    validate_against_oracle(cfg, ds)
