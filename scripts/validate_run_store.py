from _common import base_parser, load
from mapsfs.validation import validate_run_store

p = base_parser("Validate collected MAPS-FS run metrics.")
args = p.parse_args(); cfg, _ = load(args)
df = validate_run_store(cfg)
print(f"VALIDATION PASSED: {len(df)} unique run records")
