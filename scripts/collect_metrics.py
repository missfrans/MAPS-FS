from _common import base_parser, load
from mapsfs.runner import collect_all_metrics

p = base_parser("Collect per-run metrics.json files into metrics_all.csv.")
args = p.parse_args(); cfg, _ = load(args)
df = collect_all_metrics(cfg)
print(f"Collected {len(df)} unique rows")
