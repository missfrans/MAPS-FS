from _common import base_parser, load
from mapsfs.screening import build_screen_design
from mapsfs.oracle import oracle_design

p = base_parser("Print MAPS-FS screening and exhaustive-oracle experiment sizes without training.")
args = p.parse_args(); cfg, datasets = load(args)
for ds in datasets:
    screen, summary = build_screen_design(cfg, ds)
    oracle = oracle_design(cfg, ds)
    pilot = len(cfg.get('screening',{}).get('pilot_seeds',cfg['experiment']['seeds']))
    oracle_seeds = len(cfg.get('oracle',{}).get('seeds',cfg['experiment']['seeds']))
    n_search = int((~oracle.reference.astype(bool)).sum())
    print(f"{ds}: screening blocks={summary['selected_representation_blocks']}, screening configs/seed={len(screen)}, screening evaluations={len(screen)*pilot}")
    print(f"{ds}: exhaustive search configs/seed={n_search}, exhaustive search evaluations={n_search*oracle_seeds}, reference configs/seed={int(oracle.reference.astype(bool).sum())}")
