from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapsfs.config import load_config, validate_config  # noqa: E402


def base_parser(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "mapsfs_template.yaml")
    p.add_argument("--datasets", nargs="+", default=None)
    return p


def load(args):
    cfg = load_config(args.config)
    validate_config(cfg)
    datasets = args.datasets or list(cfg["datasets"])
    unknown = set(datasets) - set(cfg["datasets"])
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    return cfg, datasets
