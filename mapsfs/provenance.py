from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .config import output_dir
from .utils import json_dump


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _git_commit(root: Path):
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_provenance(cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(cfg["_project_root"])
    info = {
        "config_path": cfg.get("_config_path"),
        "config_hash": cfg.get("_config_hash"),
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: _version(name) for name in ["numpy","pandas","scikit-learn","scipy","statsmodels","patsy","tensorflow","PyYAML"]},
    }
    try:
        import tensorflow as tf
        info["tensorflow_build"] = tf.sysconfig.get_build_info()
        info["gpu_devices"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        info["tensorflow_runtime_note"] = str(exc)
        info["gpu_devices"] = []
    out = output_dir(cfg) / "00_provenance" / "protocol_manifest.json"
    json_dump(out, info)
    return info
