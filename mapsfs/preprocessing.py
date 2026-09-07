from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from .config import output_dir, stable_hash
from .utils import json_dump, json_load, log, set_seed


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def resolve_dataset_paths(ds_cfg: Dict[str, Any]) -> List[Path]:
    if ds_cfg.get("paths"):
        return [Path(p) for p in ds_cfg["paths"]]
    return [Path(ds_cfg["path"])]


def read_dataset(ds_cfg: Dict[str, Any]) -> pd.DataFrame:
    paths = resolve_dataset_paths(ds_cfg)
    frames = []
    reference = None
    for p in paths:
        part = read_table(p)
        part.columns = [str(c).strip() for c in part.columns]
        if reference is None:
            reference = list(part.columns)
        elif list(part.columns) != reference:
            missing = sorted(set(reference) - set(part.columns))
            extra = sorted(set(part.columns) - set(reference))
            if missing or extra:
                raise ValueError(f"Column mismatch in {p.name}: missing={missing}, extra={extra}")
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]



def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(ds_cfg: Dict[str, Any], strong_hash: bool = False) -> str:
    entries = []
    for path in resolve_dataset_paths(ds_cfg):
        stat = path.stat()
        item = {"name": path.name, "size_bytes": int(stat.st_size)}
        if strong_hash:
            item["sha256"] = _file_sha256(path)
        entries.append(item)
    return stable_hash(entries)

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    tokens = ["", " ", "NA", "N/A", "nan", "NaN", "None", "none", "null", "NULL", "-", "--", "?"]
    return df.replace(tokens, np.nan).replace([np.inf, -np.inf], np.nan)


def make_binary_label(y: pd.Series, positive_values: List[Any]) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        vals = set(pd.to_numeric(y.dropna(), errors="coerce").dropna().unique().tolist())
        if vals.issubset({0, 1, 0.0, 1.0}):
            return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    positives = {str(v).strip().lower() for v in positive_values}
    return y.astype(str).str.strip().str.lower().isin(positives).astype(int)


def _to_numeric_array(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()


def _to_string_array(X):
    frame = pd.DataFrame(X, dtype=object).mask(pd.DataFrame(X).isna(), np.nan)
    for col in frame.columns:
        frame[col] = frame[col].map(lambda v: np.nan if pd.isna(v) else str(v))
    return frame.to_numpy(dtype=object)


def infer_column_types(X_fit: pd.DataFrame, numeric_conversion_ratio: float = 0.95) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for col in X_fit.columns:
        s = X_fit[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
            continue
        converted = pd.to_numeric(s, errors="coerce")
        denom = max(int(s.notna().sum()), 1)
        if float(converted.notna().sum()) / denom >= numeric_conversion_ratio:
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocessor(X_fit: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[ColumnTransformer, List[str]]:
    pp = cfg.get("preprocessing", {})
    numeric, categorical = infer_column_types(X_fit, float(pp.get("numeric_conversion_ratio", 0.95)))

    numeric_pipe = Pipeline([
        ("to_numeric", FunctionTransformer(_to_numeric_array, validate=False)),
        ("imputer", SimpleImputer(strategy=pp.get("numeric_imputer", "median"))),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("to_string", FunctionTransformer(_to_string_array, validate=False)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor, numeric + categorical


def prepared_dir(cfg: Dict[str, Any], dataset: str, seed: int) -> Path:
    return output_dir(cfg) / "01_prepared" / dataset / f"seed_{seed}"


def _schema_hash(feature_names: List[str]) -> str:
    return stable_hash(feature_names)


def prepare_dataset_seed(cfg: Dict[str, Any], dataset: str, seed: int, force: bool = False) -> Dict[str, Any]:
    set_seed(seed)
    ds_cfg = cfg["datasets"][dataset]
    data_fp = dataset_fingerprint(ds_cfg, bool(cfg.get("preprocessing", {}).get("hash_data_files", False)))
    out = prepared_dir(cfg, dataset, seed)
    manifest_path = out / "manifest.json"

    prep_signature = stable_hash({
        "dataset": ds_cfg,
        "splits": cfg.get("splits", {}),
        "preprocessing": cfg.get("preprocessing", {}),
        "seed": seed,
    })
    if manifest_path.exists() and not force:
        old = json_load(manifest_path)
        if old.get("preparation_hash") == prep_signature:
            log(f"PREP SKIP compatible cache | {dataset} seed={seed}")
            return old

    raw = normalize_missing(read_dataset(ds_cfg))
    label_col = ds_cfg["label_col"]
    if label_col not in raw.columns:
        raise KeyError(f"Label column '{label_col}' not found in {dataset}")

    y = make_binary_label(raw[label_col], ds_cfg.get("positive_values", []))
    drop = set(ds_cfg.get("drop_cols", [])) | {label_col}
    if cfg.get("preprocessing", {}).get("drop_identifier_columns", True):
        drop |= set(ds_cfg.get("identifier_cols", []))
    X = raw.drop(columns=[c for c in drop if c in raw.columns], errors="ignore")

    all_empty = [c for c in X.columns if X[c].isna().all()]
    if all_empty:
        X = X.drop(columns=all_empty)

    splits = cfg.get("splits", {})
    test_size = float(splits.get("test_size", 0.30))
    val_within_train = float(splits.get("validation_size_within_train", 0.20))
    stratify = y if bool(splits.get("stratify", True)) else None
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    strat_pool = y_pool if bool(splits.get("stratify", True)) else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_pool, y_pool, test_size=val_within_train, random_state=seed, stratify=strat_pool
    )

    if cfg.get("preprocessing", {}).get("drop_constant_columns_train_only", False):
        constant = [c for c in X_fit.columns if X_fit[c].nunique(dropna=True) <= 1]
        X_fit = X_fit.drop(columns=constant)
        X_val = X_val.drop(columns=constant, errors="ignore")
        X_test = X_test.drop(columns=constant, errors="ignore")
    else:
        constant = []

    preprocessor, ordered_names = build_preprocessor(X_fit, cfg)
    X_fit_arr = np.asarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    X_val_arr = np.asarray(preprocessor.transform(X_val), dtype=np.float32)
    X_test_arr = np.asarray(preprocessor.transform(X_test), dtype=np.float32)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_fit.npy", X_fit_arr)
    np.save(out / "X_val.npy", X_val_arr)
    np.save(out / "X_test.npy", X_test_arr)
    np.save(out / "y_fit.npy", y_fit.to_numpy(dtype=np.int8))
    np.save(out / "y_val.npy", y_val.to_numpy(dtype=np.int8))
    np.save(out / "y_test.npy", y_test.to_numpy(dtype=np.int8))
    pd.DataFrame({"feature_index": range(len(ordered_names)), "feature_name": ordered_names}).to_csv(out / "feature_names.csv", index=False)
    joblib.dump(preprocessor, out / "preprocessor.joblib")

    available = len(ordered_names)
    invalid_k = [int(k) for k in ds_cfg.get("top_k_values", []) if int(k) >= available]
    # top-k equal to full dimensionality is deliberately rejected because it is
    # not a reduced representation; use the explicit 'full' reference instead.
    if invalid_k and cfg.get("preprocessing", {}).get("invalid_k_policy", "error") == "error":
        raise ValueError(
            f"{dataset} seed={seed}: top_k values {invalid_k} are >= available predictors ({available}). "
            "Remove them from top_k_values; the full representation is handled separately."
        )

    manifest = {
        "dataset": dataset,
        "seed": seed,
        "preparation_hash": prep_signature,
        "data_fingerprint": data_fp,
        "schema_hash": _schema_hash(ordered_names),
        "n_raw_rows": int(len(raw)),
        "n_raw_columns": int(raw.shape[1]),
        "n_predictors": int(available),
        "n_fit": int(len(y_fit)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "positive_rate_fit": float(np.mean(y_fit)),
        "positive_rate_validation": float(np.mean(y_val)),
        "positive_rate_test": float(np.mean(y_test)),
        "dropped_columns": sorted([c for c in drop if c in raw.columns]),
        "all_empty_columns": all_empty,
        "train_only_constant_columns": constant,
        "feature_names": ordered_names,
    }
    json_dump(manifest_path, manifest)
    log(f"PREP DONE | {dataset} seed={seed} predictors={available} fit={len(y_fit)} val={len(y_val)} test={len(y_test)}")
    return manifest


def load_prepared(cfg: Dict[str, Any], dataset: str, seed: int, mmap: bool = True):
    out = prepared_dir(cfg, dataset, seed)
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"Prepared cache missing: {out}. Run 01_prepare_feature_rankings.py first.")
    mode = "r" if mmap else None
    arrays = {
        "X_fit": np.load(out / "X_fit.npy", mmap_mode=mode),
        "X_val": np.load(out / "X_val.npy", mmap_mode=mode),
        "X_test": np.load(out / "X_test.npy", mmap_mode=mode),
        "y_fit": np.load(out / "y_fit.npy", mmap_mode=mode),
        "y_val": np.load(out / "y_val.npy", mmap_mode=mode),
        "y_test": np.load(out / "y_test.npy", mmap_mode=mode),
        "feature_names": pd.read_csv(out / "feature_names.csv"),
        "manifest": json_load(out / "manifest.json"),
    }
    return arrays


def validate_cross_seed_schema(cfg: Dict[str, Any], dataset: str, seeds: List[int]) -> str:
    manifests = [json_load(prepared_dir(cfg, dataset, s) / "manifest.json") for s in seeds]
    hashes = {m["schema_hash"] for m in manifests}
    counts = {m["n_predictors"] for m in manifests}
    if len(hashes) != 1 or len(counts) != 1:
        details = [(m["seed"], m["n_predictors"], m["schema_hash"][:12]) for m in manifests]
        raise ValueError(
            f"Cross-seed feature schema mismatch for {dataset}: {details}. "
            "MAPS-FS requires a common representation space across repeated seeds."
        )
    return manifests[0]["schema_hash"]
