from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

# Register built-ins on import.
from . import models as _models  # noqa: F401
from .config import output_dir, stable_hash, training_protocol_hash
from .feature_ranking import load_ranking
from .gpu import configure_tensorflow
from .plugins import load_plugins, require_model
from .preprocessing import load_prepared
from .utils import json_dump, json_load, log, set_seed

CONFIG_KEY = ["dataset", "seed", "fs_method", "top_k", "dl_model"]


def _sanitize_topk(top_k: str | int) -> str:
    return str(top_k)


def run_dir(cfg: Dict[str, Any], dataset: str, seed: int, fs_method: str, top_k: str | int, model: str) -> Path:
    return output_dir(cfg) / "03_runs" / dataset / f"seed_{seed}" / fs_method / f"topk_{_sanitize_topk(top_k)}" / model


def _selected_indices(cfg: Dict[str, Any], dataset: str, seed: int, fs_method: str, top_k: str | int) -> Tuple[np.ndarray, List[str]]:
    data = load_prepared(cfg, dataset, seed, mmap=True)
    names = data["feature_names"]["feature_name"].astype(str).tolist()
    if fs_method == "full":
        return np.arange(len(names), dtype=int), names
    k = int(top_k)
    if k >= len(names):
        raise ValueError(
            f"Invalid top_k={k} for {dataset} seed={seed}: prepared feature count is {len(names)}. "
            "Use the explicit full reference for the unreduced representation."
        )
    ranking = load_ranking(cfg, dataset, seed, fs_method)
    if len(ranking) < k:
        raise ValueError(f"Ranking {fs_method} for {dataset} seed={seed} has only {len(ranking)} rows; requested k={k}.")
    selected_names = ranking.head(k)["feature_name"].astype(str).tolist()
    index_map = {name: i for i, name in enumerate(names)}
    missing = [n for n in selected_names if n not in index_map]
    if missing:
        raise ValueError(f"Ranking/prepared schema mismatch for {dataset} seed={seed} {fs_method}: {missing[:5]}")
    return np.asarray([index_map[n] for n in selected_names], dtype=int), selected_names


def _choose_threshold(y_true: np.ndarray, y_prob: np.ndarray, cfg: Dict[str, Any]) -> float:
    training = cfg.get("training", {})
    policy = training.get("threshold_policy", "fixed")
    if policy == "fixed":
        return float(training.get("fixed_threshold", 0.5))
    if len(np.unique(y_true)) < 2:
        return float(training.get("fixed_threshold", 0.5))
    if policy == "validation_f1":
        # Deterministic candidate thresholds from validation probabilities plus endpoints.
        values = np.unique(np.clip(y_prob, 0.0, 1.0))
        if len(values) > int(training.get("max_threshold_candidates", 2001)):
            qs = np.linspace(0, 1, int(training.get("max_threshold_candidates", 2001)))
            values = np.quantile(values, qs)
        best = (-1.0, 0.5)
        for t in values:
            score = f1_score(y_true, y_prob >= t, zero_division=0)
            # Tie-break toward 0.5, then lower threshold.
            key = (float(score), -abs(float(t) - 0.5), -float(t))
            if key > (best[0], -abs(best[1] - 0.5), -best[1]):
                best = (float(score), float(t))
        return best[1]
    if policy == "validation_youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        idx = int(np.nanargmax(tpr - fpr))
        t = float(thresholds[idx])
        return float(np.clip(t, 0.0, 1.0))
    raise ValueError(f"Unknown threshold policy: {policy}")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "detection_rate": recall_score(y_true, y_pred, zero_division=0),
        "far": far,
        "fnr": fnr,
        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["average_precision"] = average_precision_score(y_true, y_prob)
    except Exception:
        out["average_precision"] = np.nan
    return out


def _append_ledger(cfg: Dict[str, Any], row: Dict[str, Any]) -> None:
    path = output_dir(cfg) / "03_runs" / "request_ledger.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = ["timestamp", "stage", *CONFIG_KEY, "status", "run_dir"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _existing_compatible(out: Path, expected: Dict[str, Any]) -> bool:
    manifest = out / "run_manifest.json"
    metrics = out / "metrics.json"
    if not (manifest.exists() and metrics.exists()):
        return False
    old = json_load(manifest)
    keys = ["training_protocol_hash", "data_fingerprint", "schema_hash", "selected_feature_hash"]
    return all(old.get(k) == expected.get(k) for k in keys)


def run_one(
    cfg: Dict[str, Any], dataset: str, seed: int, fs_method: str, top_k: str | int,
    model_name: str, stage: str, resume: bool = True, overwrite_stale: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    load_plugins(cfg.get("plugins", []))
    data = load_prepared(cfg, dataset, seed, mmap=True)
    indices, selected_names = _selected_indices(cfg, dataset, seed, fs_method, top_k)
    out = run_dir(cfg, dataset, seed, fs_method, top_k, model_name)
    out.mkdir(parents=True, exist_ok=True)

    selected_hash = stable_hash(selected_names)
    expected = {
        "training_protocol_hash": training_protocol_hash(cfg, dataset, fs_method, model_name),
        "data_fingerprint": data["manifest"].get("data_fingerprint"),
        "schema_hash": data["manifest"]["schema_hash"],
        "selected_feature_hash": selected_hash,
    }
    if resume and _existing_compatible(out, expected):
        metrics = json_load(out / "metrics.json")
        _append_ledger(cfg, {
            "timestamp": time.time(), "stage": stage, "dataset": dataset, "seed": seed,
            "fs_method": fs_method, "top_k": top_k, "dl_model": model_name,
            "status": "reused", "run_dir": str(out),
        })
        log(f"RUN REUSE | stage={stage} {dataset} seed={seed} {fs_method}-{top_k}-{model_name}")
        return metrics

    has_existing_artifacts = any(out.iterdir())
    if has_existing_artifacts and not overwrite_stale:
        raise RuntimeError(
            f"Stale/incompatible or incomplete run detected at {out}. Refusing to mix artifacts. "
            "Use --overwrite-stale only after reviewing the protocol change."
        )
    if has_existing_artifacts and overwrite_stale:
        import shutil
        shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

    X_fit = np.asarray(data["X_fit"][:, indices], dtype=np.float32)
    X_val = np.asarray(data["X_val"][:, indices], dtype=np.float32)
    X_test = np.asarray(data["X_test"][:, indices], dtype=np.float32)
    y_fit = np.asarray(data["y_fit"], dtype=np.int8)
    y_val = np.asarray(data["y_val"], dtype=np.int8)
    y_test = np.asarray(data["y_test"], dtype=np.int8)

    max_train = cfg.get("training", {}).get("train_sample_size")
    if max_train is not None and len(y_fit) > int(max_train):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(len(y_fit), int(max_train), replace=False)
        X_fit, y_fit = X_fit[sampled], y_fit[sampled]

    configure_tensorflow(cfg)
    spec = require_model(model_name)
    params = cfg.get("models", {}).get("params", {}).get(model_name, {}).copy()
    params.setdefault("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))
    model = spec.builder(X_fit.shape[1], seed, params)
    X_fit_m, X_val_m, X_test_m = spec.adapter(X_fit), spec.adapter(X_val), spec.adapter(X_test)

    try:
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required to train the built-in models.") from exc

    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(cfg.get("training", {}).get("early_stopping_patience", 5)),
        restore_best_weights=True,
    )]
    start = time.perf_counter()
    history = model.fit(
        X_fit_m, y_fit,
        validation_data=(X_val_m, y_val),
        epochs=int(cfg.get("training", {}).get("epochs", 30)),
        batch_size=int(cfg.get("training", {}).get("batch_size", 256)),
        verbose=int(cfg.get("training", {}).get("verbose", 0)),
        callbacks=callbacks,
    )
    train_sec = time.perf_counter() - start

    val_prob = model.predict(X_val_m, batch_size=int(cfg.get("training", {}).get("batch_size", 256)), verbose=0).ravel()
    threshold = _choose_threshold(y_val, val_prob, cfg)

    batch = int(cfg.get("training", {}).get("batch_size", 256))
    warm_n = min(len(X_test_m), max(batch, 256))
    _ = model.predict(X_test_m[:warm_n], batch_size=batch, verbose=0)
    t0 = time.perf_counter()
    test_prob = model.predict(X_test_m, batch_size=batch, verbose=0).ravel()
    test_sec = time.perf_counter() - t0
    latency_us = (test_sec / len(y_test)) * 1_000_000.0 if len(y_test) else np.nan

    metrics = evaluate_predictions(y_test, test_prob, threshold)
    metrics.update({
        "dataset": dataset, "seed": int(seed), "fs_method": fs_method,
        "top_k": str(top_k), "dl_model": model_name, "n_features": int(len(indices)),
        "n_fit": int(len(y_fit)), "n_validation": int(len(y_val)), "n_test": int(len(y_test)),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "training_time_sec": float(train_sec), "testing_time_sec": float(test_sec),
        "inference_time_per_flow_us": float(latency_us),
        "inference_time_per_flow_ms": float(latency_us / 1000.0),
        "throughput_flows_per_sec": float(len(y_test) / test_sec) if test_sec > 0 else np.nan,
        "trainable_params": int(model.count_params()), "threshold": float(threshold),
        "threshold_policy": cfg.get("training", {}).get("threshold_policy", "fixed"),
        "stage_created": stage,
    })

    pd.DataFrame(history.history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame({"feature_name": selected_names, "feature_index": indices}).to_csv(out / "selected_features.csv", index=False)
    pd.DataFrame([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                 index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(out / "confusion_matrix.csv")
    if cfg.get("training", {}).get("save_predictions", False):
        pd.DataFrame({"y_true": y_test, "y_prob": test_prob, "y_pred": test_prob >= threshold}).to_csv(out / "predictions.csv", index=False)
    if cfg.get("training", {}).get("save_models", False):
        model.save(out / "model.keras")

    json_dump(out / "metrics.json", metrics)
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)
    manifest = {
        **expected,
        "dataset": dataset, "seed": seed, "fs_method": fs_method, "top_k": str(top_k), "dl_model": model_name,
        "selected_features": selected_names,
        "config_hash": cfg.get("_config_hash"),
    }
    json_dump(out / "run_manifest.json", manifest)
    _append_ledger(cfg, {
        "timestamp": time.time(), "stage": stage, "dataset": dataset, "seed": seed,
        "fs_method": fs_method, "top_k": top_k, "dl_model": model_name,
        "status": "trained", "run_dir": str(out),
    })
    log(f"RUN DONE | stage={stage} {dataset} seed={seed} {fs_method}-{top_k}-{model_name} F1={metrics['f1']:.5f} FAR={metrics['far']:.6f} us/flow={latency_us:.3f}")
    try:
        keras.backend.clear_session()
    except Exception:
        pass
    del model, X_fit, X_val, X_test, X_fit_m, X_val_m, X_test_m
    gc.collect()
    return metrics


def collect_all_metrics(cfg: Dict[str, Any]) -> pd.DataFrame:
    root = output_dir(cfg) / "03_runs"
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for path in root.glob("**/metrics.json"):
        try:
            rows.append(json_load(path))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if len(df):
        df["top_k"] = df["top_k"].astype(str)
        df = df.sort_values(CONFIG_KEY).drop_duplicates(CONFIG_KEY, keep="last")
    out = root / "metrics_all.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def fetch_metrics(cfg: Dict[str, Any], dataset: str, seeds: Iterable[int], configs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in configs.iterrows():
        for seed in seeds:
            path = run_dir(cfg, dataset, int(seed), str(c.fs_method), str(c.top_k), str(c.dl_model)) / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Expected run missing: {path}")
            rows.append(json_load(path))
    return pd.DataFrame(rows)
