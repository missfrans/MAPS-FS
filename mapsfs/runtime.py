from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import models as _models  # noqa: F401
from .config import output_dir
from .plugins import load_plugins, require_model
from .gpu import configure_tensorflow
from .preprocessing import load_prepared
from .runner import _selected_indices, _choose_threshold, evaluate_predictions
from .utils import log, set_seed


def _stats(values):
    v = np.asarray(values, float)
    n = len(v)
    mean = float(v.mean()) if n else np.nan
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sd, ci, float(v.min()) if n else np.nan, float(v.max()) if n else np.nan


def profile_candidate(cfg: Dict[str, Any], dataset: str, seed: int, fs_method: str, top_k, model_name: str, warmup: int, repeats: int):
    set_seed(seed); load_plugins(cfg.get("plugins", []))
    data = load_prepared(cfg, dataset, seed, mmap=True)
    idx, _ = _selected_indices(cfg, dataset, seed, fs_method, top_k)
    X_fit = np.asarray(data["X_fit"][:, idx], np.float32); y_fit = np.asarray(data["y_fit"])
    X_val = np.asarray(data["X_val"][:, idx], np.float32); y_val = np.asarray(data["y_val"])
    X_test = np.asarray(data["X_test"][:, idx], np.float32); y_test = np.asarray(data["y_test"])
    configure_tensorflow(cfg)
    spec = require_model(model_name)
    params = cfg.get("models", {}).get("params", {}).get(model_name, {}).copy()
    params.setdefault("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3))
    model = spec.builder(X_fit.shape[1], seed, params)
    Xf, Xv, Xt = spec.adapter(X_fit), spec.adapter(X_val), spec.adapter(X_test)
    from tensorflow import keras
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(cfg.get("training", {}).get("early_stopping_patience", 5)), restore_best_weights=True)]
    t0 = time.perf_counter()
    hist = model.fit(Xf, y_fit, validation_data=(Xv, y_val), epochs=int(cfg.get("training", {}).get("epochs", 30)), batch_size=int(cfg.get("training", {}).get("batch_size", 256)), verbose=0, callbacks=callbacks)
    train_sec = time.perf_counter() - t0
    val_prob = model.predict(Xv, batch_size=int(cfg.get("training", {}).get("batch_size", 256)), verbose=0).ravel()
    threshold = _choose_threshold(y_val, val_prob, cfg)
    for _ in range(warmup):
        _ = model.predict(Xt, batch_size=int(cfg.get("training", {}).get("batch_size", 256)), verbose=0)
    times=[]
    for _ in range(repeats):
        t0=time.perf_counter(); _=model.predict(Xt, batch_size=int(cfg.get("training", {}).get("batch_size", 256)), verbose=0); times.append(time.perf_counter()-t0)
    per_flow=[x/len(y_test)*1e6 for x in times]; throughput=[len(y_test)/x for x in times]
    lm,ls,lci,lmin,lmax=_stats(per_flow); tm,ts,tci,_,_=_stats(throughput)
    params_count = int(model.count_params()) if hasattr(model, "count_params") else np.nan
    keras.backend.clear_session()
    return {"dataset":dataset,"seed":seed,"fs_method":fs_method,"top_k":str(top_k),"dl_model":model_name,"n_features":len(idx),"training_time_sec":train_sec,"epochs_ran":len(hist.history.get("loss",[])),"threshold":threshold,"warmup_runs":warmup,"repeats":repeats,"latency_us_mean":lm,"latency_us_std":ls,"latency_us_ci95_half_width":lci,"latency_us_min":lmin,"latency_us_max":lmax,"throughput_mean":tm,"throughput_std":ts,"throughput_ci95_half_width":tci,"trainable_params":params_count}


def profile_final_candidates(cfg: Dict[str, Any], dataset: str):
    candidates = pd.read_csv(output_dir(cfg)/"07_efficiency"/dataset/"pareto_efficient_configurations.csv")
    seeds=[int(s) for s in cfg.get("runtime_profile",{}).get("seeds",cfg["experiment"]["seeds"])]
    warm=int(cfg.get("runtime_profile",{}).get("warmup_runs",5)); repeats=int(cfg.get("runtime_profile",{}).get("repeats",30))
    rows=[]
    for _,c in candidates.iterrows():
        for seed in seeds:
            rows.append(profile_candidate(cfg,dataset,seed,str(c.fs_method),str(c.top_k),str(c.dl_model),warm,repeats))
    out=output_dir(cfg)/"10_runtime"/dataset; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"runtime_profile_raw.csv",index=False)
    if rows:
        agg=pd.DataFrame(rows).groupby(["dataset","fs_method","top_k","dl_model"]).agg(latency_us_mean=("latency_us_mean","mean"),latency_us_between_seed_std=("latency_us_mean","std"),throughput_mean=("throughput_mean","mean"),training_time_sec_mean=("training_time_sec","mean")).reset_index()
        agg.to_csv(out/"runtime_profile_summary.csv",index=False)
    log(f"RUNTIME PROFILE | {dataset}: {len(rows)} trained profiles")
    return pd.DataFrame(rows)
