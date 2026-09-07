from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from .plugins import register_feature_selector
from .utils import set_seed


def _result(feature_names: List[str], scores: np.ndarray, descending: bool = True, note: str = "") -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores if descending else scores, kind="mergesort")
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "feature_index": order,
        "feature_name": [feature_names[i] for i in order],
        "score": scores[order],
        "method_note": note,
    })


@register_feature_selector("rfe")
def rank_rfe(X, y, feature_names, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    estimator = LogisticRegression(
        max_iter=int(params.get("max_iter", 3000)),
        solver=params.get("solver", "liblinear"),
        random_state=seed,
    )
    selector = RFE(estimator=estimator, n_features_to_select=1, step=int(params.get("step", 1)))
    selector.fit(X, y)
    # RFE ranking: 1 = best. Convert to a score where larger is better.
    score = (len(feature_names) + 1) - selector.ranking_.astype(float)
    return _result(feature_names, score, True, "RFE logistic-regression ranking")


@register_feature_selector("lasso")
def rank_lasso(X, y, feature_names, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    model = LogisticRegression(
        penalty="l1",
        solver=params.get("solver", "saga"),
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 5000)),
        n_jobs=params.get("n_jobs", None),
        random_state=seed,
    )
    model.fit(X, y)
    score = np.abs(model.coef_).ravel()
    return _result(feature_names, score, True, "Absolute L1-logistic coefficient magnitude")


@register_feature_selector("rf")
def rank_rf(X, y, feature_names, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    model = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth", None),
        class_weight=params.get("class_weight", None),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=seed,
    )
    model.fit(X, y)
    return _result(feature_names, model.feature_importances_, True, "Random-forest impurity importance")


@register_feature_selector("mrmr")
def rank_mrmr(X, y, feature_names, seed: int, params: Dict[str, Any]):
    """Greedy mRMR-style ranking using MI relevance and absolute-correlation redundancy.

    This is intentionally dependency-light and deterministic. Researchers who need
    a different mRMR implementation can register it as a plugin without changing
    the MAPS-FS orchestration.
    """
    set_seed(seed)
    relevance = mutual_info_classif(X, y, random_state=seed)
    corr = np.corrcoef(np.asarray(X, dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    weight = float(params.get("redundancy_weight", 1.0))

    selected: List[int] = []
    remaining = set(range(X.shape[1]))
    selection_scores = np.zeros(X.shape[1], dtype=float)
    for position in range(X.shape[1]):
        best_idx, best_score = None, -np.inf
        for j in sorted(remaining):
            redundancy = float(np.mean(corr[j, selected])) if selected else 0.0
            score = float(relevance[j]) - weight * redundancy
            if score > best_score:
                best_idx, best_score = j, score
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
        # Preserve selection order as primary ranking signal; raw greedy score remains informative.
        selection_scores[int(best_idx)] = best_score
    df = pd.DataFrame({
        "rank": np.arange(1, len(selected) + 1),
        "feature_index": selected,
        "feature_name": [feature_names[i] for i in selected],
        "score": [selection_scores[i] for i in selected],
        "method_note": "Greedy MI relevance minus absolute-correlation redundancy",
    })
    return df


@register_feature_selector("autoencoder")
def rank_autoencoder(X, y, feature_names, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required for the autoencoder selector.") from exc

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[1]
    hidden = int(params.get("hidden_units", max(8, min(64, n // 2))))
    bottleneck = int(params.get("bottleneck_units", max(4, min(32, n // 4))))
    inp = keras.Input(shape=(n,))
    h = keras.layers.Dense(hidden, activation="relu", name="encoder_input_weights")(inp)
    z = keras.layers.Dense(bottleneck, activation="relu")(h)
    h2 = keras.layers.Dense(hidden, activation="relu")(z)
    out = keras.layers.Dense(n, activation="linear")(h2)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(float(params.get("learning_rate", 1e-3))), loss="mse")
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=int(params.get("patience", 5)), restore_best_weights=True
    )]
    model.fit(
        X, X,
        validation_split=float(params.get("validation_split", 0.10)),
        epochs=int(params.get("epochs", 30)),
        batch_size=int(params.get("batch_size", 256)),
        verbose=0,
        callbacks=callbacks,
    )
    weights = model.get_layer("encoder_input_weights").get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    result = _result(feature_names, importance, True, "Autoencoder first-layer input-weight importance; no latent substitution")
    keras.backend.clear_session()
    return result
