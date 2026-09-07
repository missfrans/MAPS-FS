"""Example extension points for researchers.

Enable this module in YAML:
    plugins: [plugins.example_custom]
Then add `variance_rank` or `tiny_dnn` to the enabled lists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mapsfs.plugins import register_feature_selector, register_model


@register_feature_selector("variance_rank")
def variance_rank(X, y, feature_names, seed, params):
    scores = np.var(np.asarray(X), axis=0)
    order = np.argsort(-scores)
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "feature_index": order,
        "feature_name": [feature_names[i] for i in order],
        "score": scores[order],
        "method_note": "Example plugin: variance ranking; not recommended as a default IDS selector",
    })


def flat_adapter(X):
    return np.asarray(X, dtype=np.float32)


@register_model("tiny_dnn", flat_adapter)
def tiny_dnn(n_features, seed, params):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(seed)
    inp = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(int(params.get("hidden_units", 32)), activation="relu")(inp)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(float(params.get("learning_rate", 1e-3))), loss="binary_crossentropy", metrics=["accuracy"])
    return model
