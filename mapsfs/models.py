from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .plugins import register_model
from .utils import set_seed


def flat_adapter(X):
    return np.asarray(X, dtype=np.float32)


def sequence_adapter(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape((X.shape[0], X.shape[1], 1))


def _keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    except Exception as exc:
        raise ImportError("TensorFlow is required for the built-in deep-learning models.") from exc


def _compile(model, params):
    _, keras = _keras()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(params.get("learning_rate", 1e-3))),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        jit_compile=bool(params.get("jit_compile", False)),
    )
    return model


@register_model("cnn", sequence_adapter)
def build_cnn(n_features: int, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    _, keras = _keras()
    filters = list(params.get("filters", [32, 64]))
    kernel = int(params.get("kernel_size", 3))
    dense = int(params.get("dense_units", 128))
    dropout = float(params.get("dropout", 0.5))
    inp = keras.Input(shape=(n_features, 1), name="feature_axis")
    x = inp
    for i, f in enumerate(filters):
        x = keras.layers.Conv1D(int(f), kernel, padding="same", activation="relu", name=f"conv_{i+1}")(x)
        if i == 0:
            x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(dense, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inp, out, name="CNN_IDS"), params)


@register_model("rnn", sequence_adapter)
def build_rnn(n_features: int, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    _, keras = _keras()
    units = int(params.get("units", 64))
    dense = int(params.get("dense_units", 128))
    dropout = float(params.get("dropout", 0.5))
    inp = keras.Input(shape=(n_features, 1), name="feature_axis")
    x = keras.layers.SimpleRNN(units, activation="tanh")(inp)
    x = keras.layers.Dense(dense, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inp, out, name="RNN_IDS"), params)


@register_model("lstm", sequence_adapter)
def build_lstm(n_features: int, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    _, keras = _keras()
    units = int(params.get("units", 64))
    dense = int(params.get("dense_units", 128))
    dropout = float(params.get("dropout", 0.5))
    inp = keras.Input(shape=(n_features, 1), name="feature_axis")
    x = keras.layers.LSTM(units)(inp)
    x = keras.layers.Dense(dense, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inp, out, name="LSTM_IDS"), params)


@register_model("dnn", flat_adapter)
def build_dnn(n_features: int, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    _, keras = _keras()
    units = list(params.get("hidden_units", [128, 64]))
    dropout = float(params.get("dropout", 0.5))
    inp = keras.Input(shape=(n_features,), name="features")
    x = inp
    for u in units:
        x = keras.layers.Dense(int(u), activation="relu")(x)
        x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inp, out, name="DNN_IDS"), params)


@register_model("mlp", flat_adapter)
def build_mlp(n_features: int, seed: int, params: Dict[str, Any]):
    set_seed(seed)
    _, keras = _keras()
    units = list(params.get("hidden_units", [128, 64, 32]))
    dropout = list(params.get("dropouts", [0.5, 0.5, 0.3]))
    if len(dropout) < len(units):
        dropout += [dropout[-1] if dropout else 0.0] * (len(units) - len(dropout))
    inp = keras.Input(shape=(n_features,), name="features")
    x = inp
    for u, d in zip(units, dropout):
        x = keras.layers.Dense(int(u), activation="relu")(x)
        if float(d) > 0:
            x = keras.layers.Dropout(float(d))(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inp, out, name="MLP_IDS"), params)
