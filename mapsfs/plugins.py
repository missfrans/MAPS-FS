from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class ModelSpec:
    builder: Callable[..., Any]
    adapter: Callable[[Any], Any]


FEATURE_SELECTORS: Dict[str, Callable[..., Any]] = {}
MODELS: Dict[str, ModelSpec] = {}


def register_feature_selector(name: str):
    def deco(func):
        FEATURE_SELECTORS[name.lower()] = func
        return func
    return deco


def register_model(name: str, adapter: Callable[[Any], Any]):
    def deco(func):
        MODELS[name.lower()] = ModelSpec(builder=func, adapter=adapter)
        return func
    return deco


def load_plugins(module_names):
    for module_name in module_names or []:
        importlib.import_module(module_name)


def require_feature_selector(name: str):
    key = name.lower()
    if key not in FEATURE_SELECTORS:
        raise KeyError(f"Unknown feature selector '{name}'. Registered: {sorted(FEATURE_SELECTORS)}")
    return FEATURE_SELECTORS[key]


def require_model(name: str) -> ModelSpec:
    key = name.lower()
    if key not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}")
    return MODELS[key]
