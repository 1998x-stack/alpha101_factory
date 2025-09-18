# -*- coding: utf-8 -*-
from typing import Dict, Type
from .base import Factor

# dynamic discovery
import importlib
import pkgutil

_REGISTRY: Dict[str, Type[Factor]] = {}
_LOADED = False

def register(cls: Type[Factor]):
    _REGISTRY[cls.name] = cls
    return cls

def _ensure_loaded():
    global _LOADED
    if _LOADED:
        return
    # import all modules in alpha101_factory.factors (except base/registry)
    package = importlib.import_module(__package__)   # alpha101_factory.factors
    for _, mod_name, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg:
            continue
        if mod_name in ("base", "registry"):
            continue
        importlib.import_module(f"{__package__}.{mod_name}")
    _LOADED = True

def get_factor(name: str) -> Type[Factor]:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"Factor not found: {name}")
    return _REGISTRY[name]

def list_factors():
    _ensure_loaded()
    return sorted(_REGISTRY.keys())
