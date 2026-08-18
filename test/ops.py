"""Operator loading: config.OPERATORS → each op folder's op.py module."""

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Dict, List

from . import config

# Repo root = parent of this package.
ROOT = pathlib.Path(config.__file__).resolve().parent.parent

_LOADED: Dict[str, ModuleType] = {}

_REQUIRED = ("NAME", "DEFAULT_SIZES", "gen_inputs", "metric", "register")
_DEFAULTS = {
    "SIZE_LABEL": "Size",
    "SIZE_COLS": ["Size"],
    "METRIC_LABEL": "Metric",
    "METRIC_FMT": ".3f",
    "size_key": lambda size: size[0],
    "reference": None,
    "prepare": lambda: None,
}


def list_ops() -> List[str]:
    """Names of operators registered in config.OPERATORS."""
    return list(config.OPERATORS)


def load_op(name: str, reload: bool = False) -> ModuleType:
    """Import an operator's op.py (from config.OPERATORS) and register it.

    Validates the required protocol attributes, fills optional defaults,
    then calls module.register(). Modules are cached; reload=True forces
    re-execution (e.g. after editing op.py).
    """
    if name in _LOADED and not reload:
        return _LOADED[name]
    if name not in config.OPERATORS:
        raise KeyError(f"Unknown operator: {name!r}. Available: {list_ops()}. "
                       f"Add it to config.OPERATORS.")

    src = ROOT / config.OPERATORS[name] / "op.py"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — create an op.py for {name!r} "
            f"(see test/README.md, section 2).")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    spec = importlib.util.spec_from_file_location(f"test_ops.{name}", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = [attr for attr in _REQUIRED if not hasattr(module, attr)]
    if missing:
        raise AttributeError(f"op.py of {name!r} is missing required "
                             f"attribute(s): {missing} (see test/README.md).")
    for attr, default in _DEFAULTS.items():
        if not hasattr(module, attr):
            setattr(module, attr, default)

    module.register()
    _LOADED[name] = module
    print(f"Loaded operator '{name}': {list_implementations_from(module)}")
    return module


def list_implementations_from(module: ModuleType) -> List[str]:
    """Helper used by load_op's log line (kept import-light)."""
    from .registry import list_implementations
    return list_implementations(module.NAME)
