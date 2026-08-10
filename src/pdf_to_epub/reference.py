"""Compatibility loader for legacy callers of the modular implementation."""

from __future__ import annotations

from types import ModuleType


def load_reference_module() -> ModuleType:
    """Return the backward-compatible engine shim."""
    from . import _engine

    return _engine
