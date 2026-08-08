"""Compatibility loader for the bundled conversion engine."""

from __future__ import annotations

from types import ModuleType


def load_reference_module() -> ModuleType:
    """Return the bundled engine kept behind the legacy compatibility API."""
    from . import _engine

    return _engine
