"""Command-line entry point with complete backward-compatible option handling."""

from __future__ import annotations

import sys

from .reference import load_reference_module


def main(argv: list[str] | None = None) -> int:
    """Run the established CLI parser and conversion workflow."""
    reference = load_reference_module()
    if argv is None:
        return reference.main()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return reference.main()
    finally:
        sys.argv = old_argv
