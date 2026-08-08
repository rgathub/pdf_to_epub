"""EPUB validation responsibility boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EPUBValidator:
    """Run the bundled converter's structural EPUB diagnostics."""

    def validate(self, output_path: str | Path) -> dict[str, Any]:
        from .reference import load_reference_module

        converter = load_reference_module().PDFToEPUBConverter(ocr_enabled=False)
        return converter.validate_epub_diagnostics(output_path)
