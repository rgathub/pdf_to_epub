"""CLI-independent conversion orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ConversionOptions
from .reference import load_reference_module


class PDFToEPUBConverter:
    """Coordinate configuration, page processing, OCR, assembly, and validation.

    The bundled conversion engine remains the execution engine so that
    output semantics, OCR fallback, chapter detection, TOC anchors, validation,
    and worker limits stay byte-for-byte compatible while responsibilities are
    available as separate modules for future replacement.
    """

    def __init__(self, options: ConversionOptions | None = None, **kwargs: Any):
        if options is not None and kwargs:
            raise TypeError("pass either options or keyword options, not both")
        self.options = options or ConversionOptions(**kwargs)
        reference = load_reference_module()
        self._engine = reference.PDFToEPUBConverter(**self.options.as_kwargs())

    @classmethod
    def preflight_environment(cls, ocr_enabled: bool = True,
                              ocr_device: str = "auto") -> dict[str, Any]:
        """Check dependencies using the compatible implementation."""
        return load_reference_module().PDFToEPUBConverter.preflight_environment(
            ocr_enabled, ocr_device
        )

    def convert(self, pdf_path: str | Path, output_path: str | Path) -> bool:
        """Convert one PDF, preserving the reference behavior."""
        return self._engine.convert(str(pdf_path), str(output_path))

    @property
    def stats(self) -> dict[str, Any]:
        """Expose the latest conversion statistics."""
        return self._engine._stats
