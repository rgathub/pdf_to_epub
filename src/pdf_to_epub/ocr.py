"""OCR responsibility boundary.

The bundled engine owns EasyOCR lifecycle, preprocessing, retries, and cache
serialization. This small service keeps that concern injectable for callers
that need to inspect or replace OCR independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class OCRService:
    """Describe OCR settings and cache location for a conversion."""

    def __init__(self, enabled: bool = True, cache_dir: str | None = None):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def diagnostics(self) -> dict[str, Any]:
        """Return non-invasive OCR service diagnostics."""
        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
