"""Configuration and statistics models for PDF to EPUB conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ConversionOptions:
    """Options accepted by the bundled converter."""

    image_quality: int = 85
    max_image_size: tuple[int, int] = (1200, 1200)
    ocr_enabled: bool = True
    ocr_language: str = "en"
    ocr_dpi: int = 200
    keep_scan_images: bool = False
    ocr_device: str = "auto"
    ocr_confidence: float = 0.0
    ocr_preprocess: bool = False
    ocr_cache_dir: str | None = None
    resume: bool = False
    layout: str = "compact"
    validate_output: bool = False
    report_path: str | None = None
    repeated_header_action: str = "retain"
    chapter_overrides: dict[int, str] = field(default_factory=dict)
    image_placement: str = "position"
    language: str = "en"
    ocr_retry: bool = True
    ocr_retry_dpi: int | None = None
    ocr_retry_confidence: float = 0.35
    workers: int = 1
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Return constructor keyword arguments for the compatible engine."""
        return {
            key: value for key, value in self.__dict__.items()
            if key != "max_image_size" or value is not None
        }


def load_chapter_overrides(value: Any) -> dict[int, str]:
    """Parse chapter overrides using the bundled converter's format."""
    from .reference import load_reference_module

    return load_reference_module()._load_chapter_overrides(value)


class ConversionStats(dict):
    """Named marker for the reference engine's JSON-compatible statistics."""
