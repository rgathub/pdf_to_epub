"""Configuration models and conversion-independent helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

try:
    from PIL import Image
except ImportError:
    Image = None

_DECOMPRESSION_ERRORS = (Image.DecompressionBombError,) if Image is not None else ()
_IMAGE_ERRORS = (OSError, RuntimeError, ValueError) + _DECOMPRESSION_ERRORS


def _temporary_output_path(output_path: str | Path) -> Path:
    destination = Path(output_path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    return Path(temporary)


def _discard_temporary_output(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class ConversionOptions:
    """Options accepted by the public converter."""
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
    ocr_confidence_threshold: float = 0.4  # Optimized for better space handling

    def as_kwargs(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_chapter_overrides(value: Any) -> dict[int, str]:
    """Load exact chapter page/title mappings from JSON or a config object."""
    if not value:
        return {}
    if isinstance(value, (str, os.PathLike)):
        with Path(value).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if isinstance(value, dict) and "chapters" in value:
        value = value["chapters"]
    entries = []
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).isdigit() and isinstance(item, str):
                entries.append({"page": key, "title": item})
            elif isinstance(item, int) and isinstance(key, str):
                entries.append({"page": item, "title": key})
            else:
                raise ValueError("chapter overrides must map page numbers to titles or titles to page numbers")
    else:
        raise ValueError("chapter overrides must be a JSON object or list")
    overrides = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each chapter override must be an object")
        page = entry.get("page")
        title = entry.get("title")
        if isinstance(page, bool) or not isinstance(page, int):
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError("chapter override pages must be integers") from exc
        if page < 1 or not isinstance(title, str) or not title.strip():
            raise ValueError("each chapter override requires a positive page and title")
        overrides[page] = title.strip()
    return overrides


class ConversionStats(dict):
    """Named marker for JSON-compatible conversion statistics."""
