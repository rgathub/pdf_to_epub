"""EPUB assembly responsibility boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EPUBAssembler:
    """Write EPUB output using the bundled engine's assembly behavior."""

    def __init__(self, image_quality: int = 85):
        self.image_quality = image_quality

    def write(self, output_path: str | Path, book: Any) -> None:
        """Write an ebooklib book with the same image quality setting."""
        from .reference import load_reference_module

        epub = load_reference_module().epub
        epub.write_epub(
            str(output_path), book,
            {"image_quality": self.image_quality, "epub3_landmark": True},
        )
