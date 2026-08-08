"""PDF page extraction responsibility boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


class PDFPageExtractor:
    """Lazily expose page count and page iteration through PyMuPDF."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

    def page_count(self) -> int:
        """Return the number of pages without retaining a document handle."""
        from .reference import load_reference_module

        fitz = load_reference_module().fitz
        with fitz.open(str(self.pdf_path)) as document:
            return document.page_count

    def pages(self) -> Iterator[object]:
        """Yield PyMuPDF page objects while the document is open."""
        from .reference import load_reference_module

        fitz = load_reference_module().fitz
        document = fitz.open(str(self.pdf_path))
        try:
            yield from document
        finally:
            document.close()
