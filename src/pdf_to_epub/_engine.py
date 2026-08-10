"""Backward-compatible exports for the former monolithic engine."""

from .config import (
    _discard_temporary_output,
    _DECOMPRESSION_ERRORS,
    _IMAGE_ERRORS,
    _temporary_output_path,
    load_chapter_overrides as _load_chapter_overrides,
)
from .cli import main
from .orchestrator import (
    PDFToEPUBConverter,
    Image,
    epub,
    fitz,
    logger,
    _parallel_worker_state,
)
from .pdf_pages import _same_path

__all__ = [
    "PDFToEPUBConverter", "_same_path", "_temporary_output_path",
    "_discard_temporary_output", "_load_chapter_overrides", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
