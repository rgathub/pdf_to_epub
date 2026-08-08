"""Modular facade for the repository's PDF to EPUB converter."""

from .config import ConversionOptions
from .orchestrator import PDFToEPUBConverter

__all__ = ["ConversionOptions", "PDFToEPUBConverter"]
