"""Modular facade for the repository's PDF to EPUB converter."""

from .config import ConversionOptions
from .orchestrator import PDFToEPUBConverter
from ._version import __version__

__all__ = ["ConversionOptions", "PDFToEPUBConverter", "__version__"]
