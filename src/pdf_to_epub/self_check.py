"""Lightweight dependency-free self-check for package wiring."""

from .config import ConversionOptions
from .orchestrator import PDFToEPUBConverter


def run() -> None:
    options = ConversionOptions(ocr_enabled=False)
    converter = PDFToEPUBConverter(options)
    assert converter.options.ocr_enabled is False
    assert isinstance(converter.preflight_environment(False)["ready"], bool)


if __name__ == "__main__":
    run()
    print("self-check passed")
