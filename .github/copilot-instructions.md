# Copilot instructions for `pdf_to_epub`

## Project shape

This is a Python 3.11+ `src`-layout package that converts PDFs to EPUBs. The
public API is `PDFToEPUBConverter` plus `ConversionOptions` from
`pdf_to_epub.__init__`; the CLI is exposed both as `python -m pdf_to_epub` and
the `pdf-to-epub` console script.

The conversion flow is composed from independent responsibility modules:

- `orchestrator.py` owns public lifecycle coordination, worker management,
  reporting, chapter discovery, and composition of the processing mixins.
- `config.py` owns conversion options, temporary-output helpers, and chapter
  override parsing.
- `pdf_pages.py` owns PyMuPDF page access, metadata, native text extraction,
  page-level image extraction, and page/chapter processing.
- `ocr.py` owns EasyOCR lifecycle, preprocessing, retries, diagnostics, and
  per-page cache handling.
- `epub_content.py` owns XHTML/CSS rendering, image encoding, EPUB assembly,
  and book-level content semantics.
- `validation.py` owns structural EPUB, XML, resource-link, accessibility,
  and quality diagnostics.
- `cli.py` owns argument/config parsing and calls the public orchestrator;
  `__main__.py` and the console-script entry point call it.
- `_engine.py` and `reference.py` exist only as legacy compatibility exports.
  New code should import the public package or responsibility modules instead.

The converter processes pages lazily, optionally performs EasyOCR, assembles EPUB
chapters and TOC metadata, and records JSON-compatible statistics. OCR cache
and resume behavior is per PDF/page. Page workers are opt-in: serial processing
is the default, CPU OCR is capped at two workers, and CUDA/auto OCR remains
serial because OCR readers are not safely shared across threads. Multi-input
merging remains serial.

Output files are written to a temporary file beside the requested destination,
optionally validated, and only then replaced into place. Preserve this
failure-safe output behavior when modifying conversion or validation code.

## Commands

Run from the repository root in PowerShell. The documented development
environment uses `.venv` and either CPU or CUDA OCR dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
```

Use `requirements-gpu.txt` instead for the pinned CUDA 12.8 PyTorch wheels.
For package installation and the CI-equivalent setup:

```powershell
python -m pip install .
```

Run the dependency-light test suite:

```powershell
python -m unittest discover -s tests -v
```

Run one test or one test class:

```powershell
python -m unittest tests.test_package.PackageTests.test_validator_rejects_non_epub -v
python -m unittest tests.test_package.PackageTests -v
```

Build distributions:

```powershell
python -m pip install build
python -m build
```

There is no configured lint command in the repository. Do not introduce a new
linter as part of routine changes unless the project configuration is updated
deliberately.

Useful smoke checks and CLI forms:

```powershell
python -m pdf_to_epub.self_check
python -m pdf_to_epub --help
python -m pdf_to_epub -i input.pdf -o output.epub --no-ocr
```

## Repository-specific conventions

- Keep optional/heavy dependencies such as EasyOCR imported lazily where the
  existing code does so, so no-OCR usage and preflight checks remain usable
  without the OCR runtime.
- Keep Windows path semantics in mind. Path comparisons are absolute and
  case-insensitive, and the CI/test/release workflows run on Windows.
- Use `ConversionOptions` for library-facing option groups and retain the
  engine's option names and defaults when adding or changing options.
- Progress callbacks receive structured dictionaries such as
  `{"event": "page", "page": ..., "pages": ..., "percent": ...}`; preserve
  this shape for callers.
- Conversion statistics are stored on the converter as JSON-compatible
  dictionaries. Update the existing counters/reporting structures when adding
  conversion stages rather than introducing incompatible result objects.
- EPUB validation uses `defusedxml` and is part of the conversion lifecycle
  when `validate_output` is enabled. Keep validation structural and fail-safe.
- Tests use the standard library `unittest` runner and are dependency-light;
  test package wiring, compatibility behavior, path semantics, callbacks, and
  validation without requiring large PDF fixtures or GPU OCR.
- The version is defined in `src/pdf_to_epub/_version.py`. A `v*` tag triggers
  the PyPI release workflow after `python -m build`.
