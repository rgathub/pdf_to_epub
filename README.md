# PDF to EPUB

Modular Python code for converting PDF documents to EPUB files.

## Requirements

- Python 3.11 or newer
- `pymupdf`
- `ebooklib`
- `Pillow`
- `easyocr` and its dependencies for OCR-enabled conversion
- CUDA-enabled `torch`, `torchvision`, and `torchaudio` for GPU OCR

`requirements.txt` targets CUDA 12.8-enabled PyTorch wheels for GPU OCR.
Use a compatible NVIDIA driver, or replace the PyTorch index URL with the
CUDA version appropriate for the target machine.

## Setup

Create and activate the virtual environment from the repository directory:

```powershell
cd D:\Code\pdf_to_epub
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The package is located in the repository root, so run module commands from
its parent directory:

```powershell
cd D:\Code
```

## Usage

```powershell
python -m pdf_to_epub --help
python -m pdf_to_epub -i input.pdf -o output.epub --no-ocr
python -m pdf_to_epub.self_check
```

Run OCR on a CUDA-capable NVIDIA GPU:

```powershell
python -m pdf_to_epub `
  -i input.pdf `
  -o output.epub `
  --ocr `
  --ocr-device cuda `
  --validate `
  --report conversion-report.json
```

Use `--ocr-device cpu` for CPU OCR. Use `--no-ocr` when the PDF already
contains extractable text and OCR is unnecessary.

The modular API is available through `PDFToEPUBConverter` and
`ConversionOptions`:

```python
from pdf_to_epub import ConversionOptions, PDFToEPUBConverter

options = ConversionOptions(ocr_enabled=False)
converter = PDFToEPUBConverter(**options.as_kwargs())
```

## Project layout

- `config.py` - conversion options and chapter override parsing
- `ocr.py` - OCR and cache service boundary
- `pdf_pages.py` - lazy PDF page extraction
- `epub_content.py` - EPUB writing boundary
- `validation.py` - structural EPUB diagnostics
- `orchestrator.py` - conversion API and lifecycle coordination
- `cli.py` - command-line adapter
- `_engine.py` - bundled conversion engine
- `reference.py` - compatibility loader for the bundled engine
- `self_check.py` - package self-check entry point

## Review findings

The compatibility loader in `reference.py` now resolves to the bundled
engine. Commands use `pdf_to_epub`, not `pdf_to_epub_refactored`.

## Validation

The bundled engine was tested against the PDF in `test\` using CUDA OCR.
The 348-page conversion completed successfully with 348 OCR pages, zero OCR
failures, and one low-confidence retry. The generated EPUB passed structural
validation with zero errors and warnings, a clean ZIP integrity check, and a
valid EPUB mimetype.
