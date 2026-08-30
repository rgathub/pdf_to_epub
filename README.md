# PDF to EPUB

Modular Python code for converting PDF documents to EPUB files.

## Requirements

- Python 3.11 or newer
- `pymupdf`
- `ebooklib`
- `Pillow`
- `easyocr` and its dependencies for OCR-enabled conversion
- `numpy` (required by EasyOCR)
- CUDA-enabled `torch`, `torchvision`, and `torchaudio` for GPU OCR
- `defusedxml` for safe EPUB XML validation

The `requirements.txt` file installs all GPU-accelerated PyTorch dependencies.
Use a compatible NVIDIA driver, or replace the PyTorch index URL with the
CUDA version appropriate for the target machine. `requirements-cpu.txt`
installs OCR without CUDA-specific PyTorch wheels.

## Setup

Create and activate the virtual environment from the repository directory:

```powershell
cd D:\Code\pdf_to_epub
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a normal package installation, install the project itself:

```powershell
python -m pip install .
```

For CPU-only OCR (no GPU PyTorch), use:

```powershell
python -m pip install -r requirements-cpu.txt
```

## Usage

### Command-line Examples

```powershell
python -m pdf_to_epub --help
python -m pdf_to_epub -i input.pdf -o output.epub --no-ocr
python -m pdf_to_epub.self_check
```

After installation, the equivalent console command is:

```powershell
pdf-to-epub -i input.pdf -o output.epub --no-ocr
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

### Progress Bar

Enable a visual progress bar during conversion:

```powershell
python -m pdf_to_epub -i input.pdf -o output.epub --progress-bar
```

The progress bar shows real-time conversion progress with elapsed time,
remaining pages, and throughput rate. It automatically closes when complete.

### Installation Options

Install the progress bar dependency explicitly:

```powershell
python -m pip install tqdm
```

Or use the optional dependency group:

```powershell
python -m pip install pdf-to-epub[progress]
```

The `--progress-bar` option is available by default when tqdm is installed.

The modular API is available through `PDFToEPUBConverter` and
`ConversionOptions`:

```python
from pdf_to_epub import ConversionOptions, PDFToEPUBConverter

options = ConversionOptions(ocr_enabled=False)
converter = PDFToEPUBConverter(**options.as_kwargs())
```

Library callers can receive structured page-progress events:

```python
events = []
options = ConversionOptions(
    ocr_enabled=False,
    progress_callback=events.append,
)
```

The orchestrator also emits standard-library logging records through the
`pdf_to_epub.orchestrator` logger.

## Project layout

- `src\pdf_to_epub\config.py` - conversion options and chapter override parsing
- `src\pdf_to_epub\ocr.py` - OCR and cache service boundary
- `src\pdf_to_epub\pdf_pages.py` - lazy PDF page extraction
- `src\pdf_to_epub\epub_content.py` - EPUB writing boundary
- `src\pdf_to_epub\validation.py` - structural EPUB diagnostics
- `src\pdf_to_epub\orchestrator.py` - conversion API and lifecycle coordination
- `src\pdf_to_epub\cli.py` - command-line adapter
- `src\pdf_to_epub\_engine.py` - backward-compatible exports for the modular implementation
- `src\pdf_to_epub\reference.py` - legacy compatibility loader for the engine shim
- `src\pdf_to_epub\self_check.py` - package self-check entry point
- `pyproject.toml` - build metadata and console-script configuration

## Validation

The modular converter was tested against the PDF in `test\` using CUDA OCR.
The 348-page conversion completed successfully with 348 OCR pages, zero OCR
failures, and one low-confidence retry. The generated EPUB passed structural
validation with zero errors and warnings, a clean ZIP integrity check, and a
valid EPUB mimetype.

Conversions write to a temporary file in the destination directory and
replace the requested output only after writing and optional validation
complete successfully.

## Building a wheel

```powershell
python -m pip install build
python -m build
```

The distributable wheel and source archive are written to `dist\`.

The package version is defined in `src\pdf_to_epub\_version.py`. Tagging a
commit such as `v0.1.0` triggers the PyPI release workflow; configure PyPI
trusted publishing for the repository before creating a release tag.

## Testing

Run the dependency-light test suite:

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests and builds the distributions on Windows for
every push and pull request.
