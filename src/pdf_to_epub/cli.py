"""Command-line adapter for the modular converter."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None
try:
    from ebooklib import epub
except ImportError:
    epub = None

from .config import _discard_temporary_output, _temporary_output_path, load_chapter_overrides
from .epub_content import EPUBAssembler
from .orchestrator import PDFToEPUBConverter
from .pdf_pages import _same_path

def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to EPUB format with embedded images support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_epub.py -i document.pdf -o output.epub
  python pdf_to_epub.py -i doc1.pdf doc2.pdf doc3.pdf -o merged.epub
  python pdf_to_epub.py -i input.pdf --quality 90 --max-size 800x600
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        metavar="PDF",
        required=False,
        help="Input PDF file(s) to convert",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output.epub",
        help="Output EPUB file path (default: output.epub)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        metavar="N",
        help="Image quality for JPEG compression (1-100, default: 85)",
    )

    parser.add_argument(
        "--max-size",
        type=str,
        default="1200x1200",
        metavar="WxH",
        help="Maximum image dimensions in pixels (default: 1200x1200)",
    )

    parser.add_argument(
        "--ocr-lang",
        default="en",
        metavar="LANG",
        help="EasyOCR language code(s) for scanned pages (default: en)",
    )

    parser.add_argument(
        "--ocr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="OCR device selection (default: auto)",
    )

    parser.add_argument(
        "--ocr-confidence",
        type=float,
        default=0.0,
        metavar="N",
        help="Minimum OCR confidence from 0.0 to 1.0 (default: 0.0)",
    )

    parser.add_argument(
        "--ocr_confidence_threshold",
        type=float,
        default=0.4,
        metavar="N",
        help="OCR text retention threshold for space handling (default: 0.4)",
    )

    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--ocr",
        dest="ocr_enabled",
        action="store_true",
        help="Enable OCR fallback (default)",
    )
    ocr_group.add_argument(
        "--no-ocr",
        dest="ocr_enabled",
        action="store_false",
        help="Disable OCR fallback",
    )
    parser.set_defaults(ocr_enabled=True)

    parser.add_argument(
        "--ocr-preprocess",
        action="store_true",
        help="Apply grayscale and contrast preprocessing before OCR",
    )

    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--ocr-retry",
        dest="ocr_retry",
        action="store_true",
        help="Retry low-confidence OCR pages at higher DPI (default)",
    )
    retry_group.add_argument(
        "--no-ocr-retry",
        dest="ocr_retry",
        action="store_false",
        help="Disable low-confidence OCR retries",
    )
    parser.set_defaults(ocr_retry=True)

    parser.add_argument(
        "--ocr-retry-dpi",
        type=int,
        metavar="DPI",
        help="DPI for low-confidence OCR retries (default: 1.5x OCR DPI, at least 300)",
    )

    parser.add_argument(
        "--ocr-retry-confidence",
        type=float,
        default=0.35,
        metavar="N",
        help="Retry OCR below this average confidence (default: 0.35)",
    )

    parser.add_argument(
        "--ocr-cache-dir",
        metavar="DIR",
        help="Directory for per-page OCR JSON cache files",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse OCR cache files (requires --ocr-cache-dir or defaults beside output)",
    )

    parser.add_argument(
        "--layout",
        choices=("compact", "preserve"),
        default="compact",
        help="Text layout in EPUB pages (default: compact)",
    )

    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        metavar="DPI",
        help="Render resolution for OCR (default: 200)",
    )

    parser.add_argument(
        "--keep-scan-images",
        action="store_true",
        help="Keep full-page scan images after successful OCR",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run structural, XHTML, link, accessibility, and quality validation "
            "after writing (errors invalidate output; warnings are reported)"
        ),
    )

    parser.add_argument(
        "--report",
        metavar="JSON",
        help="Write conversion statistics, settings, and structured validation diagnostics to JSON",
    )

    parser.add_argument(
        "--repeated-header-action",
        "--running-header-action",
        dest="repeated_header_action",
        choices=("retain", "filter", "keep"),
        default="retain",
        help="Retain or filter repeated running headers (default: retain)",
    )

    parser.add_argument(
        "--chapter-overrides",
        metavar="JSON",
        help="JSON file (or config object) with exact chapter title/page mappings",
    )

    parser.add_argument(
        "--image-placement",
        choices=("position", "end"),
        default="position",
        help="Place images near their PDF position or append them (default: position)",
    )

    parser.add_argument(
        "--language",
        default="en",
        help="EPUB language metadata, or use the PDF language when present",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Page workers for single-PDF conversion (positive integer, default: 1); "
            "CPU OCR is capped at 2 workers and auto/CUDA OCR stays serial"
        ),
    )

    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable tqdm progress bar during conversion",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check dependencies, OCR availability, and GPU support, then exit",
    )

    parser.add_argument(
        "--config",
        metavar="JSON",
        help="Load CLI defaults from a JSON configuration file",
    )

    # Load config values as defaults; explicit command-line options still win.
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config")
    probe_args, _ = config_probe.parse_known_args()
    if probe_args.config:
        try:
            with Path(probe_args.config).open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            if not isinstance(config, dict):
                raise ValueError("configuration must be a JSON object")
            valid_dests = {
                action.dest for action in parser._actions if action.dest != "help"
            }
            parser.set_defaults(
                **{key: value for key, value in config.items() if key in valid_dests}
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Error: Could not load config '{probe_args.config}': {exc}")
            return 1

    args = parser.parse_args()

    required_preflight = PDFToEPUBConverter.preflight_environment(
        False, args.ocr_device
    )
    if not required_preflight["ready"]:
        missing = [
            name
            for name, available in required_preflight["required"].items()
            if not available
        ]
        print(
            "Error: missing required dependencies: "
            + ", ".join(missing)
            + ". Install pymupdf, ebooklib, and Pillow."
        )
        return 1

    try:
        chapter_overrides = load_chapter_overrides(args.chapter_overrides)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: Could not load chapter overrides: {exc}")
        return 1

    # Parse max size
    try:
        args.quality = int(args.quality)
        args.ocr_dpi = int(args.ocr_dpi)
        args.ocr_confidence = float(args.ocr_confidence)
        width, height = map(int, str(args.max_size).lower().split("x"))
    except (TypeError, ValueError):
        print(
            f"Error: Invalid max-size format '{args.max_size}'. Use WxH format (e.g., 1200x1200)"
        )
        return 1
    try:
        args.workers = int(args.workers)
    except (TypeError, ValueError):
        print("Error: workers must be a positive integer.")
        return 1
    if width <= 0 or height <= 0:
        print("Error: max-size dimensions must be positive.")
        return 1
    if args.ocr_dpi <= 0:
        print("Error: ocr-dpi must be positive.")
        return 1
    if not 0 <= args.ocr_confidence <= 1:
        print("Error: ocr-confidence must be between 0 and 1.")
        return 1
    if not 0 <= args.ocr_retry_confidence <= 1:
        print("Error: ocr-retry-confidence must be between 0 and 1.")
        return 1
    if args.ocr_retry_dpi is not None and args.ocr_retry_dpi <= 0:
        print("Error: ocr-retry-dpi must be positive.")
        return 1
    if not 1 <= args.quality <= 100:
        print("Error: quality must be between 1 and 100.")
        return 1
    if args.ocr_device not in {"auto", "cpu", "cuda"}:
        print("Error: ocr-device must be auto, cpu, or cuda.")
        return 1
    if args.layout not in {"compact", "preserve"}:
        print("Error: layout must be compact or preserve.")
        return 1
    if args.workers < 1:
        print("Error: workers must be a positive integer.")
        return 1
    if 0 <= args.ocr_confidence_threshold <= 1:
        # Validate threshold is in range
        pass
    else:
        print("Error: ocr-confidence-threshold must be between 0 and 1.")
        return 1

    if args.preflight:
        preflight = PDFToEPUBConverter.preflight_environment(
            args.ocr_enabled, args.ocr_device
        )
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["ready"] else 1

    if not args.input:
        print("Error: at least one input PDF is required.")
        return 1

    # Check if input files exist
    for pdf_file in args.input:
        if not os.path.isfile(pdf_file):
            print(f"Error: Input file not found: {pdf_file}")
            return 1
        if _same_path(pdf_file, args.output):
            print(f"Error: Output path must differ from input PDF: {pdf_file}")
            return 1

    cache_dir = args.ocr_cache_dir
    if args.resume and not cache_dir:
        cache_dir = str(Path(args.output).with_suffix(".ocr-cache"))

    # Convert single PDF or merge multiple PDFs
    converter = PDFToEPUBConverter(
        image_quality=args.quality,
        max_image_size=(width, height),
        ocr_enabled=args.ocr_enabled,
        ocr_language=args.ocr_lang,
        ocr_dpi=args.ocr_dpi,
        keep_scan_images=args.keep_scan_images,
        ocr_device=args.ocr_device,
        ocr_confidence=args.ocr_confidence,
        ocr_preprocess=args.ocr_preprocess,
        ocr_cache_dir=cache_dir,
        resume=args.resume,
        layout=args.layout,
        validate_output=args.validate,
        report_path=args.report,
        repeated_header_action=args.repeated_header_action,
        chapter_overrides=chapter_overrides,
        image_placement=args.image_placement,
        language=args.language,
        ocr_retry=args.ocr_retry,
        ocr_retry_dpi=args.ocr_retry_dpi,
        ocr_retry_confidence=args.ocr_retry_confidence,
        workers=args.workers,
        progress_bar=args.progress_bar,
        ocr_confidence_threshold=args.ocr_confidence_threshold,
    )

    if len(args.input) == 1:
        # Single file conversion
        success = converter.convert(args.input[0], args.output)
        return 0 if success else 1
    else:
        # Multiple files - merge into single EPUB
        merge_started = time.perf_counter()
        output_path = Path(args.output)

        epub_book = epub.EpubBook()
        epub_book.set_title("Merged PDF")
        epub_book.set_language(args.language)
        style = converter.add_styles(epub_book)

        chapters = []
        failed = False
        converter._page_metadata = {}
        converter._ocr_page_diagnostics = []
        converter._stats = converter._new_stats()
        converter._stats["metadata"] = {
            "title": "Merged PDF",
            "language": args.language,
        }

        for pdf_file in args.input:
            try:
                converter._current_pdf_path = str(Path(pdf_file).resolve())
                with fitz.open(pdf_file) as doc:
                    if not chapters:
                        converter._stats["metadata"] = converter._extract_pdf_metadata(
                            doc, pdf_file
                        )
                        converter.language = converter._stats["metadata"]["language"]
                        epub_book.set_title(converter._stats["metadata"]["title"])
                        epub_book.set_language(
                            converter._stats["metadata"]["language"]
                        )
                        if converter._stats["metadata"].get("author"):
                            epub_book.add_author(
                                converter._stats["metadata"]["author"]
                            )
                        converter._extract_cover(epub_book, doc)
                    for page in doc:
                        chapter = converter.process_page(
                            epub_book,
                            page,
                            len(chapters) + 1,
                        )
                        if chapter:
                            chapter.add_item(style)
                            chapters.append(chapter)
            except (OSError, RuntimeError, ValueError) as e:
                failed = True
                print(f"Warning: Could not process {pdf_file}: {e}")
                traceback.print_exc()

        converter._apply_repeated_header_policy(chapters)
        converter._apply_book_semantics(chapters)
        for chapter in chapters:
            epub_book.add_item(chapter)

        converter.add_toc(epub_book, chapters)
        converter._stats["chapters"] = len(converter._identify_chapters(chapters))

        temporary_output: Path | None = None
        try:
            temporary_output = _temporary_output_path(output_path)
            EPUBAssembler(args.quality).write(temporary_output, epub_book)
        except (OSError, RuntimeError, ValueError) as e:
            _discard_temporary_output(temporary_output)
            print(f"Error writing merged EPUB '{output_path}': {e}")
            return 1

        validation_failed = False
        if args.validate:
            validation = converter.validate_epub_diagnostics(temporary_output)
            converter._stats["validation"] = validation
            if not validation["valid"]:
                validation_failed = True
                print(
                    f"Error: EPUB validation failed: {output_path} "
                    f"({len(validation['errors'])} error(s), "
                    f"{len(validation['warnings'])} warning(s))"
                )
        if validation_failed:
            _discard_temporary_output(temporary_output)
            temporary_output = None
        else:
            try:
                os.replace(temporary_output, output_path)
                temporary_output = None
            except OSError as exc:
                _discard_temporary_output(temporary_output)
                print(f"Error writing merged EPUB '{output_path}': {exc}")
                return 1
        converter._stats["elapsed_seconds"] = round(
            time.perf_counter() - merge_started, 3
        )
        if args.report:
            converter._write_report(
                ", ".join(args.input),
                output_path,
            )
        if failed or validation_failed:
            return 1
        print(f"Successfully merged {len(args.input)} PDF(s) into: {output_path}")
        return 0

    return 0
