#!/usr/bin/env python3
"""
PDF to EPUB Converter with Embedded Images Support

This script converts PDF files to EPUB format while preserving embedded images.
It uses PyMuPDF (fitz) for PDF parsing and ebooklib for EPUB creation.

Requirements:
    pip install pymupdf ebooklib Pillow easyocr
    EasyOCR downloads its recognition models on first use.

Usage:
    python pdf_to_epub.py -i input.pdf -o output.epub
    python pdf_to_epub.py -i pdf1.pdf pdf2.pdf -o merged.epub
    python pdf_to_epub.py -i input.pdf -o output.epub --workers 2 --ocr-device cpu

Page parallelism is opt-in (workers defaults to 1). Each page worker owns its
own PyMuPDF handle, converter, and OCR reader. To avoid excessive model memory,
CPU OCR is capped at two workers; auto/CUDA OCR remains serial because GPU OCR
readers are not safely shared across threads. The workers option applies only
to single-PDF conversion; multi-input merging remains serial.
"""

# The converter keeps EPUB assembly, OCR, image processing, and CLI handling
# together because they share the same per-book state and lifecycle.
# pylint: disable=import-outside-toplevel,no-else-return
# pylint: disable=too-many-arguments,too-many-branches,too-many-instance-attributes
# pylint: disable=too-many-locals,too-many-positional-arguments
# pylint: disable=too-many-return-statements,too-many-statements

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
import posixpath
import re
import sys
import threading
import traceback
import zipfile
import importlib.util
import platform
import time
from urllib.parse import unquote, urlsplit
try:
    import defusedxml.ElementTree as ElementTree
except ImportError:
    from xml.etree import ElementTree

from io import BytesIO
from pathlib import Path


_parallel_worker_state = threading.local()

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from ebooklib import epub
except ImportError:
    epub = None

try:
    from PIL import Image
except ImportError:
    Image = None


class PDFToEPUBConverter:
    """Converts PDF files to EPUB format with embedded images support."""

    def __init__(
        self,
        image_quality: int = 85,
        max_image_size: tuple[int, int] = (1200, 1200),
        ocr_enabled: bool = True,
        ocr_language: str = "en",
        ocr_dpi: int = 200,
        keep_scan_images: bool = False,
        ocr_device: str = "auto",
        ocr_confidence: float = 0.0,
        ocr_preprocess: bool = False,
        ocr_cache_dir: str | None = None,
        resume: bool = False,
        layout: str = "compact",
        validate_output: bool = False,
        report_path: str | None = None,
        repeated_header_action: str = "retain",
        chapter_overrides: dict[int, str] | None = None,
        image_placement: str = "position",
        language: str = "en",
        ocr_retry: bool = True,
        ocr_retry_dpi: int | None = None,
        ocr_retry_confidence: float = 0.35,
        workers: int = 1,
    ):
        """
        Initialize the converter.

        Args:
            image_quality: JPEG quality for embedded images (1-100)
            max_image_size: Maximum width and height for images in EPUB
            ocr_enabled: Run OCR when a page has no extractable text
            ocr_language: EasyOCR language code(s), such as "en" or "en+hi"
            ocr_dpi: Resolution used to render pages for OCR
            keep_scan_images: Keep full-page scan images after successful OCR
            ocr_device: OCR device: auto, cpu, or cuda
            ocr_confidence: Minimum EasyOCR confidence to keep
            ocr_preprocess: Apply conservative grayscale/contrast preprocessing
            ocr_cache_dir: Optional directory for per-page OCR JSON cache files
            resume: Reuse cached OCR results when available
            layout: Output layout, compact or preserve
            validate_output: Validate the generated EPUB archive
            report_path: Optional JSON conversion report path
            repeated_header_action: Retain or filter repeated running headers
            chapter_overrides: Exact page-number to chapter-title mappings
            image_placement: Place images by page position or append them
            language: EPUB language metadata (default: en)
            ocr_retry: Retry low-confidence OCR pages at higher quality
            ocr_retry_dpi: DPI used for OCR retries (default: ocr_dpi * 1.5)
            ocr_retry_confidence: Retry when average OCR confidence is below this
            workers: Number of page workers for single-PDF conversion; 1 is serial
        """
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.image_quality = image_quality
        self.max_image_size = max_image_size
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.keep_scan_images = keep_scan_images
        self.ocr_device = ocr_device.casefold()
        self.ocr_confidence = ocr_confidence
        self.ocr_preprocess = ocr_preprocess
        self.ocr_cache_dir = Path(ocr_cache_dir) if ocr_cache_dir else None
        self.resume = resume
        self.layout = layout
        self.validate_output = validate_output
        self.report_path = report_path
        self.repeated_header_action = (
            "retain" if repeated_header_action == "keep" else repeated_header_action
        )
        self.chapter_overrides = chapter_overrides or {}
        self.image_placement = image_placement
        self.language = language or "en"
        self.ocr_retry = ocr_retry
        self.ocr_retry_dpi = ocr_retry_dpi or max(300, int(ocr_dpi * 1.5))
        self.ocr_retry_confidence = ocr_retry_confidence
        self.workers = workers
        self._ocr_reader = None
        self._ocr_gpu = False
        self._current_pdf_path = None
        self._ocr_page_diagnostics = []
        self._last_ocr_segments = []
        self._page_metadata = {}
        self._last_ocr_segments = []
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> dict[str, object]:
        """Create counters used by optional reporting."""
        return {
            "pages": 0,
            "images": 0,
            "ocr_pages": 0,
            "ocr_cache_hits": 0,
            "ocr_failures": 0,
            "ocr_retries": 0,
            "chapters": 0,
            "headings": 0,
            "paragraphs": 0,
            "pages_with_text": 0,
            "pages_with_images": 0,
            "page_statistics": [],
            "cover_extracted": False,
            "metadata": {},
            "elapsed_seconds": 0.0,
            "validation": None,
            "repeated_headers": [],
        }

    def _worker_options(self) -> dict[str, object]:
        """Return independent settings for a page-processing worker."""
        return {
            "image_quality": self.image_quality,
            "max_image_size": self.max_image_size,
            "ocr_enabled": self.ocr_enabled,
            "ocr_language": self.ocr_language,
            "ocr_dpi": self.ocr_dpi,
            "keep_scan_images": self.keep_scan_images,
            "ocr_device": self.ocr_device,
            "ocr_confidence": self.ocr_confidence,
            "ocr_preprocess": self.ocr_preprocess,
            "ocr_cache_dir": str(self.ocr_cache_dir)
            if self.ocr_cache_dir
            else None,
            "resume": self.resume,
            "layout": self.layout,
            "repeated_header_action": self.repeated_header_action,
            "chapter_overrides": dict(self.chapter_overrides),
            "image_placement": self.image_placement,
            "language": self.language,
            "ocr_retry": self.ocr_retry,
            "ocr_retry_dpi": self.ocr_retry_dpi,
            "ocr_retry_confidence": self.ocr_retry_confidence,
            "workers": 1,
        }

    def _initialize_parallel_worker(self, pdf_path: str) -> None:
        """Create isolated converter state for one worker thread."""
        worker = PDFToEPUBConverter(**self._worker_options())
        worker._current_pdf_path = str(Path(pdf_path).resolve())
        _parallel_worker_state.converter = worker
        _parallel_worker_state.pdf_path = pdf_path

    @staticmethod
    def _process_parallel_page(page_num: int) -> dict[str, object]:
        """Process one page using only the current thread's PDF and converter."""
        worker = _parallel_worker_state.converter
        with fitz.open(_parallel_worker_state.pdf_path) as document:
            page = document[page_num - 1]
            worker._stats = worker._new_stats()
            worker._page_metadata = {}
            worker._ocr_page_diagnostics = []
            local_book = epub.EpubBook()
            chapter = worker.process_page(local_book, page, page_num)
            return {
                "page_num": page_num,
                "chapter": chapter,
                "items": list(local_book.get_items()),
                "stats": worker._stats,
                "page_metadata": worker._page_metadata,
                "ocr_diagnostics": worker._ocr_page_diagnostics,
            }

    def _effective_workers(self) -> int:
        """Apply conservative limits around OCR model memory and GPU thread safety."""
        if self.workers <= 1:
            return 1
        if self.ocr_enabled and self.ocr_device != "cpu":
            print(
                "Warning: parallel OCR is disabled for auto/CUDA devices; "
                "use --ocr-device cpu to parallelize OCR.",
                flush=True,
            )
            return 1
        limit = 2 if self.ocr_enabled else 4
        if self.workers > limit:
            print(
                f"Warning: limiting page workers to {limit} to control memory use.",
                flush=True,
            )
        return min(self.workers, limit)

    def _process_pages_parallel(
        self, pdf_path: str, total_pages: int
    ) -> list[dict[str, object]]:
        """Process pages concurrently while keeping all EPUB assembly on this thread."""
        worker_count = self._effective_workers()
        if worker_count == 1:
            worker = PDFToEPUBConverter(**self._worker_options())
            worker._current_pdf_path = str(Path(pdf_path).resolve())
            results = []
            with fitz.open(pdf_path) as document:
                for page_num in range(1, total_pages + 1):
                    worker._stats = worker._new_stats()
                    worker._page_metadata = {}
                    worker._ocr_page_diagnostics = []
                    local_book = epub.EpubBook()
                    chapter = worker.process_page(
                        local_book, document[page_num - 1], page_num
                    )
                    results.append(
                        {
                            "page_num": page_num,
                            "chapter": chapter,
                            "items": list(local_book.get_items()),
                            "stats": worker._stats,
                            "page_metadata": worker._page_metadata,
                            "ocr_diagnostics": worker._ocr_page_diagnostics,
                        }
                    )
            return results
        with ThreadPoolExecutor(
            max_workers=worker_count,
            initializer=self._initialize_parallel_worker,
            initargs=(pdf_path,),
            thread_name_prefix="pdf-page",
        ) as executor:
            return list(
                executor.map(
                    self._process_parallel_page,
                    range(1, total_pages + 1),
                )
            )

    def _merge_page_result(
        self,
        epub_book: epub.EpubBook,
        chapters: list[epub.EpubHtml],
        result: dict[str, object],
        style: epub.EpubItem,
    ) -> None:
        """Merge an isolated page result into the main EPUB in page order."""
        for item in result["items"]:
            epub_book.add_item(item)
        chapter = result["chapter"]
        if chapter:
            chapter.add_item(style)
            chapters.append(chapter)

        page_stats = result["stats"]
        for key in (
            "pages",
            "images",
            "ocr_pages",
            "ocr_cache_hits",
            "ocr_failures",
            "ocr_retries",
            "headings",
            "paragraphs",
            "pages_with_text",
            "pages_with_images",
        ):
            self._stats[key] += page_stats[key]
        self._stats["page_statistics"].extend(page_stats["page_statistics"])
        self._page_metadata.update(result["page_metadata"])
        self._ocr_page_diagnostics.extend(result["ocr_diagnostics"])

    def convert(self, pdf_path: str, output_path: str) -> bool:
        """
        Convert a PDF file to EPUB format.

        Args:
            pdf_path: Path to the input PDF file
            output_path: Path for the output EPUB file

        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            started = time.perf_counter()
            self._current_pdf_path = str(Path(pdf_path).resolve())
            self._stats = self._new_stats()
            self._ocr_page_diagnostics = []
            self._page_metadata = {}
            epub_book = epub.EpubBook()

            style = self.add_styles(epub_book)

            # Create chapters for each page
            chapters = []

            with fitz.open(pdf_path) as doc:
                metadata = self._extract_pdf_metadata(doc, pdf_path)
                self._stats["metadata"] = metadata
                self.language = metadata["language"]
                epub_book.set_title(metadata["title"])
                epub_book.set_language(metadata["language"])
                if metadata.get("author"):
                    epub_book.add_author(metadata["author"])
                if metadata.get("subject"):
                    epub_book.add_metadata("DC", "subject", metadata["subject"])
                if metadata.get("keywords"):
                    epub_book.add_metadata("DC", "description", metadata["keywords"])
                self._extract_cover(epub_book, doc)
                total_pages = doc.page_count
                if self.workers > 1:
                    page_results = self._process_pages_parallel(
                        pdf_path, total_pages
                    )
                    for result in page_results:
                        page_num = result["page_num"]
                        if (
                            page_num == 1
                            or page_num % 10 == 0
                            or page_num == total_pages
                        ):
                            percent = page_num / total_pages * 100
                            print(
                                f"Processing page {page_num}/{total_pages} "
                                f"({percent:.1f}%)",
                                flush=True,
                            )
                        self._merge_page_result(
                            epub_book, chapters, result, style
                        )
                else:
                    for i, page in enumerate(doc):
                        page_num = i + 1
                        if page_num == 1 or page_num % 10 == 0 or page_num == total_pages:
                            percent = page_num / total_pages * 100
                            print(
                                f"Processing page {page_num}/{total_pages} "
                                f"({percent:.1f}%)",
                                flush=True,
                            )
                        chapter = self.process_page(epub_book, page, page_num)
                        if chapter:
                            chapter.add_item(style)
                            chapters.append(chapter)

            self._apply_repeated_header_policy(chapters)
            self._apply_book_semantics(chapters)
            # Add chapters to the book
            for chapter in chapters:
                epub_book.add_item(chapter)

            # Detect chapter headings after all pages have been processed.
            self.add_toc(epub_book, chapters)
            self._stats["chapters"] = len(self._identify_chapters(chapters))

            # Save the EPUB file
            epub.write_epub(
                output_path,
                epub_book,
                {"image_quality": self.image_quality, "epub3_landmark": True},
            )
            if self.validate_output:
                validation = self.validate_epub_diagnostics(output_path)
                self._stats["validation"] = validation
                if not validation["valid"]:
                    print(
                        f"Warning: EPUB validation failed: {output_path} "
                        f"({len(validation['errors'])} error(s), "
                        f"{len(validation['warnings'])} warning(s))"
                    )
            self._stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self._write_report(pdf_path, output_path)

            print(f"Successfully converted: {pdf_path} -> {output_path}")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            print(f"Error converting {pdf_path}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def preflight_environment(ocr_enabled: bool = True, ocr_device: str = "auto") -> dict:
        """Check required packages and optional OCR/GPU availability."""
        result = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "required": {
                "pymupdf": fitz is not None,
                "ebooklib": epub is not None,
                "pillow": Image is not None,
            },
            "optional": {},
            "ocr_device": "disabled" if not ocr_enabled else "unavailable",
            "cuda_available": False,
        }
        for module in ("easyocr", "torch", "numpy"):
            result["optional"][module] = importlib.util.find_spec(module) is not None
        if ocr_enabled and result["optional"]["torch"]:
            try:
                import torch

                result["cuda_available"] = bool(torch.cuda.is_available())
                if ocr_device == "cuda" and not result["cuda_available"]:
                    result["ocr_device"] = "cpu (CUDA unavailable)"
                elif ocr_device == "cpu":
                    result["ocr_device"] = "cpu"
                else:
                    result["ocr_device"] = (
                        "cuda" if result["cuda_available"] else "cpu"
                    )
            except (ImportError, OSError, RuntimeError):
                result["ocr_device"] = "cpu (torch unavailable)"
        elif ocr_enabled:
            result["ocr_device"] = "cpu (torch unavailable)"
        result["ready"] = all(result["required"].values()) and (
            not ocr_enabled
            or (result["optional"]["easyocr"] and result["optional"]["numpy"])
        )
        return result

    def _extract_pdf_metadata(self, doc, pdf_path: str) -> dict[str, str]:
        """Normalize PDF metadata for EPUB Dublin Core fields."""
        raw = doc.metadata or {}
        title = (raw.get("title") or "").strip() or self._extract_title(pdf_path)
        language = (raw.get("language") or "").strip() or self.language
        return {
            "title": title,
            "author": (raw.get("author") or "").strip(),
            "subject": (raw.get("subject") or "").strip(),
            "keywords": (raw.get("keywords") or "").strip(),
            "language": language,
            "creator": (raw.get("creator") or "").strip(),
            "producer": (raw.get("producer") or "").strip(),
        }

    def _extract_cover(self, epub_book: epub.EpubBook, doc) -> None:
        """Use the first substantial PDF image, or a rendered first page, as cover."""
        if not doc.page_count:
            return
        page = doc[0]
        image_bytes = None
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            page_area = page.rect.get_area()
            coverage = max(
                (
                    rect.get_area() / page_area
                    for rect in page.get_image_rects(xref)
                ),
                default=0,
            )
            if coverage >= 0.5:
                try:
                    image_bytes = page.parent.extract_image(xref)["image"]
                    break
                except (KeyError, OSError, RuntimeError):
                    continue
        if image_bytes is None:
            try:
                pixmap = page.get_pixmap(dpi=150, alpha=False)
                image_bytes = pixmap.tobytes("jpeg")
            except (OSError, RuntimeError, ValueError):
                return
        try:
            cover = Image.open(BytesIO(image_bytes))
            epub_book.set_cover("cover.jpg", self._resize_image(cover))
            self._stats["cover_extracted"] = True
        except (OSError, RuntimeError, ValueError):
            return

    def _extract_title(self, pdf_path: str) -> str:
        """Extract title from PDF filename or first page."""
        # Try to extract from filename
        base_name = Path(pdf_path).stem
        if "_" in base_name:
            title = base_name.split("_", 1)[0]
        elif "-" in base_name:
            title = base_name.split("-", 1)[0]
        else:
            title = base_name

        # Clean up title - remove special characters but keep spaces and hyphens
        title = re.sub(r"[^a-zA-Z0-9\s\-]", "", title).strip()

        if not title or len(title) < 2:
            # Fallback to generic title
            title = Path(pdf_path).name

        return title

    def process_page(self, epub_book, page, page_num: int) -> epub.EpubHtml | None:
        """Process a single PDF page and create an EPUB chapter."""
        self._stats["pages"] += 1

        # Create chapter for this page
        chapter = epub.EpubHtml(
            uid=f"page_{page_num}",
            file_name=f"page_{page_num}.xhtml",
            title=f"Page {page_num}",
            lang=self.language,
        )

        # Extract text content from the page with better formatting preservation
        text_parts, images_to_embed, header_text = self._extract_text_and_images(
            epub_book, page, page_num
        )
        page_metadata = getattr(self, "_page_metadata", {}).get(page_num, {})

        # Combine text parts and create content
        full_text = "\n".join(text_parts).strip()
        
        chapter.source_text = "\n".join(
            part for part in (header_text, full_text) if part
        )

        # Create EPUB content with text and images
        content = self.create_epub_content(
            full_text,
            images_to_embed,
            header_text,
            text_segments=page_metadata.get("text_segments"),
            semantic_segments=page_metadata.get("semantic_segments"),
            page_num=page_num,
        )

        if content:
            chapter.set_content(content)
        chapter.page_number = page_num
        chapter._page_metadata = page_metadata

        return chapter

    def format_chapter_text(self, text: str) -> str:
        """Normalize extracted lines using the configured layout mode."""
        lines = [
            self._repair_ocr_line(re.sub(r"\s+", " ", line).strip())
            for line in text.splitlines()
        ]
        lines = self._merge_hyphenated_lines(lines)
        lines = [line for line in lines if line]
        if self.layout == "preserve":
            return "\n\n".join(lines)
        return " ".join(lines)

    def _repair_ocr_line(self, line: str) -> str:
        """Repair common OCR spacing and line-break artifacts."""
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([(\[])\s+", r"\1", line)
        line = re.sub(r"\s+([)\]])", r"\1", line)
        return line.strip()

    def _merge_hyphenated_lines(self, lines: list[str]) -> list[str]:
        """Join words split by a scan line break while preserving real hyphens."""
        merged = []
        for line in lines:
            line = self._repair_ocr_line(line)
            if (
                merged
                and merged[-1].endswith("-")
                and line
                and line[0].islower()
            ):
                merged[-1] = merged[-1][:-1] + line
            else:
                merged.append(line)
        return merged

    def _merge_semantic_paragraphs(
        self, segments: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Combine OCR/native continuation lines into flowing paragraphs."""
        if not segments:
            return []
        result = []
        current = None
        for segment in segments:
            text = self._repair_ocr_line(str(segment.get("text", "")))
            if not text:
                continue
            item = dict(segment)
            item["text"] = text
            if item.get("kind") == "heading":
                if current is not None:
                    result.append(current)
                    current = None
                result.append(item)
                continue
            if current is None:
                current = item
                continue
            previous = str(current["text"])
            separated = (
                previous.endswith((".", "!", "?", ":", ";", '"', "'"))
                and text[:1].isupper()
                and len(previous) < 180
            )
            if separated:
                result.append(current)
                current = item
            else:
                current["text"] = f"{previous} {text}"
        if current is not None:
            result.append(current)
        for item in result:
            item["text"] = " ".join(
                self._merge_hyphenated_lines(str(item["text"]).splitlines())
            )
        return result

    def _group_positioned_lines(
        self, lines: list[tuple[float, float, float, str]]
    ) -> list[str]:
        """Group nearby PDF lines into paragraphs without losing reading order."""
        return [
            text for _top, text in self._group_positioned_line_data(lines)
        ]

    def _group_positioned_line_data(
        self, lines: list[tuple[float, float, float, str]]
    ) -> list[tuple[float, str]]:
        """Group nearby PDF lines while retaining each group's vertical position."""
        if not lines:
            return []
        grouped = []
        current = [lines[0]]
        heights = [max(1.0, lines[0][2] - lines[0][0])]
        for line in lines[1:]:
            previous = current[-1]
            gap = line[0] - previous[2]
            height = max(1.0, line[2] - line[0])
            same_column = abs(line[1] - previous[1]) <= max(12.0, height * 2)
            if gap <= max(8.0, sum(heights) / len(heights) * 0.9) and same_column:
                current.append(line)
                heights.append(height)
            else:
                grouped.append(
                    (min(item[0] for item in current), " ".join(item[3] for item in current))
                )
                current = [line]
                heights = [height]
        grouped.append(
            (min(item[0] for item in current), " ".join(item[3] for item in current))
        )
        return grouped

    def _extract_text_and_images(
        self, epub_book: epub.EpubBook, page, page_num: int
    ) -> tuple[list[str], list[tuple[int, str, float, int]], str]:
        """Extract text and images from a PDF page."""
        text_parts = []
        images_to_embed = []
        header_text = ""
        ocr_succeeded = False
        header_candidates = []

        # Get text blocks with positions and retain line geometry for grouping.
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        native_lines = []
        native_style_lines = []
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = [
                    re.sub(r"\s+", " ", span["text"]).strip()
                    for span in line["spans"]
                    if span.get("text", "").strip()
                ]
                text = " ".join(part for part in spans if part).strip()
                if text:
                    bbox = line.get("bbox", (0, 0, 0, 0))
                    native_lines.append((bbox[1], bbox[0], bbox[3], text))
                    spans = [span for span in line["spans"] if span.get("text", "").strip()]
                    sizes = [float(span.get("size", 0)) for span in spans]
                    flags = [int(span.get("flags", 0)) for span in spans]
                    native_style_lines.append(
                        (
                            bbox[1],
                            bbox[0],
                            bbox[3],
                            text,
                            max(sizes, default=0.0),
                            any(flag & 16 for flag in flags),
                        )
                    )
        native_lines = self._order_native_lines(native_lines, page.rect.width)
        native_style_lines = self._order_native_lines(
            native_style_lines, page.rect.width
        )
        text_segments = self._group_positioned_line_data(native_lines)
        semantic_segments = self._detect_semantic_segments(
            text_segments,
            native_style_lines,
            page.rect.height,
        )
        semantic_segments = self._merge_semantic_paragraphs(semantic_segments)
        text_segments = [
            (segment["top"], str(segment["text"])) for segment in semantic_segments
        ]
        text_parts.extend(text for _top, text in text_segments)
        header_candidates.extend(
            text
            for _top, text in self._header_candidates_from_lines(
                native_lines, page.rect.height
            )
        )

        if self.ocr_enabled and not self._has_meaningful_text(text_parts):
            try:
                ocr_text, header_text = self._ocr_page(page, page_num)
                if ocr_text:
                    text_parts.append(ocr_text)
                    text_segments.extend(
                        self._last_ocr_segments
                        or [(page.rect.height * 0.2, ocr_text)]
                    )
                    semantic_segments = self._detect_semantic_segments(
                        text_segments, [], page.rect.height, ocr=True
                    )
                    semantic_segments = self._merge_semantic_paragraphs(
                        semantic_segments
                    )
                    text_segments = [
                        (segment["top"], str(segment["text"]))
                        for segment in semantic_segments
                    ]
                ocr_succeeded = len(ocr_text.split()) >= 20
                self._stats["ocr_pages"] += 1
                self._stats["pages_with_text"] += bool(ocr_text)
            except RuntimeError as exc:
                self._stats["ocr_failures"] += 1
                self._record_ocr_failure(page_num, str(exc))
                print(f"Warning: OCR unavailable on page {page_num}: {exc}")
            header_candidates.extend(
                line for line in header_text.splitlines() if line.strip()
            )

        # Extract images from the page - handle all color modes
        images = page.get_images(full=True)

        for img_index, img_info in enumerate(images):
            try:
                xref = img_info[0]
                if (
                    ocr_succeeded
                    and not self.keep_scan_images
                    and self._is_full_page_image(page, xref)
                ):
                    continue

                image_data = page.parent.extract_image(xref)
                pil_img = Image.open(BytesIO(image_data["image"]))
                resized_data = self._resize_image(pil_img)

                img_name = f"image_page_{page_num}_{img_index}.jpg"
                epub_item = epub.EpubItem(
                    uid=f"image_{page_num}_{img_index}",
                    file_name=img_name,
                    media_type="image/jpeg",
                    content=resized_data,
                )
                epub_book.add_item(epub_item)
                self._stats["images"] += 1

                rects = page.get_image_rects(xref)
                top = min((rect.y0 for rect in rects), default=page.rect.height)
                alt_text = self._image_alt_text(
                    page_num, img_index, text_segments, top
                )
                html_img = (
                    f'<img src="{img_name}" alt="{html.escape(alt_text)}" '
                    'role="img" '
                    'style="max-width:100%; height:auto; display:block; '
                    'margin:1em auto;">'
                )
                caption = self._image_caption(text_segments, top)
                if caption:
                    caption_id = f"caption-page-{page_num}-{img_index}"
                    html_img = (
                        f'<figure class="illustration" aria-labelledby="{caption_id}">'
                        f"{html_img}<figcaption id=\"{caption_id}\">"
                        f"{html.escape(caption)}</figcaption></figure>"
                    )
                else:
                    html_img = f'<figure class="illustration">{html_img}</figure>'
                insertion_index = sum(
                    segment_top <= top for segment_top, _text in text_segments
                )
                images_to_embed.append((img_index, html_img, top, insertion_index))

            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as e:
                print(f"Warning: Could not extract image {img_index}: {e}")

        if self.image_placement == "position":
            images_to_embed.sort(key=lambda item: (item[2], item[0]))
        self._stats["pages_with_text"] += bool(text_parts) and not ocr_succeeded
        self._stats["pages_with_images"] += bool(images_to_embed)
        self._stats["headings"] += sum(
            segment["kind"] == "heading" for segment in semantic_segments
        )
        self._stats["paragraphs"] += sum(
            segment["kind"] == "paragraph" for segment in semantic_segments
        )
        self._stats["page_statistics"].append(
            {
                "page": page_num,
                "text_characters": len("\n".join(text_parts).strip()),
                "text_segments": len(text_segments),
                "headings": sum(
                    segment["kind"] == "heading" for segment in semantic_segments
                ),
                "paragraphs": sum(
                    segment["kind"] == "paragraph" for segment in semantic_segments
                ),
                "images": len(images_to_embed),
                "ocr_used": ocr_succeeded,
            }
        )
        self._page_metadata[page_num] = {
            "text_segments": text_segments,
            "semantic_segments": semantic_segments,
            "header_candidates": header_candidates,
            "images": images_to_embed,
            "body_text": "\n".join(text_parts).strip(),
            "header_text": header_text,
        }
        return text_parts, images_to_embed, header_text

    def _detect_semantic_segments(
        self,
        text_segments: list[tuple[float, str]],
        style_lines: list[tuple],
        page_height: float,
        ocr: bool = False,
    ) -> list[dict[str, object]]:
        """Classify positioned text into accessible headings and paragraphs."""
        if not text_segments:
            return []
        sizes = [line[4] for line in style_lines if len(line) > 4 and line[4] > 0]
        body_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
        style_by_text = {line[3]: line for line in style_lines if len(line) > 3}
        result = []
        for top, text in text_segments:
            style = style_by_text.get(text)
            size = style[4] if style and len(style) > 4 else body_size
            bold = bool(style and len(style) > 5 and style[5])
            short = len(text) <= 100 and len(text.split()) <= 14
            title_like = self._looks_like_chapter_heading(text)
            visually_prominent = bool(body_size and size >= body_size * 1.18)
            is_heading = short and (title_like or visually_prominent or bold and top < page_height * 0.35)
            level = 1 if title_like or visually_prominent and top < page_height * 0.25 else 2
            result.append(
                {
                    "top": top,
                    "text": text,
                    "kind": "heading" if is_heading else "paragraph",
                    "level": level,
                    "ocr": ocr,
                }
            )
        return result

    def _header_candidates_from_lines(
        self,
        lines: list[tuple[float, float, float, str]],
        page_height: float,
    ) -> list[tuple[float, str]]:
        """Return short top-margin lines that are safe running-header candidates."""
        candidates = []
        top_limit = page_height * 0.18
        for top, _left, _bottom, text in lines:
            if top <= top_limit and len(text) <= 80 and len(text.split()) <= 10:
                candidates.append((top, text))
        return candidates

    def _order_native_lines(
        self, lines: list[tuple[float, float, float, str]], page_width: float
    ) -> list[tuple[float, float, float, str]]:
        """Order native text by columns when a page has a multi-column layout."""
        if len(lines) < 6 or page_width <= 0:
            return sorted(lines, key=lambda item: (item[0], item[1]))

        x_positions = sorted({line[1] for line in lines})
        gaps = [
            (right - left, left, right)
            for left, right in zip(x_positions, x_positions[1:])
        ]
        if not gaps:
            return sorted(lines, key=lambda item: (item[0], item[1]))

        gap, left_edge, right_edge = max(gaps)
        if gap < page_width * 0.18:
            return sorted(lines, key=lambda item: (item[0], item[1]))

        left_column = [line for line in lines if line[1] <= left_edge]
        right_column = [line for line in lines if line[1] >= right_edge]
        if len(left_column) < 3 or len(right_column) < 3:
            return sorted(lines, key=lambda item: (item[0], item[1]))

        return sorted(left_column, key=lambda item: (item[0], item[1])) + sorted(
            right_column, key=lambda item: (item[0], item[1])
        )

    def _has_meaningful_text(self, text_parts: list[str]) -> bool:
        """Return whether native extraction contains enough body text."""
        text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
        if not text:
            return False
        if self._looks_like_page_number(text):
            return False
        return len(re.findall(r"[A-Za-z]{2,}", text)) >= 20

    def _is_full_page_image(self, page, xref: int) -> bool:
        """Return whether an image covers nearly the entire PDF page."""
        page_area = page.rect.get_area()
        if page_area <= 0:
            return False

        return any(
            rect.get_area() / page_area >= 0.95 for rect in page.get_image_rects(xref)
        )

    def _ocr_page(self, page, page_num: int | None = None) -> tuple[str, str]:
        """Run EasyOCR and separate top headers from body text."""
        self._last_ocr_segments = []
        cache_file = self._ocr_cache_file(page_num)
        cached_payload = self._load_ocr_cache(cache_file)
        if cached_payload is not None:
            self._stats["ocr_cache_hits"] += 1
            results, cached_size = cached_payload
        else:
            cached_size = None
        if cached_payload is None:
            try:
                import easyocr
                import numpy as np
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "OCR requires EasyOCR. Install it with: pip install easyocr"
                ) from exc

        try:
            if cached_payload is None and self._ocr_reader is None:
                languages = self._ocr_languages()
                if not languages:
                    raise ValueError("At least one OCR language is required.")
                use_gpu = self._resolve_ocr_gpu(torch)
                self._ocr_gpu = use_gpu
                print(
                    f"OCR device: {'CUDA GPU' if use_gpu else 'CPU'}",
                    flush=True,
                )
                self._ocr_reader = easyocr.Reader(
                    languages,
                    gpu=use_gpu,
                    verbose=False,
                )

            results = cached_payload[0] if cached_payload is not None else None
            image_size = cached_size
            cache_hit = cached_payload is not None
            if results is None:
                results, image_size = self._ocr_attempt(
                    page, np, self.ocr_dpi, self.ocr_preprocess, False
                )
                if cache_file:
                    self._save_ocr_cache(cache_file, results, image_size)

            confidence_values = []
            for result in results:
                if len(result) < 3:
                    continue
                try:
                    confidence_values.append(float(result[2]))
                except (TypeError, ValueError):
                    continue
            average_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else 0.0
            )
            should_retry = (
                self.ocr_retry
                and bool(confidence_values)
                and average_confidence < self.ocr_retry_confidence
            )
            if should_retry:
                if cached_payload is not None:
                    try:
                        import easyocr
                        import numpy as np
                        import torch
                    except ImportError as exc:
                        raise RuntimeError(
                            "OCR retry requires EasyOCR. Install it with: "
                            "pip install easyocr"
                        ) from exc
                    if self._ocr_reader is None:
                        self._ocr_gpu = self._resolve_ocr_gpu(torch)
                        self._ocr_reader = easyocr.Reader(
                            self._ocr_languages(),
                            gpu=self._ocr_gpu,
                            verbose=False,
                        )
                self._stats["ocr_retries"] += 1
                print(
                    f"Retrying low-confidence OCR on page {page_num or '?'} "
                    f"at {self.ocr_retry_dpi} DPI.",
                    flush=True,
                )
                results, image_size = self._ocr_attempt(
                    page,
                    np,
                    self.ocr_retry_dpi,
                    True,
                    True,
                )
                cache_hit = False
                if cache_file:
                    self._save_ocr_cache(cache_file, results, image_size)

            height = image_size[1] if image_size else page.rect.height
            body = []
            headers = []
            confidence_values = []
            for result in results:
                if len(result) < 3:
                    continue
                bbox, text, confidence = result
                try:
                    confidence = float(confidence)
                    confidence_values.append(confidence)
                    if confidence < self.ocr_confidence:
                        continue
                except (TypeError, ValueError):
                    continue
                text = re.sub(r"\s+", " ", str(text)).strip()
                if not text:
                    continue
                top = min(point[1] for point in bbox)
                bottom = max(point[1] for point in bbox)
                center_y = (top + bottom) / 2
                if center_y >= height * 0.88 and (
                    self._looks_like_page_number(text)
                    or self._looks_like_page_number_fragment(text)
                ):
                    continue
                if center_y <= height * 0.18:
                    headers.append((top, bbox[0][0], text))
                else:
                    body.append((top, bbox[0][0], text))

            headers.sort(key=lambda item: (item[0], item[1]))
            body.sort(key=lambda item: (item[0], item[1]))
            header_candidates = [
                item
                for item in headers
                if len(item[2]) <= 80 and len(item[2].split()) <= 10
            ][:1]
            grouped_body = self._group_ocr_lines(body, height)
            self._last_ocr_segments = [
                (top * page.rect.height / max(1, height), text)
                for top, text in self._group_ocr_line_data(body, height)
            ]
            self._record_ocr_success(
                page_num,
                confidence_values,
                cache_hit,
                retried=should_retry,
            )
            return (
                "\n".join(grouped_body),
                "\n".join(item[2] for item in header_candidates),
            )
        except Exception as exc:
            raise RuntimeError(
                "OCR failed. Ensure EasyOCR can load its models and that the "
                f"language codes are supported: {self.ocr_language}."
            ) from exc

    def _ocr_attempt(
        self, page, np, dpi: int, preprocess: bool, force_render: bool
    ) -> tuple[list, tuple[int, int]]:
        """Run one OCR pass, optionally forcing a higher-DPI page render."""
        image = None
        if not force_render:
            ocr_image_xref = self._select_ocr_image(page)
            if ocr_image_xref is not None:
                image_data = page.parent.extract_image(ocr_image_xref)
                image = Image.open(BytesIO(image_data["image"]))
        if image is None:
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
        image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
        if preprocess:
            image = self._preprocess_ocr_image(image)
        try:
            import easyocr
            results = self._read_ocr(np, image, detail=1, paragraph=False)
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
            if not self._ocr_gpu:
                raise
            print("Warning: GPU OCR failed; retrying this page on CPU.")
            self._ocr_reader = easyocr.Reader(
                self._ocr_languages(),
                gpu=False,
                verbose=False,
            )
            self._ocr_gpu = False
            results = self._read_ocr(np, image, detail=1, paragraph=False)
        return results, image.size

    def _record_ocr_success(
        self,
        page_num: int | None,
        confidences: list[float],
        cache_hit: bool,
        retried: bool = False,
    ) -> None:
        """Record confidence details for one successful OCR page."""
        if page_num is None:
            return
        self._ocr_page_diagnostics.append(
            {
                "page": page_num,
                "status": "cache" if cache_hit else "success",
                "retried": retried,
                "average_confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
                "min_confidence": min(confidences) if confidences else None,
                "detections": len(confidences),
                "confidence_values": confidences,
            }
        )

    def _record_ocr_failure(self, page_num: int, error: str) -> None:
        """Record an OCR failure for the JSON diagnostics report."""
        self._ocr_page_diagnostics.append(
            {"page": page_num, "status": "failure", "error": error}
        )

    def _ocr_diagnostics(self) -> dict[str, object]:
        """Build aggregate and per-page OCR confidence diagnostics."""
        successful = [
            item
            for item in self._ocr_page_diagnostics
            if item.get("status") in {"success", "cache"}
        ]
        confidence_values = [
            value
            for item in successful
            for value in item.get("confidence_values", [])
        ]
        low_pages = [
            item["page"]
            for item in successful
            if item.get("min_confidence") is not None
            and item["min_confidence"] < self.ocr_confidence
        ]
        failures = [
            item for item in self._ocr_page_diagnostics if item.get("status") == "failure"
        ]
        return {
            "page_count": self._stats["pages"],
            "ocr_page_count": len(self._ocr_page_diagnostics),
            "average_confidence": (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            ),
            "min_confidence": min(confidence_values) if confidence_values else None,
            "low_confidence_pages": low_pages,
            "failures": failures,
            "retry_count": self._stats["ocr_retries"],
            "pages": self._ocr_page_diagnostics,
        }

    def _resolve_ocr_gpu(self, torch) -> bool:
        """Resolve the configured OCR device, safely falling back when needed."""
        if self.ocr_device == "cpu":
            return False
        available = bool(torch.cuda.is_available())
        if self.ocr_device == "cuda" and not available:
            print("Warning: CUDA requested but unavailable; using CPU.")
            return False
        return available

    def _preprocess_ocr_image(self, image: Image.Image) -> Image.Image:
        """Apply conservative preprocessing that helps low-contrast scans."""
        from PIL import ImageEnhance, ImageOps

        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray)
        return ImageEnhance.Contrast(gray).enhance(1.15)

    def _group_ocr_lines(
        self, items: list[tuple[float, float, str]], image_height: float
    ) -> list[str]:
        """Group OCR words into reading-order lines and paragraphs."""
        return [
            text for _top, text in self._group_ocr_line_data(items, image_height)
        ]

    def _group_ocr_line_data(
        self, items: list[tuple[float, float, str]], image_height: float
    ) -> list[tuple[float, str]]:
        """Group OCR words while retaining each line's visual position."""
        if not items:
            return []
        line_height = max(8.0, image_height * 0.025)
        lines: list[list[tuple[float, float, str]]] = []
        for item in items:
            if not lines or abs(item[0] - lines[-1][0][0]) > line_height:
                lines.append([item])
            else:
                lines[-1].append(item)
        return [
            (
                min(part[0] for part in line),
                " ".join(part[2] for part in sorted(line, key=lambda value: value[1])),
            )
            for line in lines
        ]

    def _ocr_cache_file(self, page_num: int | None) -> Path | None:
        """Return a stable per-page cache path when caching is configured."""
        if not self.ocr_cache_dir or page_num is None or not self._current_pdf_path:
            return None
        source = Path(self._current_pdf_path)
        try:
            stat = source.stat()
            identity = (
                f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{page_num}|"
                f"{self.ocr_language}|{self.ocr_dpi}|{self.ocr_confidence}|"
                f"{self.ocr_preprocess}|{self.max_image_size}|{self.ocr_device}"
            )
        except OSError:
            identity = f"{source}|{page_num}|{self.ocr_language}"
        import hashlib

        return self.ocr_cache_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"

    def _load_ocr_cache(self, cache_file: Path | None):
        """Load cached EasyOCR results, if resume mode and valid JSON are enabled."""
        if not self.resume or not cache_file or not cache_file.is_file():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if isinstance(cached, list):
                return cached, None
            if isinstance(cached, dict) and isinstance(cached.get("results"), list):
                size = cached.get("image_size")
                return cached["results"], size if isinstance(size, list) else None
            return None
        except (OSError, ValueError, TypeError):
            return None

    def _save_ocr_cache(self, cache_file: Path, results, image_size) -> None:
        """Persist OCR results without making caching a conversion requirement."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = [
                [bbox, str(text), float(confidence)]
                for bbox, text, confidence in results
            ]
            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump({"image_size": list(image_size), "results": serializable}, handle)
        except (OSError, TypeError, ValueError):
            print(f"Warning: Could not save OCR cache: {cache_file}")

    def _ocr_languages(self) -> list[str]:
        """Return normalized EasyOCR language codes."""
        return [
            {"eng": "en", "hin": "hi"}.get(code.strip(), code.strip())
            for code in re.split(r"[+,]", self.ocr_language)
            if code.strip()
        ]

    def _read_ocr(self, np, image, **options):
        """Run OCR with the configured reader."""
        return self._ocr_reader.readtext(
            np.asarray(image),
            canvas_size=max(self.max_image_size),
            **options,
        )

    def _select_ocr_image(self, page) -> int | None:
        """Select a page-covering image, or render the complete page."""
        candidates = []
        page_area = page.rect.get_area()
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            coverage = max(
                (rect.get_area() / page_area for rect in page.get_image_rects(xref)),
                default=0,
            )
            candidates.append((coverage, xref))

        if not candidates:
            return None
        coverage, xref = max(candidates)
        return xref if coverage >= 0.5 else None

    def _looks_like_page_number(self, text: str) -> bool:
        """Identify numeric footer text without removing body numbers."""
        normalized = re.sub(r"\s+", " ", text.strip()).casefold()
        if re.fullmatch(
            r"(?:page\s*)?[\d?]+"
            r"(?:\s*(?:of|0f|/)\s*[\d?]+)?"
            r"(?:\s+\d+%)?",
            normalized,
        ):
            return True
        return bool(re.fullmatch(r"[ivxlcdm]+", normalized))

    def _looks_like_page_number_fragment(self, text: str) -> bool:
        """Identify OCR fragments from a split page-number footer."""
        normalized = re.sub(r"\s+", " ", text.strip()).casefold()
        return bool(
            re.fullmatch(r"page", normalized)
            or re.fullmatch(r"(?:of|0f)\s*[\d?]+", normalized)
            or re.fullmatch(r"[\d?]+%?", normalized)
        )

    def create_epub_content(
        self,
        text: str,
        images: list[tuple],
        header: str = "",
        text_segments: list[tuple[float, str]] | None = None,
        semantic_segments: list[dict[str, object]] | None = None,
        page_num: int | None = None,
        section_kind: str = "body",
        section_title: str = "",
    ) -> str | None:
        """Create EPUB content with text and embedded images."""
        if not text and not images and not header:
            return None

        html_parts = []

        if header:
            escaped_header = html.escape(header).replace("\n", "<br/>")
            html_parts.append(
                f'<header class="page-header" role="banner"><p>{escaped_header}</p></header>'
            )

        if text:
            text = self.format_chapter_text(text)
            if semantic_segments:
                positioned_images = {}
                if self.image_placement == "position":
                    for image in images:
                        _index, img_html, _top, insertion_index = self._image_parts(
                            image
                        )
                        positioned_images.setdefault(insertion_index, []).append(
                            img_html
                        )
                section_html = []
                current_section = []
                section_number = 0
                
                # Format semantic segments as well
                for segment in semantic_segments:
                    if "text" in segment:
                        segment["text"] = self.format_chapter_text(str(segment["text"]))
                        
                for index, segment_info in enumerate(semantic_segments):
                    if index in positioned_images:
                        current_section.extend(positioned_images[index])
                    segment = html.escape(str(segment_info["text"]))
                    if segment_info["kind"] == "heading":
                        if current_section:
                            section_html.append(
                                self._render_content_section(
                                    current_section, section_number, page_num
                                )
                            )
                            section_number += 1
                            current_section = []
                        level = int(segment_info.get("level", 2))
                        heading_id = self._content_id(
                            "heading", page_num, index, segment_info["text"]
                        )
                        current_section.append(
                            f'<h{level} id="{heading_id}">{segment}</h{level}>'
                        )
                    else:
                        current_section.append(f"<p>{segment}</p>")
                if len(semantic_segments) in positioned_images:
                    current_section.extend(positioned_images[len(semantic_segments)])
                if current_section:
                    section_html.append(
                        self._render_content_section(
                            current_section, section_number, page_num
                        )
                    )
                html_parts.append(
                    self._wrap_page_content(
                        "\n".join(section_html),
                        page_num,
                        section_kind,
                        section_title,
                    )
                )
            elif text_segments and self.image_placement == "position":
                positioned_images = {}
                for image in images:
                    _index, img_html, _top, insertion_index = self._image_parts(image)
                    positioned_images.setdefault(insertion_index, []).append(img_html)
                segment_html = []
                separator = "<br/><br/>" if self.layout == "preserve" else " "
                
                # Format text segments as well
                formatted_segments = [(top, self.format_chapter_text(segment)) for top, segment in text_segments]
                
                for index, (_top, segment) in enumerate(formatted_segments):
                    if index in positioned_images:
                        segment_html.extend(positioned_images[index])
                    segment_html.append(f"<p>{html.escape(segment)}</p>")
                if len(text_segments) in positioned_images:
                    segment_html.extend(positioned_images[len(text_segments)])
                html_parts.append(
                    self._wrap_page_content(
                        separator.join(segment_html),
                        page_num,
                        section_kind,
                        section_title,
                    )
                )
            else:
                escaped_text = html.escape(text).replace("\n", "<br/>")
                html_parts.append(
                    self._wrap_page_content(
                        f"<p>{escaped_text}</p>",
                        page_num,
                        section_kind,
                        section_title,
                    )
                )

        # Add images in order, unless positioned images were already inserted.
        if not (
            text
            and self.image_placement == "position"
            and (text_segments or semantic_segments)
        ):
            image_html = "\n".join(self._image_parts(image)[1] for image in images)
            html_parts.append(
                self._wrap_page_content(
                    image_html, page_num, section_kind, section_title
                )
            )

        return "\n".join(html_parts) if html_parts else None

    @staticmethod
    def _content_id(prefix: str, page_num: int | None, index: int, text: object) -> str:
        """Create a stable, XHTML-safe identifier for generated content."""
        slug = re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-")[:32]
        return f"{prefix}-{page_num or 0}-{index}-{slug or 'item'}"

    def _render_content_section(
        self, content: list[str], section_number: int, page_num: int | None
    ) -> str:
        """Group a heading and its following content into a reflowable section."""
        heading = next(
            (part for part in content if re.match(r"<h[1-6]\b", part)), ""
        )
        heading_match = re.search(r'id="([^"]+)"', heading)
        labelled_by = (
            f' aria-labelledby="{heading_match.group(1)}"'
            if heading_match
            else ""
        )
        section_id = self._content_id("section", page_num, section_number, heading)
        return (
            f'<section id="{section_id}" class="content-section"{labelled_by}>'
            + "\n".join(content)
            + "</section>"
        )

    def _wrap_page_content(
        self,
        content: str,
        page_num: int | None,
        section_kind: str,
        section_title: str,
    ) -> str:
        """Wrap page content in accessible, reflowable EPUB section markup."""
        if not content:
            return ""
        kind = section_kind if section_kind in {"frontmatter", "chapter", "body"} else "body"
        title = html.escape(section_title)
        labelled_by = ""
        if title:
            heading_match = re.search(r'<h[1-6] id="([^"]+)"', content)
            if heading_match:
                labelled_by = f' aria-labelledby="{heading_match.group(1)}"'
            else:
                title_id = self._content_id("title", page_num, 0, section_title)
                content = (
                    f'<h1 id="{title_id}" class="visually-hidden">{title}</h1>\n'
                    f"{content}"
                )
                labelled_by = f' aria-labelledby="{title_id}"'
        page_break = ""
        if page_num and page_num > 1:
            page_break = (
                f'<span id="page-{page_num}" class="page-break" role="doc-pagebreak" '
                f'epub:type="pagebreak" aria-label="Page {page_num}"></span>\n'
            )
        epub_type = {
            "frontmatter": "frontmatter",
            "chapter": "chapter",
            "body": "bodymatter",
        }[kind]
        return (
            f'{page_break}<main id="main-{page_num or 0}" role="main">'
            f'<section class="page-section {kind}" epub:type="{epub_type}"'
            f'{labelled_by}>\n{content}\n</section></main>'
        )

    @staticmethod
    def _image_alt_text(
        page_num: int, image_index: int, text_segments: list[tuple[float, str]]
        | None, top: float
    ) -> str:
        """Create useful, non-empty alt text without pretending to know the image."""
        nearby = ""
        if text_segments:
            candidates = [
                text for segment_top, text in text_segments if abs(segment_top - top) < 120
            ]
            if candidates:
                nearby = re.sub(r"\s+", " ", candidates[0]).strip()[:80]
        if nearby:
            return f"Illustration related to: {nearby}"
        return f"Illustration on page {page_num}, image {image_index + 1}"

    @staticmethod
    def _image_caption(
        text_segments: list[tuple[float, str]] | None, top: float
    ) -> str:
        """Return an explicit nearby figure caption when one is recognizable."""
        if not text_segments:
            return ""
        candidates = [
            (abs(segment_top - top), text.strip())
            for segment_top, text in text_segments
            if abs(segment_top - top) <= 110
        ]
        caption_pattern = re.compile(
            r"^(?:fig(?:ure)?\.?|illustration|plate|photo(?:graph)?|image|table)"
            r"\s*(?:\d+[A-Za-z]?)?\s*[:.\-]\s*.+$",
            re.IGNORECASE,
        )
        explicit = [
            (distance, text)
            for distance, text in candidates
            if 8 <= len(text) <= 180 and caption_pattern.match(text)
        ]
        if not explicit:
            return ""
        return min(explicit, key=lambda item: item[0])[1]

    @staticmethod
    def _image_parts(image: tuple) -> tuple[int, str, float, int]:
        """Normalize legacy and position-aware image tuples."""
        if len(image) >= 4:
            return image[0], image[1], image[2], image[3]
        return image[0], image[1], float("inf"), 0

    def _apply_repeated_header_policy(
        self, chapters: list[epub.EpubHtml]
    ) -> None:
        """Learn recurring top-margin headers and optionally remove them."""
        candidates_by_page = {
            getattr(chapter, "page_number", index + 1): getattr(
                chapter, "_page_metadata", {}
            ).get("header_candidates", [])
            for index, chapter in enumerate(chapters)
        }
        normalized = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
        header_pages = {}
        for page_number, candidates in candidates_by_page.items():
            for candidate in candidates:
                key = normalized(candidate)
                if key:
                    header_pages.setdefault(key, set()).add(page_number)
        repeated = {
            value for value, pages in header_pages.items() if len(pages) >= 2
        }
        self._stats["repeated_headers"] = sorted(repeated)
        if self.repeated_header_action != "filter" or not repeated:
            return

        for chapter in chapters:
            metadata = getattr(chapter, "_page_metadata", {})
            body_lines = [
                line
                for line in metadata.get("body_text", "").splitlines()
                if normalized(line) not in repeated
            ]
            header_lines = [
                line
                for line in metadata.get("header_text", "").splitlines()
                if normalized(line) not in repeated
            ]
            body_text = "\n".join(body_lines).strip()
            header_text = "\n".join(header_lines).strip()
            text_segments = [
                (top, line)
                for top, line in metadata.get("text_segments", [])
                if normalized(line) not in repeated
            ]
            semantic_segments = [
                segment
                for segment in metadata.get("semantic_segments", [])
                if normalized(str(segment.get("text", ""))) not in repeated
            ]
            positioned_images = []
            for image in metadata.get("images", []):
                index, img_html, top, _insertion_index = self._image_parts(image)
                insertion_index = sum(
                    segment_top <= top for segment_top, _line in text_segments
                )
                positioned_images.append((index, img_html, top, insertion_index))
            metadata["body_text"] = body_text
            metadata["header_text"] = header_text
            metadata["text_segments"] = text_segments
            metadata["semantic_segments"] = semantic_segments
            metadata["images"] = positioned_images
            chapter.source_text = "\n".join(
                part for part in (header_text, body_text) if part
            )
            content = self.create_epub_content(
                self.format_chapter_text(body_text),
                metadata.get("images", []),
                header_text,
                text_segments=text_segments,
                semantic_segments=semantic_segments,
                page_num=getattr(chapter, "page_number", None),
                section_kind=getattr(chapter, "_section_kind", "body"),
                section_title=getattr(chapter, "_section_title", ""),
            )
            if content:
                chapter.set_content(content)

    def _is_front_matter_title(self, title: str) -> bool:
        """Recognize common front-matter labels without requiring metadata."""
        return bool(
            re.search(
                r"\b(contents|copyright|dedication|foreword|preface|"
                r"author'?s note|introduction|prologue|acknowledg(?:e)?ments?)\b",
                title,
                re.IGNORECASE,
            )
        )

    def _is_primary_chapter_heading(self, title: str) -> bool:
        """Return whether a title is a likely body chapter rather than front matter."""
        return bool(
            re.match(r"^\s*(?:chapter|part|section)\b", title, re.IGNORECASE)
            or re.match(r"^\s*(?:\d{1,3}|[IVXLCDM]{1,8})[.)-]\s+", title, re.IGNORECASE)
        )

    def _chapter_display_title(
        self, chapter: epub.EpubHtml, fallback: str
    ) -> str:
        """Prefer a concise page heading over noisy contents-page OCR."""
        if getattr(chapter, "page_number", None) in self.chapter_overrides:
            return fallback
        candidates = [
            str(segment.get("text", "")).strip()
            for segment in getattr(chapter, "_page_metadata", {}).get(
                "semantic_segments", []
            )
            if segment.get("kind") == "heading"
        ]
        candidates = [
            candidate
            for candidate in candidates
            if self._looks_like_chapter_heading(candidate)
            and not self._is_front_matter_title(candidate)
        ]
        if candidates:
            return min(candidates, key=len)
        return fallback

    def _apply_book_semantics(self, chapters: list[epub.EpubHtml]) -> None:
        """Apply book-level section classes and reflowable page semantics."""
        if not chapters:
            return
        entries = self._identify_chapters(chapters)
        titles_by_page = {}
        for chapter, title in entries:
            page = getattr(chapter, "page_number", None)
            titles_by_page[page] = self._chapter_display_title(chapter, title)
        primary_pages = [
            page
            for page, title in titles_by_page.items()
            if page is not None and not self._is_front_matter_title(title)
            and (self._is_primary_chapter_heading(title) or len(entries) == 1)
        ]
        if not primary_pages:
            primary_pages = [
                getattr(chapter, "page_number", 1)
                for chapter in chapters
                if any(
                    segment.get("kind") == "heading"
                    and self._is_primary_chapter_heading(str(segment.get("text", "")))
                    for segment in getattr(chapter, "_page_metadata", {}).get(
                        "semantic_segments", []
                    )
                )
            ]
        first_body_page = min(primary_pages) if primary_pages else 1

        for chapter in chapters:
            page_num = getattr(chapter, "page_number", None)
            metadata = getattr(chapter, "_page_metadata", {})
            title = titles_by_page.get(page_num, "")
            if page_num is not None and page_num < first_body_page:
                section_kind = "frontmatter"
            elif page_num in titles_by_page:
                section_kind = "chapter"
            else:
                section_kind = "body"
            chapter._section_kind = section_kind
            chapter._section_title = title if section_kind == "chapter" else ""
            if title:
                chapter.title = title
            body_text = metadata.get("body_text", "")
            header_text = metadata.get("header_text", "")
            content = self.create_epub_content(
                self.format_chapter_text(body_text),
                metadata.get("images", []),
                header_text,
                text_segments=metadata.get("text_segments"),
                semantic_segments=metadata.get("semantic_segments"),
                page_num=page_num,
                section_kind=section_kind,
                section_title=chapter._section_title,
            )
            if content:
                chapter.set_content(content)

    def _resize_image(self, pil_img: Image.Image) -> bytes:
        """Resize image to fit within max dimensions while maintaining aspect ratio."""
        try:
            width, height = pil_img.size
            max_w, max_h = self.max_image_size

            # Check if resizing is needed
            if width <= max_w and height <= max_h:
                # Convert to JPEG directly
                img_buffer = BytesIO()
                pil_img.convert("RGB").save(
                    img_buffer, format="JPEG", quality=self.image_quality
                )
                img_buffer.seek(0)
                return img_buffer.getvalue()

            # Calculate new dimensions maintaining aspect ratio
            ratio = min(max_w / width, max_h / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)

            # Resize image using the high-quality LANCZOS filter.
            pil_resized = pil_img.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Convert to JPEG and save
            img_buffer = BytesIO()
            pil_resized.convert("RGB").save(
                img_buffer, format="JPEG", quality=self.image_quality
            )
            img_buffer.seek(0)

            return img_buffer.getvalue()

        except (OSError, RuntimeError, ValueError) as e:
            print(f"Warning: Could not resize image: {e}")
            # Convert original to JPEG if resizing fails
            img_buffer = BytesIO()
            pil_img.convert("RGB").save(
                img_buffer, format="JPEG", quality=self.image_quality
            )
            img_buffer.seek(0)
            return img_buffer.getvalue()

    def add_toc(self, epub_book: epub.EpubBook, chapters: list[epub.EpubHtml]):
        """Add a heading-based table of contents to the EPUB book."""
        entries = self._identify_chapters(chapters)
        if not entries:
            # Do not present every physical page as a chapter when heading
            # detection failed; that creates a misleading and unusable TOC.
            entries = [(chapters[0], "Start")] if chapters else []

        epub_book.toc = tuple(
            epub.Link(
                f"{chapter.file_name}#main-{getattr(chapter, 'page_number', None) or index}",
                title,
                chapter.id,
            )
            for index, (chapter, title) in enumerate(entries, start=1)
        )
        nav = epub.EpubNav(title="Table of Contents")
        epub_book.add_item(nav)
        epub_book.spine = ["nav", *chapters]

    @staticmethod
    def _validation_diagnostic(
        severity: str,
        code: str,
        message: str,
        path: str | None = None,
        **details,
    ) -> dict[str, object]:
        """Create a stable, JSON-serializable validation diagnostic."""
        diagnostic = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if path:
            diagnostic["path"] = path
        if details:
            diagnostic["details"] = details
        return diagnostic

    @staticmethod
    def _xml_local_name(tag: object) -> str:
        """Return an XML tag or attribute name without its namespace."""
        return str(tag).rsplit("}", 1)[-1].split(":", 1)[-1]

    @staticmethod
    def _href_target(source: str, href: str) -> tuple[str, str]:
        """Resolve an EPUB-relative href into a ZIP path and fragment."""
        parsed = urlsplit(str(href).strip())
        fragment = unquote(parsed.fragment)
        target = unquote(parsed.path)
        if not target:
            target = source
        elif target.startswith("/"):
            target = target[1:]
        else:
            target = posixpath.join(posixpath.dirname(source), target)
        return posixpath.normpath(target), fragment

    @staticmethod
    def _is_external_href(href: str) -> bool:
        """Return whether an href is an external URL or a non-resource scheme."""
        parsed = urlsplit(str(href).strip())
        return bool(parsed.scheme or parsed.netloc) and parsed.scheme not in {"file"}

    @staticmethod
    def _element_text(element: ElementTree.Element) -> str:
        """Extract normalized visible text from an XML element."""
        return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()

    @staticmethod
    def _is_image_resource(path: str, media_type: str = "") -> bool:
        """Recognize image resources without depending on a MIME database."""
        return media_type.startswith("image/") or Path(path).suffix.casefold() in {
            ".avif",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".svg",
            ".webp",
        }

    def _validate_image_bytes(
        self, data: bytes, path: str, media_type: str = ""
    ) -> tuple[bool, str | None]:
        """Check an embedded image using Pillow when available, with safe fallbacks."""
        if not data:
            return False, "image resource is empty"
        if Path(path).suffix.casefold() == ".svg" or media_type == "image/svg+xml":
            try:
                root = ElementTree.fromstring(data)
                if self._xml_local_name(root.tag) != "svg":
                    return False, "SVG root element is not <svg>"
                return True, None
            except ElementTree.ParseError:
                return False, "SVG is not well-formed XML"
        if Image is not None:
            try:
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                return True, None
            except (OSError, RuntimeError, ValueError) as exc:
                return False, f"image cannot be decoded: {exc}"
        signatures = (
            data.startswith(b"\xff\xd8\xff"),
            data.startswith(b"\x89PNG\r\n\x1a\n"),
            data.startswith((b"GIF87a", b"GIF89a")),
            data.startswith(b"RIFF") and data[8:12] == b"WEBP",
        )
        return (True, None) if any(signatures) else (False, "unknown image format")

    def validate_epub_diagnostics(self, output_path: str | Path) -> dict[str, object]:
        """Validate EPUB structure, content, links, accessibility, and quality."""
        errors: list[dict[str, object]] = []
        warnings: list[dict[str, object]] = []

        def add(
            severity: str,
            code: str,
            message: str,
            path: str | None = None,
            **details,
        ) -> None:
            target = errors if severity == "error" else warnings
            target.append(
                self._validation_diagnostic(
                    severity, code, message, path, **details
                )
            )

        diagnostics: dict[str, object] = {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "summary": {},
        }
        try:
            archive = zipfile.ZipFile(output_path)
        except (OSError, zipfile.BadZipFile) as exc:
            add("error", "invalid-zip", f"EPUB is not a readable ZIP archive: {exc}")
            diagnostics["summary"] = {"error_count": 1, "warning_count": 0}
            return diagnostics

        with archive:
            names = set(archive.namelist())
            required = {"mimetype", "META-INF/container.xml"}
            for missing in sorted(required - names):
                add("error", "missing-required-file", f"Missing required EPUB file: {missing}", missing)
            bad_entry = archive.testzip()
            if bad_entry is not None:
                add("error", "corrupt-zip-entry", f"Corrupt ZIP entry: {bad_entry}", bad_entry)
            if "mimetype" in names and archive.read("mimetype") != b"application/epub+zip":
                add("error", "invalid-mimetype", "EPUB mimetype is not exactly application/epub+zip", "mimetype")
            if "mimetype" in names and names and list(archive.namelist())[0] != "mimetype":
                add("warning", "mimetype-order", "mimetype is not the first ZIP entry", "mimetype")
            if not names:
                add("error", "empty-archive", "EPUB archive contains no files")
                diagnostics["summary"] = {"error_count": len(errors), "warning_count": len(warnings)}
                return diagnostics

            package_path = None
            container_root = None
            if "META-INF/container.xml" in names:
                try:
                    container_root = ElementTree.fromstring(
                        archive.read("META-INF/container.xml")
                    )
                except ElementTree.ParseError as exc:
                    add("error", "malformed-container", f"container.xml is malformed XML: {exc}", "META-INF/container.xml")
                if container_root is not None:
                    rootfiles = [
                        item for item in container_root.iter()
                        if self._xml_local_name(item.tag) == "rootfile"
                    ]
                    if not rootfiles:
                        add("error", "missing-rootfile", "container.xml has no rootfile entry", "META-INF/container.xml")
                    else:
                        package_path = rootfiles[0].get("full-path")
                        if not package_path or package_path not in names:
                            add("error", "missing-package", "container.xml rootfile does not point to an archive member", "META-INF/container.xml", target=package_path)

            package_root = None
            manifest: dict[str, dict[str, str]] = {}
            spine_ids: list[str] = []
            if package_path:
                try:
                    package_root = ElementTree.fromstring(archive.read(package_path))
                except (KeyError, ElementTree.ParseError) as exc:
                    add("error", "malformed-package", f"Package document is malformed XML: {exc}", package_path)
                if package_root is not None:
                    for item in package_root.iter():
                        if self._xml_local_name(item.tag) == "item":
                            item_id = item.get("id")
                            href = item.get("href")
                            if item_id and href:
                                target, _fragment = self._href_target(package_path, href)
                                manifest[item_id] = {
                                    "href": href,
                                    "path": target,
                                    "media_type": item.get("media-type", ""),
                                    "properties": item.get("properties", ""),
                                }
                            elif item_id or href:
                                add("error", "invalid-manifest-item", "Manifest item is missing id or href", package_path)
                        elif self._xml_local_name(item.tag) == "itemref":
                            itemref = item.get("idref")
                            if itemref:
                                spine_ids.append(itemref)
                    for item_id, item in manifest.items():
                        if item["path"] not in names and not self._is_external_href(item["href"]):
                            add("error", "missing-manifest-resource", f"Manifest resource is missing: {item['path']}", package_path, item_id=item_id)
                        elif (
                            item["path"] in names
                            and self._is_image_resource(item["path"], item["media_type"])
                        ):
                            valid_image, reason = self._validate_image_bytes(
                                archive.read(item["path"]),
                                item["path"],
                                item["media_type"],
                            )
                            if not valid_image:
                                add(
                                    "error",
                                    "invalid-image",
                                    f"Image is invalid: {reason}",
                                    item["path"],
                                    item_id=item_id,
                                )
                    for item_id in spine_ids:
                        if item_id not in manifest:
                            add("error", "invalid-spine-reference", f"Spine references unknown manifest item: {item_id}", package_path)

            xhtml_paths = {
                name for name in names
                if name.casefold().endswith((".xhtml", ".html"))
            }
            xhtml_paths.update(
                item["path"] for item in manifest.values()
                if item["media_type"] in {"application/xhtml+xml", "text/html"}
                and item["path"] in names
            )
            parsed_docs: dict[str, ElementTree.Element] = {}
            ids_by_path: dict[str, set[str]] = {}
            links: list[tuple[str, str, str]] = []
            page_text: dict[str, str] = {}
            page_images: dict[str, int] = {}
            for path in sorted(xhtml_paths):
                try:
                    root = ElementTree.fromstring(archive.read(path))
                except (KeyError, ElementTree.ParseError) as exc:
                    add("error", "malformed-xhtml", f"XHTML document is malformed XML: {exc}", path)
                    continue
                parsed_docs[path] = root
                ids: set[str] = set()
                for element in root.iter():
                    element_id = element.get("id")
                    if element_id:
                        if element_id in ids:
                            add("error", "duplicate-id", f"Duplicate id '{element_id}' in XHTML document", path, id=element_id)
                        ids.add(element_id)
                ids_by_path[path] = ids
                page_text[path] = self._element_text(root)
                image_count = 0
                for element in root.iter():
                    tag = self._xml_local_name(element.tag).casefold()
                    for attr_name, attr_value in element.attrib.items():
                        attribute = self._xml_local_name(attr_name).casefold()
                        if not attr_value:
                            continue
                        if attribute in {"href", "src", "data", "poster"}:
                            links.append((path, str(attr_value), tag))
                    if tag == "img":
                        image_count += 1
                        alt = element.get("alt")
                        if alt is None:
                            add("warning", "missing-image-alt", "Image is missing alt text", path)
                        elif not alt.strip():
                            add("warning", "empty-image-alt", "Image has empty alt text; verify it is intentionally decorative", path)
                page_images[path] = image_count

                for section in root.iter():
                    tag = self._xml_local_name(section.tag).casefold()
                    classes = set(str(section.get("class", "")).split())
                    is_page = tag in {"main", "body"} or "page-section" in classes
                    is_section = tag in {"section", "article"} or "content-section" in classes
                    if is_page or is_section:
                        has_text = bool(self._element_text(section))
                        has_image = any(
                            self._xml_local_name(child.tag).casefold() == "img"
                            for child in section.iter()
                        )
                        if not has_text and not has_image:
                            add("warning", "empty-section", "Section/page has no text or images", path)

                headings = [
                    element for element in root.iter()
                    if re.fullmatch(r"h[1-6]", self._xml_local_name(element.tag).casefold())
                ]
                previous_level = None
                for heading in headings:
                    heading_text = self._element_text(heading)
                    if not heading_text:
                        add("warning", "empty-heading", "Heading has no visible text", path)
                    level = int(self._xml_local_name(heading.tag)[1])
                    if previous_level is not None and level > previous_level + 1:
                        add("warning", "heading-level-jump", f"Heading level jumps from h{previous_level} to h{level}", path, from_level=previous_level, to_level=level)
                    previous_level = level

            # Resolve every internal XHTML/resource reference and verify fragments.
            for source, href, tag in links:
                if self._is_external_href(href):
                    continue
                target, fragment = self._href_target(source, href)
                if target not in names:
                    add("error", "broken-resource-link", f"Link target does not exist: {href}", source, target=target, element=tag)
                    continue
                if fragment and target in ids_by_path and fragment not in ids_by_path[target]:
                    add("error", "broken-fragment-link", f"Link fragment does not exist: {href}", source, target=target, fragment=fragment)
                if tag == "img":
                    media_type = next(
                        (item["media_type"] for item in manifest.values() if item["path"] == target),
                        "",
                    )
                    if not self._is_image_resource(target, media_type):
                        add("error", "invalid-image-resource", f"Image points to a non-image resource: {href}", source)
                    else:
                        try:
                            valid_image, reason = self._validate_image_bytes(
                                archive.read(target), target, media_type
                            )
                        except KeyError:
                            valid_image, reason = False, "image resource is missing"
                        if not valid_image:
                            add("error", "invalid-image", f"Image is invalid: {reason}", target)

            # CSS can contain resource links that are not represented by XHTML attributes.
            for css_path in (
                item["path"] for item in manifest.values()
                if item["media_type"] == "text/css" and item["path"] in names
            ):
                try:
                    css_text = archive.read(css_path).decode("utf-8")
                except (UnicodeDecodeError, KeyError) as exc:
                    add("error", "invalid-stylesheet", f"Stylesheet cannot be decoded: {exc}", css_path)
                    continue
                for raw_href in re.findall(r"url\(\s*['\"]?([^'\")\s]+)", css_text, re.IGNORECASE):
                    if raw_href.startswith("data:") or self._is_external_href(raw_href):
                        continue
                    target, _fragment = self._href_target(css_path, raw_href)
                    if target not in names:
                        add("error", "broken-resource-link", f"Stylesheet resource does not exist: {raw_href}", css_path, target=target)

            # Validate OPF/nav TOC destinations separately so malformed navigation is explicit.
            nav_paths = {
                path for path, item in (
                    (entry["path"], entry) for entry in manifest.values()
                )
                if "nav" in item["properties"].split() or Path(path).name.casefold() == "nav.xhtml"
            }
            nav_paths.update(path for path in xhtml_paths if Path(path).name.casefold() == "nav.xhtml")
            for nav_path in sorted(nav_paths):
                root = parsed_docs.get(nav_path)
                if root is None:
                    continue
                for element in root.iter():
                    if self._xml_local_name(element.tag).casefold() != "a":
                        continue
                    href = element.get("href")
                    if not href or self._is_external_href(href):
                        continue
                    target, fragment = self._href_target(nav_path, href)
                    if target not in names or (fragment and fragment not in ids_by_path.get(target, set())):
                        add("error", "invalid-toc-target", f"TOC/nav target is invalid: {href}", nav_path, target=target, fragment=fragment)

            # NCX is still common in EPUB 2-compatible output.
            for ncx_path in (
                path for path, item in manifest.items()
                if item["media_type"] == "application/x-dtbncx+xml"
            ):
                path = manifest[ncx_path]["path"]
                try:
                    ncx_root = ElementTree.fromstring(archive.read(path))
                except (KeyError, ElementTree.ParseError) as exc:
                    add("error", "malformed-ncx", f"NCX document is malformed XML: {exc}", path)
                    continue
                for element in ncx_root.iter():
                    if self._xml_local_name(element.tag).casefold() != "content":
                        continue
                    href = element.get("src")
                    if href and not self._is_external_href(href):
                        target, fragment = self._href_target(path, href)
                        if target not in names or (fragment and fragment not in ids_by_path.get(target, set())):
                            add("error", "invalid-toc-target", f"NCX target is invalid: {href}", path, target=target, fragment=fragment)

            # Compare emitted text to converter page statistics when available.
            source_stats = {
                f"page_{item.get('page')}.xhtml": item
                for item in self._stats.get("page_statistics", [])
                if item.get("page") is not None
            }
            for path, text in page_text.items():
                stat = source_stats.get(Path(path).name)
                if not stat or Path(path).name.casefold() == "nav.xhtml":
                    continue
                expected = int(stat.get("text_characters") or 0)
                actual = len(text)
                images = page_images.get(path, 0)
                if expected >= 80 and actual < max(20, int(expected * 0.35)):
                    add("warning", "suspicious-text-loss", "Emitted page text is substantially shorter than extracted source text", path, expected_characters=expected, actual_characters=actual, images=images)
                elif expected >= 20 and actual == 0 and not images:
                    add("warning", "empty-page", "Page emitted no text or images despite extracted source text", path, expected_characters=expected)

            # Repeated full-page text is a quality warning, not an invalid EPUB.
            seen_text: dict[str, str] = {}
            for path, text in page_text.items():
                normalized = re.sub(r"\s+", " ", text).strip().casefold()
                if len(normalized) < 80 or Path(path).name.casefold() == "nav.xhtml":
                    continue
                if normalized in seen_text:
                    add("warning", "duplicate-page-content", "Page content duplicates another XHTML page", path, other_page=seen_text[normalized])
                else:
                    seen_text[normalized] = path

        diagnostics["valid"] = not errors
        diagnostics["summary"] = {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "xhtml_count": len(parsed_docs) if "parsed_docs" in locals() else 0,
        }
        return diagnostics

    def validate_epub(self, output_path: str | Path) -> bool:
        """Return whether an EPUB passes validation errors (warnings do not fail it)."""
        diagnostics = self.validate_epub_diagnostics(output_path)
        return bool(diagnostics["valid"])

    def _write_report(self, pdf_path: str, output_path: str | Path) -> None:
        """Write an optional JSON report, without affecting conversion success."""
        if not self.report_path:
            return
        ocr_diagnostics = self._ocr_diagnostics()
        self._stats["ocr_diagnostics"] = ocr_diagnostics
        report = {
            "input": str(pdf_path),
            "output": str(output_path),
            "settings": {
                "ocr_enabled": self.ocr_enabled,
                "ocr_language": self.ocr_language,
                "ocr_device": self.ocr_device,
                "ocr_confidence": self.ocr_confidence,
                "ocr_preprocess": self.ocr_preprocess,
                "ocr_retry": self.ocr_retry,
                "ocr_retry_dpi": self.ocr_retry_dpi,
                "ocr_retry_confidence": self.ocr_retry_confidence,
                "layout": self.layout,
                "language": self.language,
                "repeated_header_action": self.repeated_header_action,
                "image_placement": self.image_placement,
                "chapter_overrides": self.chapter_overrides,
                "workers": self.workers,
            },
            "stats": self._stats,
            "validation": self._stats.get("validation"),
            "ocr_diagnostics": ocr_diagnostics,
            "preflight": self.preflight_environment(
                self.ocr_enabled, self.ocr_device
            ),
        }
        try:
            report_path = Path(self.report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            print(f"Warning: Could not write report '{self.report_path}': {exc}")

    def _identify_chapters(
        self, chapters: list[epub.EpubHtml]
    ) -> list[tuple[epub.EpubHtml, str]]:
        """Find likely chapter-start pages from the collected page text."""
        override_entries = []
        override_pages = set()
        for page_number, title in sorted(self.chapter_overrides.items()):
            chapter = next(
                (
                    item
                    for item in chapters
                    if getattr(item, "page_number", None) == page_number
                ),
                None,
            )
            if chapter is not None:
                override_entries.append((chapter, title))
                override_pages.add(page_number)

        automatic_entries = self._identify_contents_entries(chapters)
        if not automatic_entries:
            seen_titles = set()
            for chapter in chapters:
                if getattr(chapter, "page_number", None) in override_pages:
                    continue
                text = getattr(chapter, "source_text", "")
                lines = [
                    re.sub(r"\s+", " ", line).strip(" -:|")
                    for line in text.splitlines()
                ]
                lines = [
                    line
                    for line in lines
                    if line
                    and not re.fullmatch(r"Page\s+\d+", line, re.IGNORECASE)
                ]

                heading = None
                for index, line in enumerate(lines[:12]):
                    if re.fullmatch(r"\d+", line) and index + 1 < len(lines):
                        candidate = lines[index + 1]
                        if self._looks_like_chapter_heading(candidate):
                            heading = candidate
                            break
                    numbered_heading = re.sub(
                        r"^(?:chapter|part|section)?\s*\d{1,3}[.)-]\s*",
                        "",
                        line,
                        flags=re.IGNORECASE,
                    ).strip()
                    if numbered_heading != line and self._looks_like_chapter_heading(
                        numbered_heading
                    ):
                        heading = numbered_heading
                        break
                    if (
                        index < 6
                        and self._looks_like_chapter_heading(line)
                        and (
                            re.match(r"^(chapter|part|section)\b", line, re.IGNORECASE)
                            or (
                                index + 1 < len(lines)
                                and (
                                    len(lines[index + 1]) >= 40
                                    or lines[index + 1].endswith((".", "!", "?"))
                                )
                            )
                        )
                    ):
                        heading = line
                        break

                if heading:
                    key = heading.casefold()
                    if key not in seen_titles:
                        automatic_entries.append((chapter, heading))
                        seen_titles.add(key)

        return override_entries + [
            entry
            for entry in automatic_entries
            if getattr(entry[0], "page_number", None) not in override_pages
        ]

    def _is_contents_continuation_page(self, chapter: epub.EpubHtml) -> bool:
        """Identify a following page that continues a multi-page contents list."""
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in getattr(chapter, "source_text", "").splitlines()
            if line.strip()
        ]
        numbered_lines = sum(
            bool(re.match(r"^\s*(?:\d{1,3}|[IVXLCDM]+)[.)_-]?\s*", line))
            for line in lines
        )
        return numbered_lines >= 2

    def _identify_contents_entries(
        self, chapters: list[epub.EpubHtml]
    ) -> list[tuple[epub.EpubHtml, str]]:
        """Use an OCR'd Contents page to locate chapter-start pages."""
        contents_index = next(
            (
                index
                for index, chapter in enumerate(chapters)
                if re.search(
                    r"\bcontents\b",
                    getattr(chapter, "source_text", ""),
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if contents_index is None:
            return []

        contents_page_indices = [contents_index]
        for index in range(contents_index + 1, len(chapters)):
            if not self._is_contents_continuation_page(chapters[index]):
                break
            contents_page_indices.append(index)

        contents_text = " ".join(
            getattr(chapters[index], "source_text", "")
            for index in contents_page_indices
        )
        contents_lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in contents_text.splitlines()
            if line.strip()
        ]
        contents_text = " ".join(contents_lines)
        # Front matter is classified separately by _apply_book_semantics and
        # should not become a chapter entry based only on the contents page.
        titles = []
        # Also accept OCR'd Contents lines such as "1. The Beginning .... 7".
        for line in contents_lines:
            if not re.match(r"^\s*(?:\d{1,3}|[IVXLCDM]+)[.)_-]?\s*", line):
                continue
            cleaned = re.sub(r"[._-]{2,}\s*\d+\s*$", "", line).strip()
            cleaned = re.sub(r"^\s*(?:\d{1,3}|[IVXLCDM]+)[.)_-]?\s*", "", cleaned)
            if self._looks_like_chapter_heading(cleaned):
                titles.append(cleaned)
        numbered_entries = re.findall(
            r"(?<!\w)\d{1,2}\s*[_\.\-]?\s*"
            r"([A-Za-z][A-Za-z0-9_'’-]*(?:\s+[A-Za-z][A-Za-z0-9_'’-]*){0,10})",
            contents_text,
        )
        for entry in numbered_entries:
            title = re.sub(r"[_]+", " ", entry)
            title = re.sub(r"(?:[._-]{2,}\s*|\s+)\d{1,4}\s*$", "", title)
            title = re.sub(r"\s+", " ", title).strip(" .:-")
            if 3 <= len(title) <= 80:
                titles.append(title)

        titles = [
            re.split(
                r"\bAN\s+ACTOR(?:['’]?S)?\s+ACTOR\b|"
                r"\bAN\s*ACTORS?\s*ACTOR\b",
                title,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .:-_")
            for title in titles
        ]
        entries = []
        seen_pages = set()
        seen_titles = set()
        contents_pages = {
            getattr(chapters[index], "page_number", index + 1)
            for index in contents_page_indices
        }
        for title in titles:
            title_key = re.sub(r"[^a-z0-9]", "", title.casefold())
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            title_tokens = self._heading_tokens(title)
            if not title_tokens:
                continue
            best_match = None
            for chapter in chapters:
                page_number = getattr(chapter, "page_number", None)
                if page_number in contents_pages:
                    continue
                semantic_headings = [
                    str(segment.get("text", "")).strip()
                    for segment in getattr(chapter, "_page_metadata", {}).get(
                        "semantic_segments", []
                    )
                    if segment.get("kind") == "heading"
                ]
                candidates = semantic_headings or self._page_heading_lines(chapter)[:8]
                for candidate in candidates:
                    candidate_key = re.sub(r"[^a-z0-9]", "", candidate.casefold())
                    candidate_key = re.sub(r"^\d+", "", candidate_key)
                    candidate_tokens = self._heading_tokens(candidate)
                    overlap = len(title_tokens & candidate_tokens)
                    required = max(2, int(len(title_tokens) * 0.7))
                    exact = title_key in candidate_key
                    if not exact and overlap < required:
                        continue
                    score = (100 if exact else 0) + overlap
                    match = (score, -(page_number or 0), chapter, candidate)
                    if best_match is None or match[:2] > best_match[:2]:
                        best_match = match
            if best_match is not None:
                _score, _page_order, chapter, _candidate = best_match
                if chapter.file_name not in seen_pages:
                    entries.append((chapter, title))
                    seen_pages.add(chapter.file_name)

        return sorted(
            entries,
            key=lambda item: getattr(item[0], "page_number", 0) or 0,
        )

    def _page_heading_lines(self, chapter: epub.EpubHtml) -> list[str]:
        """Return early non-header lines from a page's collected text."""
        lines = [
            re.sub(r"\s+", " ", line).strip(" -:|")
            for line in getattr(chapter, "source_text", "").splitlines()
        ]
        return [
            line
            for line in lines
            if line
            and not re.fullmatch(r"Page\s+\d+", line, re.IGNORECASE)
            and not re.search(r"AN\s*ACTORS?\s*ACTOR", line, re.IGNORECASE)
        ]

    def _heading_tokens(self, text: str) -> set[str]:
        """Normalize heading words for OCR-tolerant matching."""
        tokens = set()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text):
            normalized = token.casefold()
            if normalized in {"author's", "authors"}:
                normalized = "author"
            elif normalized in {"co-author's", "co-authors"}:
                normalized = "co-author"
            tokens.add(normalized)
        return tokens

    def _looks_like_chapter_heading(self, line: str) -> bool:
        """Identify concise OCR lines that look like chapter headings."""
        if not 3 <= len(line) <= 80 or len(line.split()) > 10:
            return False
        if re.search(r"\bAN\s*ACTORS?\s*ACTOR\b", line, re.IGNORECASE):
            return False
        if re.match(r"^(chapter|part|section)\b", line, re.IGNORECASE):
            return True

        letters = [char for char in line if char.isalpha()]
        uppercase_ratio = (
            sum(char.isupper() for char in letters) / len(letters) if letters else 0
        )
        if uppercase_ratio >= 0.8 and len(letters) >= 5:
            return True

        words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
        title_case_words = sum(word[0].isupper() for word in words)
        return (
            len(words) >= 2
            and title_case_words / len(words) >= 0.6
            and not line.endswith((".", ",", ";", ":"))
        )

    def add_styles(self, epub_book: epub.EpubBook) -> epub.EpubItem:
        """Add the shared stylesheet to the EPUB book."""
        style = epub.EpubItem(
            uid="css",
            file_name="style.css",
            media_type="text/css",
            content=self._get_default_styles(),
        )
        epub_book.add_item(style)
        return style

    def _get_default_styles(self) -> str:
        """Return default CSS styles for the EPUB."""
        return """
        :root {
            --epub-font-size: 1em;
            --epub-line-height: 1.5;
            --epub-page-margin: 5%;
            --epub-paragraph-indent: 1.25em;
            --epub-heading-before: 1.6em;
            --epub-heading-after: 0.55em;
        }

        html {
            font-size: 100%;
            text-rendering: optimizeLegibility;
        }

        body {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 12pt;
            font-size: var(--epub-font-size, 1em);
            line-height: var(--epub-line-height, 1.5);
            margin: 0 var(--epub-page-margin, 5%);
            padding: 0;
            widows: 2;
            orphans: 2;
        }

        main {
            display: block;
        }

        .page-section {
            display: block;
            max-width: 42em;
            margin: 0 auto;
            padding: 0.5em 0 1em;
        }

        .page-section.frontmatter {
            max-width: 38em;
            color: #333;
        }

        .page-section.chapter {
            break-before: page;
            page-break-before: always;
        }

        .page-section.body {
            break-before: auto;
            page-break-before: auto;
        }

        .content-section {
            display: block;
            margin: 0 0 1.1em;
        }

        p {
            margin: 0;
            text-align: justify;
            text-justify: inter-word;
            hyphens: auto;
            overflow-wrap: break-word;
            text-indent: var(--epub-paragraph-indent, 1.25em);
        }

        p + p {
            margin-top: 0.55em;
        }

        h1 + p, h2 + p, h3 + p,
        .page-header + main p,
        figure + p, p + figure {
            text-indent: 0;
        }

        h1, h2, h3 {
            font-family: Georgia, 'Times New Roman', serif;
            line-height: 1.2;
            margin: var(--epub-heading-before, 1.6em) 0
                var(--epub-heading-after, 0.55em);
            text-align: left;
            page-break-after: avoid;
            break-after: avoid;
            text-indent: 0;
        }

        h1 { font-size: 1.55em; }
        h2 { font-size: 1.25em; }
        h3 { font-size: 1.1em; }

        .page-header {
            margin: 0 0 0.35em;
            padding-bottom: 0.2em;
            border-bottom: 1px solid #999;
            font-size: 0.85em;
            font-weight: bold;
            text-align: center;
        }

        .page-break {
            display: block;
            height: 0;
            break-before: page;
            page-break-before: always;
        }

        .visually-hidden {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        figure.illustration {
            margin: 1em auto;
            max-width: 100%;
            break-inside: avoid;
            page-break-inside: avoid;
            text-align: center;
        }

        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.35em auto 0;
        }

        figcaption {
            margin: 0.45em auto 0;
            max-width: 38em;
            font-size: 0.9em;
            line-height: 1.3;
            text-align: center;
            text-indent: 0;
            font-style: italic;
        }

        @media screen and (max-width: 45em) {
            :root {
                --epub-font-size: 0.95em;
                --epub-page-margin: 3%;
                --epub-paragraph-indent: 1em;
            }
            body {
                font-size: 10pt;
            }
        }

        @media screen and (min-width: 60em) {
            :root {
                --epub-font-size: 1.08em;
                --epub-page-margin: 8%;
            }
        }

        @media print {
            .page-break {
                break-before: page;
                page-break-before: always;
            }
        }
        """


def _load_chapter_overrides(value) -> dict[int, str]:
    """Load exact chapter page/title mappings from JSON or a config object."""
    if not value:
        return {}
    if isinstance(value, (str, os.PathLike)):
        with Path(value).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if isinstance(value, dict) and "chapters" in value:
        value = value["chapters"]

    entries = []
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).isdigit() and isinstance(item, str):
                entries.append({"page": key, "title": item})
            elif isinstance(item, int) and isinstance(key, str):
                entries.append({"page": item, "title": key})
            else:
                raise ValueError(
                    "chapter overrides must map page numbers to titles "
                    "or titles to page numbers"
                )
    else:
        raise ValueError("chapter overrides must be a JSON object or list")

    overrides = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each chapter override must be an object")
        page = entry.get("page")
        title = entry.get("title")
        if isinstance(page, bool) or not isinstance(page, int):
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError("chapter override pages must be integers") from exc
        if page < 1 or not isinstance(title, str) or not title.strip():
            raise ValueError(
                "each chapter override requires a positive page and title"
            )
        overrides[page] = title.strip()
    return overrides


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
        chapter_overrides = _load_chapter_overrides(args.chapter_overrides)
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

        try:
            epub.write_epub(
                output_path,
                epub_book,
                {"image_quality": args.quality, "epub3_landmark": True},
            )
        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error writing merged EPUB '{output_path}': {e}")
            return 1

        if args.validate:
            validation = converter.validate_epub_diagnostics(output_path)
            converter._stats["validation"] = validation
            if not validation["valid"]:
                print(
                    f"Warning: EPUB validation failed: {output_path} "
                    f"({len(validation['errors'])} error(s), "
                    f"{len(validation['warnings'])} warning(s))"
                )
        converter._stats["elapsed_seconds"] = round(
            time.perf_counter() - merge_started, 3
        )
        if args.report:
            converter._write_report(
                ", ".join(args.input),
                output_path,
            )
        print(f"Successfully merged {len(args.input)} PDF(s) into: {output_path}")
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
