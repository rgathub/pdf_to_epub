"""Public conversion orchestrator composed from modular responsibilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import html
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
import traceback

try:
    import fitz
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

from .config import ConversionOptions, _discard_temporary_output, _temporary_output_path
from .epub_content import EPUBAssembler, EPUBContentMixin
from .ocr import OCRMixin
from .pdf_pages import PDFPageMixin, _same_path
from .validation import EPUBValidationMixin

_parallel_worker_state = threading.local()
logger = logging.getLogger(__name__)


class PDFToEPUBConverter(EPUBContentMixin, OCRMixin, PDFPageMixin, EPUBValidationMixin):
    """Convert PDF files to EPUB while coordinating modular services."""

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
        progress_callback=None,
        progress_bar: bool = False,
        options: ConversionOptions | None = None,
        ocr_confidence_threshold: float = 0.4,
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
            progress_callback: Optional callback receiving progress event dictionaries
            ocr_confidence_threshold: Minimum OCR confidence to retain text (default: 0.4)
        """
        if options is not None and isinstance(image_quality, ConversionOptions):
            raise TypeError("pass options either positionally or by keyword, not both")
        if isinstance(image_quality, ConversionOptions):
            options = image_quality
        if options is not None:
            image_quality = options.image_quality
            max_image_size = options.max_image_size
            ocr_enabled = options.ocr_enabled
            ocr_language = options.ocr_language
            ocr_dpi = options.ocr_dpi
            keep_scan_images = options.keep_scan_images
            ocr_device = options.ocr_device
            ocr_confidence = options.ocr_confidence
            ocr_preprocess = options.ocr_preprocess
            ocr_cache_dir = options.ocr_cache_dir
            resume = options.resume
            layout = options.layout
            validate_output = options.validate_output
            report_path = options.report_path
            repeated_header_action = options.repeated_header_action
            chapter_overrides = options.chapter_overrides
            image_placement = options.image_placement
            language = options.language
            ocr_retry = options.ocr_retry
            ocr_retry_dpi = options.ocr_retry_dpi
            ocr_retry_confidence = options.ocr_retry_confidence
            workers = options.workers
            progress_callback = options.progress_callback
        else:
            options = ConversionOptions(
                image_quality=image_quality,
                max_image_size=max_image_size,
                ocr_enabled=ocr_enabled,
                ocr_language=ocr_language,
                ocr_dpi=ocr_dpi,
                keep_scan_images=keep_scan_images,
                ocr_device=ocr_device,
                ocr_confidence=ocr_confidence,
                ocr_preprocess=ocr_preprocess,
                ocr_cache_dir=ocr_cache_dir,
                resume=resume,
                layout=layout,
                validate_output=validate_output,
                report_path=report_path,
                repeated_header_action=repeated_header_action,
                chapter_overrides=chapter_overrides or {},
                image_placement=image_placement,
                language=language,
            )
        self.ocr_retry_confidence = ocr_retry_confidence
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.ocr_retry_confidence = ocr_retry_confidence
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.options = options
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._epub_assembler = EPUBAssembler(image_quality)
        self.image_quality = image_quality
        self.max_image_size = max_image_size
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.keep_scan_images = keep_scan_images
        self.ocr_device = ocr_device.casefold()
        self.ocr_confidence = options.ocr_confidence or 0.0
        self.ocr_confidence_threshold = self.ocr_confidence if not options.ocr_confidence else options.ocr_confidence_threshold
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
        self._default_language = self.language
        self.ocr_retry = ocr_retry
        self.ocr_retry_dpi = ocr_retry_dpi or max(300, int(ocr_dpi * 1.5))
        self.ocr_retry_confidence = ocr_retry_confidence
        self.workers = workers
        self.progress_callback = progress_callback
        self._use_tqdm = progress_bar  # Use tqdm progress bar if enabled
        self._tqdm_bar = None
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
            "ocr_confidence": self.ocr_confidence_threshold,
            "ocr_preprocess": self.ocr_preprocess,
            "ocr_cache_dir": str(self.ocr_cache_dir)
            if self.ocr_cache_dir
            else None,
            "resume": self.resume,
            "layout": self.layout,
            "repeated_header_action": self.repeated_header_action,
            "chapter_overrides": dict(self.chapter_overrides),
            "image_placement": self.image_placement,
            "language": self._default_language,
            "ocr_retry": self.ocr_retry,
            "ocr_retry_dpi": self.ocr_retry_dpi,
            "ocr_retry_confidence": self.ocr_retry_confidence,
            "workers": 1,
            "progress_callback": None,
            "ocr_confidence_threshold": self.ocr_confidence_threshold,
        }


    def _report_progress(self, page_num: int, total_pages: int) -> None:
        """Emit a structured page-progress event and retain CLI progress output."""
        percent = page_num / total_pages * 100 if total_pages else 100.0
        event = {
            "event": "page",
            "page": page_num,
            "pages": total_pages,
            "percent": round(percent, 1),
        }
        if self.progress_callback is not None:
            self.progress_callback(event)
        
        # Initialize tqdm bar on first call for this conversion
        if self._tqdm_bar is None and total_pages > 0 and self._use_tqdm:
            try:
                self._tqdm_bar = tqdm(
                    total=total_pages,
                    desc="Converting",
                    unit=" page",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )
            except ImportError:
                self._use_tqdm = False
        
        if self._use_tqdm and self._tqdm_bar is not None:
            self._tqdm_bar.update(1)
        else:
            logger.info("Processing page %s/%s (%.1f%%)", page_num, total_pages, percent)
            if page_num == 1 or page_num % 10 == 0 or page_num == total_pages:
                print(
                    f"Processing page {page_num}/{total_pages} "
                    f"({percent:.1f}%)",
                    flush=True,
                )


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
        temporary_output: Path | None = None
        try:
            if _same_path(pdf_path, output_path):
                raise ValueError("Input and output paths must be different.")
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
                        self._report_progress(page_num, total_pages)
                        self._merge_page_result(
                            epub_book, chapters, result, style
                        )
                else:
                    for i, page in enumerate(doc):
                        page_num = i + 1
                        self._report_progress(page_num, total_pages)
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
            temporary_output = _temporary_output_path(output_path)
            self._epub_assembler.write(temporary_output, epub_book)
            if self.validate_output:
                validation = self.validate_epub_diagnostics(temporary_output)
                self._stats["validation"] = validation
                if not validation["valid"]:
                    print(
                        f"Error: EPUB validation failed: {output_path} "
                        f"({len(validation['errors'])} error(s), "
                        f"{len(validation['warnings'])} warning(s))"
                    )
                    self._stats["elapsed_seconds"] = round(
                        time.perf_counter() - started, 3
                    )
                    _discard_temporary_output(temporary_output)
                    temporary_output = None
                    self._write_report(pdf_path, output_path)
                    return False
            os.replace(temporary_output, output_path)
            temporary_output = None
            self._stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self._write_report(pdf_path, output_path)

            if self._use_tqdm and self._tqdm_bar is not None:
                self._tqdm_bar.close()
            
            print(f"Successfully converted: {pdf_path} -> {output_path}")
            if self._use_tqdm and self._tqdm_bar is not None:
                self._tqdm_bar.close()
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            _discard_temporary_output(temporary_output)
            logger.error("Conversion failed for %s: %s", pdf_path, e)
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
        for module in ("easyocr", "torch", "torchvision", "numpy"):
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
            or all(
                result["optional"][module]
                for module in ("easyocr", "torch", "torchvision", "numpy")
            )
        )
        return result


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
