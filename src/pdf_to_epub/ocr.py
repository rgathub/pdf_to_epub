"""OCR lifecycle, preprocessing, retries, and cache handling."""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    Image = ImageEnhance = ImageOps = None

from .config import _DECOMPRESSION_ERRORS


class OCRService:
    """Describe OCR settings and cache location for independent callers."""

    def __init__(self, enabled: bool = True, cache_dir: str | None = None):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def diagnostics(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "cache_dir": str(self.cache_dir) if self.cache_dir else None}


class OCRMixin:
    """EasyOCR implementation mixed into the conversion orchestrator."""

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
                    if confidence < self.ocr_confidence_threshold:
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
        try:
            if not force_render:
                ocr_image_xref = self._select_ocr_image(page)
                if ocr_image_xref is not None:
                    image_data = page.parent.extract_image(ocr_image_xref)
                    image = Image.open(BytesIO(image_data["image"]))
            if image is None:
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
        except _DECOMPRESSION_ERRORS as exc:
            raise RuntimeError(
                "OCR image exceeds Pillow's decompression safety limit."
            ) from exc
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
            and item["min_confidence"] < self.ocr_confidence_threshold
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
