"""PDF page extraction and page-level text/image processing."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Iterator
import html

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

from .config import _IMAGE_ERRORS


def _same_path(first: str | Path, second: str | Path) -> bool:
    """Compare paths using Windows-aware absolute, case-insensitive semantics."""
    import os
    return os.path.normcase(os.path.abspath(os.fspath(first))) == os.path.normcase(
        os.path.abspath(os.fspath(second))
    )


class PDFPageExtractor:
    """Lazily expose page count and page iteration through PyMuPDF."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

    def page_count(self) -> int:
        """Return the number of pages without retaining a document handle."""
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF page extraction")
        with fitz.open(str(self.pdf_path)) as document:
            return document.page_count

    def pages(self) -> Iterator[object]:
        """Yield PyMuPDF page objects while the document is open."""
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF page extraction")
        document = fitz.open(str(self.pdf_path))
        try:
            yield from document
        finally:
            document.close()


class PDFPageMixin:
    """Page metadata, extraction, grouping, and chapter construction behavior."""

    def _extract_pdf_metadata(self, doc, pdf_path: str) -> dict[str, str]:
        """Normalize PDF metadata for EPUB Dublin Core fields."""
        raw = doc.metadata or {}
        title = (raw.get("title") or "").strip() or self._extract_title(pdf_path)
        language = (raw.get("language") or "").strip() or self._default_language
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
            except _IMAGE_ERRORS:
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

            except (KeyError, TypeError) + _IMAGE_ERRORS as e:
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
