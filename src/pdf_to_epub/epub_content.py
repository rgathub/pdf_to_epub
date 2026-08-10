"""EPUB content rendering, image encoding, and assembly."""

from __future__ import annotations

import html
import re
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from ebooklib import epub
except ImportError:
    epub = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import _IMAGE_ERRORS, _DECOMPRESSION_ERRORS


class EPUBAssembler:
    """Write an ebooklib book to an EPUB archive."""

    def __init__(self, image_quality: int = 85):
        self.image_quality = image_quality

    def write(self, output_path: str | Path, book: Any) -> None:
        if epub is None:
            raise RuntimeError("EbookLib is required for EPUB assembly")
        epub.write_epub(str(output_path), book, {"image_quality": self.image_quality, "epub3_landmark": True})


class EPUBContentMixin:
    """Accessible XHTML, CSS, and image rendering behavior."""

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

        except _IMAGE_ERRORS as e:
            if isinstance(e, _DECOMPRESSION_ERRORS):
                raise RuntimeError(
                    "Image exceeds Pillow's decompression safety limit."
                ) from e
            print(f"Warning: Could not resize image: {e}")
            # Convert original to JPEG if resizing fails
            img_buffer = BytesIO()
            pil_img.convert("RGB").save(
                img_buffer, format="JPEG", quality=self.image_quality
            )
            img_buffer.seek(0)
            return img_buffer.getvalue()


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
