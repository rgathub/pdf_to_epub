"""Structural EPUB validation and diagnostics."""

from __future__ import annotations

from io import BytesIO
import posixpath
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
import zipfile
import defusedxml.ElementTree as ElementTree

try:
    from PIL import Image
except ImportError:
    Image = None

from .config import _IMAGE_ERRORS


class EPUBValidationMixin:
    """Validation implementation shared by converters and standalone validators."""

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
            except _IMAGE_ERRORS as exc:
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


class EPUBValidator(EPUBValidationMixin):
    """Standalone validator for EPUB files produced by any converter."""

    def __init__(self) -> None:
        self._stats = {"page_statistics": []}

    def validate(self, output_path: str | Path) -> dict[str, object]:
        """Return structured diagnostics for an EPUB path."""
        return self.validate_epub_diagnostics(output_path)
