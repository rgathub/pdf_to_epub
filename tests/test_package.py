import tempfile
import unittest
from pathlib import Path

from pdf_to_epub._engine import PDFToEPUBConverter, _same_path
from pdf_to_epub.validation import EPUBValidator


class PackageTests(unittest.TestCase):
    def test_no_ocr_preflight_is_ready(self):
        result = PDFToEPUBConverter.preflight_environment(False)
        self.assertTrue(result["ready"])
        self.assertEqual(result["ocr_device"], "disabled")

    def test_cuda_preflight_reports_installed_runtime(self):
        result = PDFToEPUBConverter.preflight_environment(True, "cuda")
        self.assertIn("torch", result["optional"])
        self.assertIn("torchvision", result["optional"])
        self.assertIn("cuda_available", result)

    def test_path_comparison_is_case_insensitive(self):
        self.assertTrue(_same_path(r"C:\Books\Input.pdf", r"c:\books\INPUT.PDF"))

    def test_progress_callback_receives_structured_event(self):
        events = []
        converter = PDFToEPUBConverter(
            ocr_enabled=False,
            progress_callback=events.append,
        )
        converter._report_progress(2, 4)
        self.assertEqual(
            events,
            [{"event": "page", "page": 2, "pages": 4, "percent": 50.0}],
        )

    def test_language_fallback_does_not_leak_between_documents(self):
        converter = PDFToEPUBConverter(ocr_enabled=False, language="en")

        class Document:
            def __init__(self, metadata):
                self.metadata = metadata

        self.assertEqual(
            converter._extract_pdf_metadata(
                Document({"language": "fr"}), "first.pdf"
            )["language"],
            "fr",
        )
        self.assertEqual(
            converter._extract_pdf_metadata(Document({}), "second.pdf")["language"],
            "en",
        )

    def test_validator_rejects_non_epub(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.epub"
            path.write_bytes(b"not an epub")
            result = EPUBValidator().validate(path)
            self.assertFalse(result["valid"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
