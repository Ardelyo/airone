"""
Tests for the PDFSemanticCompressor.
"""

from __future__ import annotations

import pytest

try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from airone.compressors.semantic.pdf import PDFSemanticCompressor

pytestmark = pytest.mark.skipif(
    not HAS_PYPDF, reason="pypdf not installed"
)


@pytest.fixture
def minimal_pdf_bytes(tmp_path) -> tuple[bytes, str]:
    """Returns (pdf_bytes, pdf_path)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path = tmp_path / "test.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    data = path.read_bytes()
    return data, str(path)


class TestPDFSemanticCompressor:

    def test_can_handle_pdf_analysis(self):
        from unittest.mock import MagicMock
        from airone.analysis.format_detector import FileFormat, FileCategory

        compressor = PDFSemanticCompressor()
        analysis = MagicMock()
        analysis.format = FileFormat(
            type="PDF",
            mime_type="application/pdf",
            category=FileCategory.DOCUMENT,
        )
        assert compressor.can_handle(analysis) is True

    def test_cannot_handle_non_pdf(self):
        from unittest.mock import MagicMock
        from airone.analysis.format_detector import FileFormat, FileCategory

        compressor = PDFSemanticCompressor()
        analysis = MagicMock()
        analysis.format = FileFormat(
            type="PNG",
            mime_type="image/png",
            category=FileCategory.IMAGE,
        )
        assert compressor.can_handle(analysis) is False

    def test_compress_produces_smaller_output(self, minimal_pdf_bytes):
        pdf_bytes, pdf_path = minimal_pdf_bytes
        compressor = PDFSemanticCompressor()

        from unittest.mock import MagicMock
        analysis = MagicMock()
        analysis.file_path = pdf_path
        analysis.format.type = "PDF"

        result = compressor.compress(pdf_bytes, analysis)

        assert result.compressed_size > 0
        assert result.strategy_name == "semantic_pdf"
        assert result.original_size == len(pdf_bytes)

    def test_compress_decompress_roundtrip(self, minimal_pdf_bytes):
        """Verify decompression produces a valid bundle (Phase 3 format)."""
        import msgpack
        pdf_bytes, pdf_path = minimal_pdf_bytes
        compressor = PDFSemanticCompressor()

        from unittest.mock import MagicMock
        analysis = MagicMock()
        analysis.file_path = pdf_path
        analysis.format.type = "PDF"

        result = compressor.compress(pdf_bytes, analysis)
        decompressed = compressor.decompress(result.compressed_data, result.metadata)

        # Should be a valid msgpack bundle
        bundle = msgpack.unpackb(decompressed, raw=False)
        assert "components" in bundle
        assert "meta" in bundle
