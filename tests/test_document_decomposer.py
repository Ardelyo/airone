"""
Tests for the PDF DocumentDecomposer.
Uses lightweight synthetic PDFs built with pypdf's writer.
"""

from __future__ import annotations

import io
import json
import os

import pytest

try:
    import pypdf
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from airone.analysis.document_decomposer import (
    ComponentType,
    DocumentDecomposer,
    PDFDecomposer,
)
from airone.exceptions import FormatError


pytestmark = pytest.mark.skipif(
    not HAS_PYPDF, reason="pypdf not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_pdf(tmp_path) -> str:
    """A valid minimal PDF with one blank page."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    path = tmp_path / "minimal.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    return str(path)


@pytest.fixture
def multipage_pdf(tmp_path) -> str:
    """A PDF with three blank pages."""
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=612, height=792)
    path = tmp_path / "multipage.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    return str(path)


# ---------------------------------------------------------------------------
# PDFDecomposer tests
# ---------------------------------------------------------------------------

class TestPDFDecomposer:

    def test_decompose_minimal_pdf(self, minimal_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(minimal_pdf)

        assert result.format_type == "PDF"
        assert result.page_count == 1
        assert result.source_path == minimal_pdf

    def test_decompose_multipage_pdf(self, multipage_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(multipage_pdf)

        assert result.page_count == 3

    def test_metadata_component_present(self, minimal_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(minimal_pdf)

        assert result.metadata_block is not None
        assert result.metadata_block.component_type == ComponentType.METADATA

        meta = json.loads(result.metadata_block.data.decode("utf-8"))
        assert "page_count" in meta
        assert meta["page_count"] == 1

    def test_layout_component_present(self, minimal_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(minimal_pdf)

        assert result.layout_block is not None
        layout = json.loads(result.layout_block.data.decode("utf-8"))
        assert "pages" in layout
        assert len(layout["pages"]) == 1

    def test_missing_file_raises(self):
        decomposer = PDFDecomposer()
        with pytest.raises(FileNotFoundError):
            decomposer.decompose("/nonexistent/path/file.pdf")

    def test_all_components_have_hashes(self, minimal_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(minimal_pdf)

        for comp in result.all_components:
            assert isinstance(comp.content_hash, str)
            assert len(comp.content_hash) == 64  # SHA-256 hex

    def test_summary_is_string(self, minimal_pdf):
        decomposer = PDFDecomposer()
        result = decomposer.decompose(minimal_pdf)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "PDF" in summary


class TestDocumentDecomposerFacade:

    def test_routes_pdf(self, minimal_pdf):
        decomposer = DocumentDecomposer()
        result = decomposer.decompose(minimal_pdf)
        assert result.format_type == "PDF"

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_bytes(b"dummy content")
        decomposer = DocumentDecomposer()
        with pytest.raises(FormatError):
            decomposer.decompose(str(path))
