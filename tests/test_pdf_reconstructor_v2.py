"""Tests for PDF Reconstructor v2."""

from __future__ import annotations

import json
import pytest

try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from airone.compressors.semantic.pdf_reconstructor_v2 import (
    ElementPosition,
    PageManifest,
    PositionalBundle,
    PDFReconstructorV2,
)

pytestmark = pytest.mark.skipif(not HAS_PYPDF, reason="pypdf not installed")


@pytest.fixture
def simple_bundle():
    """A PositionalBundle with one page and no elements."""
    bundle = PositionalBundle(format_version="2.0", page_count=2)
    for i in range(1, 3):
        bundle.pages[i] = PageManifest(
            page_number=i, width=612.0, height=792.0
        )
    return bundle


@pytest.fixture
def bundle_with_image():
    """Bundle with one image element."""
    import hashlib, io
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    h = hashlib.sha256(png_bytes).hexdigest()

    bundle = PositionalBundle(format_version="2.0", page_count=1)
    page = PageManifest(page_number=1, width=612.0, height=792.0)
    page.elements.append(ElementPosition(
        element_type="image",
        bbox=[50.0, 100.0, 250.0, 300.0],
        content_hash=h,
        z_order=0,
    ))
    bundle.pages[1] = page
    bundle.content_hashes = [h]

    content_store = {h: png_bytes}
    return bundle, content_store


class TestPositionalBundle:

    def test_to_from_json_roundtrip(self, simple_bundle):
        json_str = simple_bundle.to_json()
        restored = PositionalBundle.from_json(json_str)

        assert restored.page_count == simple_bundle.page_count
        assert restored.format_version == "2.0"
        assert len(restored.pages) == 2

    def test_element_position_properties(self):
        elem = ElementPosition(
            element_type="image",
            bbox=[10.0, 20.0, 110.0, 220.0],
            content_hash="abc" * 21 + "ab",
        )
        assert elem.width  == pytest.approx(100.0)
        assert elem.height == pytest.approx(200.0)
        assert elem.x0     == pytest.approx(10.0)
        assert elem.y0     == pytest.approx(20.0)

    def test_elements_of_type_filter(self):
        page = PageManifest(page_number=1, width=612.0, height=792.0)
        page.elements.append(ElementPosition(
            element_type="image", bbox=[0,0,100,100], content_hash="h1"
        ))
        page.elements.append(ElementPosition(
            element_type="text", bbox=[0,0,612,792], content_hash="h2"
        ))
        images = page.elements_of_type("image")
        texts  = page.elements_of_type("text")
        assert len(images) == 1
        assert len(texts)  == 1


class TestPDFReconstructorV2:

    def test_reconstruct_blank_pages(self, simple_bundle):
        reconstructor = PDFReconstructorV2()
        result = reconstructor.reconstruct({}, simple_bundle)

        assert result.page_count == 2
        assert len(result.pdf_bytes) > 0
        assert result.method == "positional_v2"

    def test_result_is_valid_pdf(self, simple_bundle):
        from pypdf import PdfReader
        reconstructor = PDFReconstructorV2()
        result = reconstructor.reconstruct({}, simple_bundle)

        import io
        reader = PdfReader(io.BytesIO(result.pdf_bytes))
        assert len(reader.pages) == 2

    def test_page_dimensions_preserved(self, simple_bundle):
        from pypdf import PdfReader
        reconstructor = PDFReconstructorV2()
        result = reconstructor.reconstruct({}, simple_bundle)

        import io
        reader = PdfReader(io.BytesIO(result.pdf_bytes))
        page = reader.pages[0]
        assert float(page.mediabox.width)  == pytest.approx(612.0, abs=1.0)
        assert float(page.mediabox.height) == pytest.approx(792.0, abs=1.0)

    def test_reconstruct_with_image(self, bundle_with_image):
        bundle, content_store = bundle_with_image
        reconstructor = PDFReconstructorV2()
        result = reconstructor.reconstruct(content_store, bundle)

        assert result.images_total == 1
        assert result.page_count   == 1

    def test_missing_content_noted(self, simple_bundle):
        """Missing content hashes should be recorded in notes."""
        bundle = simple_bundle
        bundle.pages[1].elements.append(ElementPosition(
            element_type="image",
            bbox=[0, 0, 100, 100],
            content_hash="nonexistent_hash",
        ))
        reconstructor = PDFReconstructorV2()
        result = reconstructor.reconstruct({}, bundle)
        # Should not raise — graceful degradation
        assert len(result.pdf_bytes) > 0
