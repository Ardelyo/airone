"""Tests for procedural compressors."""

import io
import pytest
from PIL import Image
from airone.compressors.procedural.gradient import GradientCompressor


@pytest.fixture
def horizontal_gradient_bytes():
    """200×100 horizontal gradient image as raw bytes."""
    img = Image.new("RGB", (200, 100))
    for x in range(200):
        intensity = round(x / 199 * 255)
        for y in range(100):
            img.putpixel((x, y), (intensity, intensity, intensity))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


@pytest.fixture
def vertical_gradient_bytes():
    """100×200 vertical gradient image as raw bytes."""
    img = Image.new("RGB", (100, 200))
    for y in range(200):
        intensity = round(y / 199 * 255)
        for x in range(100):
            img.putpixel((x, y), (intensity, intensity, intensity))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


@pytest.fixture
def natural_photo_bytes():
    """A non-gradient image (noisy) should NOT be handled."""
    import os
    img = Image.frombytes("RGB", (100, 100), os.urandom(100 * 100 * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGradientCompressor:

    def test_compress_and_decompress_horizontal(self, horizontal_gradient_bytes):
        compressor = GradientCompressor()
        result = compressor.compress(horizontal_gradient_bytes)
        assert result.ratio > 10           # Should be very high
        assert result.strategy_name == "procedural_gradient"

        restored = compressor.decompress(result.compressed_data, result.metadata)
        assert restored is not None
        assert len(restored) > 0

    def test_compress_and_decompress_vertical(self, vertical_gradient_bytes):
        compressor = GradientCompressor()
        result = compressor.compress(vertical_gradient_bytes)
        assert result.ratio > 10
        restored = compressor.decompress(result.compressed_data, result.metadata)
        assert restored is not None

    def test_payload_is_tiny(self, horizontal_gradient_bytes):
        """Compressed gradient should be < 500 bytes."""
        compressor = GradientCompressor()
        result = compressor.compress(horizontal_gradient_bytes)
        assert result.compressed_size < 500

    def test_rejects_non_gradient(self, natural_photo_bytes):
        """Random-noise image should raise CompressionError."""
        from airone.exceptions import CompressionError
        compressor = GradientCompressor()
        with pytest.raises(CompressionError):
            compressor.compress(natural_photo_bytes)
