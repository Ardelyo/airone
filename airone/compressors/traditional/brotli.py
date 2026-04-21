"""
AirOne Brotli Compressor
Better than ZSTD for compressible text and web assets.
Slower compression but slightly higher ratio for these types.
"""

from __future__ import annotations

import time

import brotli

from airone.compressors.base import BaseCompressor, CompressionResult
from airone.exceptions import CompressionError, DecompressionError


class BrotliCompressor(BaseCompressor):
    """
    Wraps the Brotli compression library.

    Optimal for:
        - HTML, CSS, JS, JSON (web assets)
        - Repetitive structured text
        - XML documents

    Quality range: 0 (fast) → 11 (maximum compression, slow)
    """

    name = "traditional_brotli"

    def __init__(self, quality: int = 9) -> None:
        if not 0 <= quality <= 11:
            raise ValueError("Brotli quality must be 0–11")
        self.quality = quality

    def can_handle(self, analysis) -> bool:
        if analysis is None:
            return True
        # Prefer Brotli for text, JSON, XML
        text_types = {"TXT", "JSON", "XML", "HTML", "CSS", "JS", "SVG"}
        return (
            hasattr(analysis, "format")
            and analysis.format.type in text_types
        )

    def estimate_ratio(self, analysis) -> float:
        if analysis and hasattr(analysis, "entropy"):
            e = analysis.entropy.global_entropy
            if e < 3.0:
                return 10.0
            if e < 5.0:
                return 5.0
            if e < 7.0:
                return 2.5
        return 2.0

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.perf_counter()
        try:
            compressed = brotli.compress(data, quality=self.quality)
        except brotli.error as exc:
            raise CompressionError(
                f"BrotliCompressor: compression failed: {exc}"
            ) from exc

        return CompressionResult(
            compressed_data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            strategy_name=self.name,
            execution_time=time.perf_counter() - start,
            metadata={"quality": self.quality},
        )

    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        try:
            return brotli.decompress(compressed_data)
        except brotli.error as exc:
            raise DecompressionError(
                f"BrotliCompressor: decompression failed: {exc}"
            ) from exc
