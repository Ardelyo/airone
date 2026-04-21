"""
AirOne LZMA Compressor
Highest ratio of the traditional codecs.
Best suited for large files where compression time is acceptable.
"""

from __future__ import annotations

import lzma
import time

from airone.compressors.base import BaseCompressor, CompressionResult
from airone.exceptions import CompressionError, DecompressionError


class LZMACompressor(BaseCompressor):
    """
    Wraps Python's built-in lzma module.

    Optimal for:
        - Large text corpora
        - Source code archives
        - Binary files with long-range repetitions (firmware, database dumps)

    Preset range: 0 (fast) → 9 (maximum compression)
    """

    name = "traditional_lzma"

    def __init__(self, preset: int = 6) -> None:
        if not 0 <= preset <= 9:
            raise ValueError("LZMA preset must be 0–9")
        self.preset = preset

    def can_handle(self, analysis) -> bool:
        # LZMA is a universal fallback — can handle anything
        return True

    def estimate_ratio(self, analysis) -> float:
        if analysis and hasattr(analysis, "entropy"):
            e = analysis.entropy.global_entropy
            if e < 3.0:
                return 12.0
            if e < 5.0:
                return 6.0
            if e < 7.0:
                return 3.0
        return 2.5

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.perf_counter()
        try:
            compressed = lzma.compress(data, preset=self.preset)
        except lzma.LZMAError as exc:
            raise CompressionError(
                f"LZMACompressor: compression failed: {exc}"
            ) from exc

        return CompressionResult(
            compressed_data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            strategy_name=self.name,
            execution_time=time.perf_counter() - start,
            metadata={"preset": self.preset},
        )

    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        try:
            return lzma.decompress(compressed_data)
        except lzma.LZMAError as exc:
            raise DecompressionError(
                f"LZMACompressor: decompression failed: {exc}"
            ) from exc
