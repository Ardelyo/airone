import zstandard as zstd
from ..base import BaseCompressor, CompressionResult
import time

class ZstdCompressor(BaseCompressor):

    def __init__(self, level: int = 19) -> None:
        self._level = level

    @property
    def name(self) -> str:
        return "traditional_zstd"

    def can_handle(self, analysis):
        return True  # Can compress anything

    def estimate_ratio(self, analysis):
        if analysis and hasattr(analysis, 'entropy'):
            return max(1.0, 8.0 - analysis.entropy)
        return 2.0

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.time()

        compressor = zstd.ZstdCompressor(level=self._level)
        compressed = compressor.compress(data)

        return CompressionResult(
            compressed_data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            strategy_name=self.name,
            execution_time=time.time() - start,
            metadata={"level": self._level}
        )
    
    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(compressed_data)
