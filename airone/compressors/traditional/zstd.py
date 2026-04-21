import zstandard as zstd
from ..base import BaseCompressor, CompressionResult
import time

class ZstdCompressor(BaseCompressor):
    name = "traditional_zstd"
    
    @property
    def name(self) -> str:
        return "traditional_zstd"
    
    def can_handle(self, analysis):
        return True  # Can compress anything
    
    def estimate_ratio(self, analysis):
        # Simple heuristic based on entropy
        if analysis and hasattr(analysis, 'entropy'):
            # Lower entropy = better compression
            return max(1.0, 8.0 - analysis.entropy)
        return 2.0  # Default estimate
    
    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        start = time.time()
        
        compressor = zstd.ZstdCompressor(level=19)
        compressed = compressor.compress(data)
        
        return CompressionResult(
            compressed_data=compressed,
            original_size=len(data),
            compressed_size=len(compressed),
            strategy_name=self.name,
            execution_time=time.time() - start,
            metadata={"level": 19}
        )
    
    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(compressed_data)
