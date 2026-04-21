from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class CompressionResult:
    """Standard result format for all compressors"""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    strategy_name: str
    execution_time: float
    metadata: dict
    
    @property
    def ratio(self):
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 0

class BaseCompressor(ABC):
    """
    Interface all compressors must implement
    Ensures consistency across traditional, procedural, semantic, and neural
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique compressor identifier"""
        pass
    
    @abstractmethod
    def can_handle(self, analysis) -> bool:
        """Returns True if this compressor can handle the file"""
        pass
    
    @abstractmethod
    def estimate_ratio(self, analysis) -> float:
        """Predict compression ratio before attempting"""
        pass
    
    @abstractmethod
    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        """Compress data"""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        """Decompress data"""
        pass
    
    def supports_streaming(self) -> bool:
        """Whether this compressor supports streaming"""
        return False
