class AirOneError(Exception):
    """Base exception for all AirOne errors"""
    pass

class CompressionError(AirOneError):
    """Raised when compression fails"""
    pass

class DecompressionError(AirOneError):
    """Raised when decompression fails"""
    pass

class VerificationError(AirOneError):
    """Raised when lossless verification fails"""
    pass

class FormatError(AirOneError):
    """Raised when file format is invalid"""
    pass

class StrategyError(AirOneError):
    """Raised when no suitable strategy found"""
    pass

class ReferenceNotFoundError(AirOneError):
    """Raised when reference DB lookup fails"""
    pass
