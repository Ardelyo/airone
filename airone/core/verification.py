import hashlib

def verify_lossless(original_data: bytes, decompressed_data: bytes) -> bool:
    """Verifies that the decompressed data exactly matches the original data."""
    if len(original_data) != len(decompressed_data):
        return False
        
    # We can use hashlib.sha256 or simply compare bytes
    # Byte comparison is fast enough in memory for moderate sized files
    return original_data == decompressed_data

def get_hash(data: bytes) -> str:
    """Return SHA256 string for data"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()
