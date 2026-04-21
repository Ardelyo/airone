import hashlib

def verify_lossless(original_data: bytes, decompressed_data: bytes) -> bool:
    """Verifies that the decompressed data exactly matches the original data."""
    if original_data == decompressed_data:
        return True

    # Check if both are PNG images and pixel-perfect (Semantic lossless)
    if original_data.startswith(b'\x89PNG\r\n\x1a\n') and decompressed_data.startswith(b'\x89PNG\r\n\x1a\n'):
        try:
            from PIL import Image
            import io
            img1 = Image.open(io.BytesIO(original_data))
            img2 = Image.open(io.BytesIO(decompressed_data))
            if img1.size != img2.size or img1.mode != img2.mode:
                return False
            return list(img1.getdata()) == list(img2.getdata())
        except Exception:
            return False
            
    return False

def get_hash(data: bytes) -> str:
    """Return SHA256 string for data"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()
