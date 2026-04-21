#!/usr/bin/env python3
"""
AirOne Example: Streaming Large Files
======================================
Shows how to compress and decompress large files
that don't fit in memory, using windowed streaming.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from airone.core.streaming import StreamingCompressor


def progress(done: int, total: int, ratio: float) -> None:
    pct = done / total * 100
    bar = "#" * int(pct / 5)
    print(f"\r  [{bar:<20}] {pct:5.1f}%  ratio={ratio:.2f}x", end="", flush=True)


def main():
    FILE_MB = 20   # Generate a 20MB test file

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Generate a mixed compressibility test file
        print(f"Generating {FILE_MB}MB test file...")
        data  = b"AirOne streaming example content. " * 200_000   # ~6MB compressible
        data += os.urandom(FILE_MB * 1024 * 1024 - len(data))     # fill to 20MB
        src   = tmp / "large_file.bin"
        src.write_bytes(data)
        print(f"  Written: {src.stat().st_size / 1024 / 1024:.1f} MB\n")

        compressed = tmp / "large_file.air"
        restored   = tmp / "large_file_restored.bin"

        # Compress in 2MB windows
        compressor = StreamingCompressor(window_size=2 * 1024 * 1024)

        print("Compressing (2MB windows)...")
        manifest = compressor.compress_file(str(src), str(compressed), progress_cb=progress)
        print()

        comp_size = compressed.stat().st_size
        print(f"  Compressed size : {comp_size / 1024 / 1024:.2f} MB")
        print(f"  Windows         : {manifest.window_count}")
        print(f"  Ratio           : {len(data) / comp_size:.2f}x\n")

        # Decompress with progress
        print("Decompressing...")
        compressor.decompress_file(str(compressed), str(restored), progress_cb=progress)
        print()

        # Verify
        assert restored.read_bytes() == data, "LOSSLESS CHECK FAILED!"
        print("\n  Lossless verification: PASSED")

        # Random access demo
        print("\nRandom window access (window #2):")
        w2 = compressor.decompress_window(str(compressed), window_index=2)
        print(f"  Window 2 decompressed: {len(w2):,} bytes")


if __name__ == "__main__":
    main()
