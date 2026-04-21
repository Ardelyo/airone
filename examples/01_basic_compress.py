#!/usr/bin/env python3
"""
AirOne Example: Basic Compress & Decompress
===========================================
Shows the simplest possible usage of the AirOne API.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from airone.orchestrator.orchestrator import CompressionOrchestrator


def main():
    # Create a sample file
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "sample.txt"
        output_air = tmp / "sample.txt.air"
        restored   = tmp / "sample_restored.txt"

        # Write compressible content
        input_file.write_bytes(
            b"AirOne example: the quick brown fox jumps over the lazy dog. " * 2000
        )

        original_size = input_file.stat().st_size
        print(f"Original    : {original_size:,} bytes")

        # Compress
        orch   = CompressionOrchestrator()
        result = orch.compress_file(str(input_file), str(output_air))

        print(f"Compressed  : {result.compressed_size:,} bytes")
        print(f"Ratio       : {result.ratio:.2f}x")
        print(f"Strategy    : {result.strategy_name}")
        print(f"Time        : {result.execution_time*1000:.1f}ms")

        # Decompress
        orch.decompress_file(str(output_air), str(restored))

        # Verify
        assert restored.read_bytes() == input_file.read_bytes(), "Lossless check FAILED!"
        print(f"\nLossless verified: OK")


if __name__ == "__main__":
    main()
