"""Tests for Streaming Compressor."""

from __future__ import annotations

import os
import math
import pytest

from airone.core.streaming import (
    StreamManifest,
    StreamingCompressor,
    WindowRecord,
)
from airone.exceptions import DecompressionError


@pytest.fixture
def small_file(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"AirOne streaming test. " * 1000)
    return str(path)


@pytest.fixture
def multi_window_file(tmp_path):
    """File larger than one window (window = 64KB in tests)."""
    path = tmp_path / "multi.bin"
    path.write_bytes(b"X" * (200 * 1024))  # 200 KB
    return str(path)


@pytest.fixture
def random_file(tmp_path):
    path = tmp_path / "random.bin"
    path.write_bytes(os.urandom(128 * 1024))  # 128 KB random
    return str(path)


class TestStreamManifest:

    def test_to_from_json_roundtrip(self):
        manifest = StreamManifest(
            total_original_size=1024,
            window_size=512,
            window_count=2,
        )
        manifest.windows.append(WindowRecord(
            window_index=0,
            original_offset=0,
            original_size=512,
            compressed_size=400,
            stream_offset=0,
            strategy="traditional_zstd",
            checksum="abc" * 21 + "ab",
        ))
        json_str  = manifest.to_json()
        restored  = StreamManifest.from_json(json_str)

        assert restored.total_original_size == 1024
        assert restored.window_count == 2
        assert len(restored.windows) == 1
        assert restored.windows[0].window_index == 0


class TestStreamingCompressor:

    def test_compress_decompress_small(self, small_file, tmp_path):
        output = str(tmp_path / "small.air")
        restored = str(tmp_path / "small_restored.bin")

        compressor = StreamingCompressor(window_size=64 * 1024)
        compressor.compress_file(small_file, output)
        compressor.decompress_file(output, restored)

        original_data  = open(small_file, "rb").read()
        restored_data  = open(restored, "rb").read()
        assert restored_data == original_data

    def test_compress_decompress_multi_window(
        self, multi_window_file, tmp_path
    ):
        output   = str(tmp_path / "multi.air")
        restored = str(tmp_path / "multi_restored.bin")

        compressor = StreamingCompressor(window_size=64 * 1024)
        manifest   = compressor.compress_file(multi_window_file, output)

        assert manifest.window_count >= 3

        compressor.decompress_file(output, restored)
        assert open(multi_window_file, "rb").read() == \
               open(restored, "rb").read()

    def test_random_file_roundtrip(self, random_file, tmp_path):
        output   = str(tmp_path / "random.air")
        restored = str(tmp_path / "random_restored.bin")

        compressor = StreamingCompressor(window_size=64 * 1024)
        compressor.compress_file(random_file, output)
        compressor.decompress_file(output, restored)

        assert open(random_file, "rb").read() == \
               open(restored, "rb").read()

    def test_manifest_window_count(self, multi_window_file, tmp_path):
        output     = str(tmp_path / "out.air")
        compressor = StreamingCompressor(window_size=64 * 1024)
        manifest   = compressor.compress_file(multi_window_file, output)

        file_size = os.path.getsize(multi_window_file)
        expected  = math.ceil(file_size / (64 * 1024))
        assert manifest.window_count == expected

    def test_progress_callback_called(self, small_file, tmp_path):
        output    = str(tmp_path / "cb.air")
        calls     = []

        def cb(done, total, ratio):
            calls.append((done, total, ratio))

        compressor = StreamingCompressor(window_size=4 * 1024)
        compressor.compress_file(small_file, output, progress_cb=cb)

        assert len(calls) > 0
        assert calls[-1][0] == calls[-1][1]    # done == total at end

    def test_random_window_access(self, multi_window_file, tmp_path):
        output     = str(tmp_path / "access.air")
        compressor = StreamingCompressor(window_size=64 * 1024)
        manifest   = compressor.compress_file(multi_window_file, output)

        # Read window 0 specifically
        window_0 = compressor.decompress_window(output, 0)
        assert len(window_0) > 0

    def test_corrupted_stream_raises(self, tmp_path):
        bad_file = tmp_path / "bad.air"
        bad_file.write_bytes(b"AIRSTREAM1" + b"\x00" * 8 + b"invalid json{{{")

        compressor = StreamingCompressor()
        with pytest.raises(Exception):
            compressor.decompress_file(str(bad_file), str(tmp_path / "out"))

    def test_wrong_magic_raises(self, tmp_path):
        bad_file = tmp_path / "notstream.air"
        bad_file.write_bytes(b"WRONGMAGIC1234567890")

        compressor = StreamingCompressor()
        with pytest.raises(DecompressionError, match="streaming"):
            compressor.decompress_file(
                str(bad_file), str(tmp_path / "out")
            )
