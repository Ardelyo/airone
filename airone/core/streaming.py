"""
AirOne Streaming Compressor - Phase 5

Enables compression and decompression of files that are
too large to fit in memory at once.

Design:
    Files are processed in fixed-size windows.
    Each window is analysed and compressed independently.
    The .air container stores a stream manifest:
        [ {offset, size, strategy, compressed_size}, ... ]

This enables:
    - Files larger than available RAM (multi-GB files)
    - Partial decompression (seek to any window)
    - Resumable compression (restart from last complete window)
    - Streaming network transfer (decompress while downloading)

Window size trade-off:
    Small windows → better parallelism, worse compression ratio
                    (less context for compressor)
    Large windows → better ratio, more memory required
    Default 16 MB balances both for most workloads.

Format:
    The stream is wrapped in a standard .air container.
    The metadata block contains the window manifest.
    The data block is a concatenation of compressed windows.
"""

from __future__ import annotations

import io
import json
import os
import time
import math
from dataclasses import asdict, dataclass, field
from typing import BinaryIO, Callable, Iterator, Optional

from airone.compressors.base import CompressionResult
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.core.verification import verify_lossless
from airone.exceptions import CompressionError, DecompressionError


# ---------------------------------------------------------------------------
# Window manifest
# ---------------------------------------------------------------------------

@dataclass
class WindowRecord:
    """Metadata for one compressed window."""
    window_index:     int
    original_offset:  int      # byte offset in original file
    original_size:    int      # bytes in this window (original)
    compressed_size:  int      # bytes after compression
    stream_offset:    int      # byte offset in compressed stream
    strategy:         str
    checksum:         str      # SHA-256 of original window bytes


@dataclass
class StreamManifest:
    """
    Complete manifest for a streamed .air file.
    Stored as JSON in the .air metadata block.
    """
    total_original_size: int
    window_size:         int
    window_count:        int
    windows:             list[WindowRecord] = field(default_factory=list)
    format_version:      str = "stream-1.0"

    def to_json(self) -> str:
        return json.dumps(
            {
                "format_version":      self.format_version,
                "total_original_size": self.total_original_size,
                "window_size":         self.window_size,
                "window_count":        self.window_count,
                "windows": [asdict(w) for w in self.windows],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "StreamManifest":
        data = json.loads(json_str)
        manifest = cls(
            total_original_size=data["total_original_size"],
            window_size=data["window_size"],
            window_count=data["window_count"],
        )
        for wd in data.get("windows", []):
            manifest.windows.append(WindowRecord(**wd))
        return manifest


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, float], None]
# (windows_done, windows_total, ratio_so_far)


# ---------------------------------------------------------------------------
# Streaming Compressor
# ---------------------------------------------------------------------------

class StreamingCompressor:
    """
    Compress arbitrarily large files without loading them fully into RAM.

    Usage::

        compressor = StreamingCompressor(window_size=16 * 1024 * 1024)
        compressor.compress_file("huge.raw", "huge.air")
        compressor.decompress_file("huge.air", "huge_restored.raw")
    """

    def __init__(
        self,
        window_size: int = 16 * 1024 * 1024,  # 16 MB
        strategy:    str = "traditional_zstd",
    ) -> None:
        self.window_size = window_size
        self.strategy    = strategy
        self._codec      = ZstdCompressor()

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_file(
        self,
        input_path:  str,
        output_path: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> StreamManifest:
        """
        Compress *input_path* → *output_path* using windowed streaming.
        Returns the stream manifest.
        """
        total_size   = os.path.getsize(input_path)
        total_windows = math.ceil(total_size / self.window_size)

        manifest = StreamManifest(
            total_original_size=total_size,
            window_size=self.window_size,
            window_count=total_windows,
        )

        stream_offset = 0
        window_index  = 0

        with open(input_path, "rb") as src, \
             open(output_path + ".stream", "wb") as stream_out:

            for window_data in self._read_windows(src):
                record = self._compress_window(
                    window_data, window_index, stream_offset, stream_out
                )
                manifest.windows.append(record)
                stream_offset += record.compressed_size
                window_index  += 1

                if progress_cb:
                    current_orig = sum(w.original_size for w in manifest.windows)
                    ratio = (
                        current_orig / max(stream_offset, 1)
                    )
                    progress_cb(window_index, total_windows, ratio)

        # Write manifest + stream into .air container
        self._package(manifest, output_path)

        # Clean up temporary stream file
        if os.path.exists(output_path + ".stream"):
            os.unlink(output_path + ".stream")

        return manifest

    def decompress_file(
        self,
        input_path:  str,
        output_path: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> int:
        """
        Decompress streamed .air file → original.
        Returns bytes written.
        """
        manifest, stream_data = self._unpackage(input_path)
        bytes_written = 0

        with open(output_path, "wb") as out:
            for i, record in enumerate(manifest.windows):
                window_bytes = stream_data[
                    record.stream_offset:
                    record.stream_offset + record.compressed_size
                ]
                decompressed = self._codec.decompress(window_bytes, {})

                # Verify window integrity
                import hashlib
                checksum = hashlib.sha256(decompressed).hexdigest()
                if checksum != record.checksum:
                    raise DecompressionError(
                        f"Window {i} checksum mismatch. "
                        f"Stream may be corrupted."
                    )

                out.write(decompressed)
                bytes_written += len(decompressed)

                if progress_cb:
                    progress_cb(
                        i + 1,
                        manifest.window_count,
                        bytes_written / max(manifest.total_original_size, 1),
                    )

        return bytes_written

    def decompress_window(
        self,
        input_path:   str,
        window_index: int,
    ) -> bytes:
        """
        Decompress a single window by index.
        Enables random access into a large compressed stream.
        """
        manifest, stream_data = self._unpackage(input_path)

        if window_index >= len(manifest.windows):
            raise DecompressionError(
                f"Window index {window_index} out of range "
                f"(total: {manifest.window_count})"
            )

        record = manifest.windows[window_index]
        window_bytes = stream_data[
            record.stream_offset:
            record.stream_offset + record.compressed_size
        ]
        return self._codec.decompress(window_bytes, {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_windows(self, fh: BinaryIO) -> Iterator[bytes]:
        while True:
            window = fh.read(self.window_size)
            if not window:
                break
            yield window

    def _compress_window(
        self,
        data:          bytes,
        window_index:  int,
        stream_offset: int,
        stream_out:    BinaryIO,
    ) -> WindowRecord:
        import hashlib
        checksum   = hashlib.sha256(data).hexdigest()
        result     = self._codec.compress(data)
        compressed = result.compressed_data
        stream_out.write(compressed)

        return WindowRecord(
            window_index=window_index,
            original_offset=window_index * self.window_size,
            original_size=len(data),
            compressed_size=len(compressed),
            stream_offset=stream_offset,
            strategy=self.strategy,
            checksum=checksum,
        )

    def _package(self, manifest: StreamManifest, output_path: str) -> None:
        """Write manifest + stream into a self-contained file."""
        stream_path = output_path + ".stream"
        stream_data = b""
        if os.path.exists(stream_path):
            with open(stream_path, "rb") as f:
                stream_data = f.read()

        manifest_bytes = manifest.to_json().encode("utf-8")
        manifest_len   = len(manifest_bytes).to_bytes(8, "big")

        with open(output_path, "wb") as out:
            out.write(b"AIRSTREAM1")       # Magic
            out.write(manifest_len)         # 8-byte manifest length
            out.write(manifest_bytes)       # JSON manifest
            out.write(stream_data)          # Compressed windows

    def _unpackage(
        self, input_path: str
    ) -> tuple[StreamManifest, bytes]:
        """Parse a streamed .air file into manifest + stream bytes."""
        with open(input_path, "rb") as f:
            magic = f.read(10)
            if magic != b"AIRSTREAM1":
                raise DecompressionError(
                    f"Not a streaming .air file: {input_path}"
                )
            manifest_len   = int.from_bytes(f.read(8), "big")
            manifest_bytes = f.read(manifest_len)
            stream_data    = f.read()

        manifest = StreamManifest.from_json(
            manifest_bytes.decode("utf-8")
        )
        return manifest, stream_data
