"""
AirOne Delta Encoder - Phase 4

Compresses files as differences (deltas) from a reference file.
Uses ZSTD's built-in --patch-from capability via the zstandard
Python binding (ZstdCompressionDict with training data approach),
falling back to a pure-Python XOR delta for guaranteed compatibility.

When to use delta encoding:
    - File versions (v1, v2, v3 of same document)
    - Repeated exports of same report with different data
    - Collections of similar images (same camera/scene)

Workflow:
    compress:   delta = encode(target, reference)
    decompress: target = decode(delta, reference)

The reference file is NOT modified. It must be available
at decompression time (stored separately in the collection).
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time
from dataclasses import dataclass

import zstandard as zstd

from airone.compressors.base import CompressionResult
from airone.exceptions import CompressionError, DecompressionError


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeltaMetadata:
    """
    Everything needed to reconstruct the target from a delta.
    Stored alongside the compressed delta bytes.
    """
    reference_hash:  str    # SHA-256 of the reference file
    target_hash:     str    # SHA-256 of the target (for verification)
    target_size:     int    # bytes
    reference_size:  int    # bytes
    delta_method:    str    # "zstd_dict" | "xor_rle"
    compression_ratio: float


# ---------------------------------------------------------------------------
# Core delta encoder
# ---------------------------------------------------------------------------

class DeltaEncoder:
    """
    Encodes a target file as a compact delta from a reference file.

    Two strategies, tried in order:
        1. ZSTD dictionary compression
           Training the ZSTD compressor on the reference makes it
           dramatically more effective on similar data.
           Works best when reference and target are structurally similar.

        2. XOR + RLE fallback
           Pure Python. Always works, guaranteed compatibility.
           Less efficient but completely dependency-free.
    """

    def __init__(self, zstd_level: int = 19) -> None:
        self.zstd_level = zstd_level

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(
        self,
        reference_data: bytes,
        target_data:    bytes,
    ) -> tuple[bytes, DeltaMetadata]:
        """
        Compute delta: target relative to reference.

        Returns (compressed_delta_bytes, metadata).
        """
        reference_hash = self._sha256(reference_data)
        target_hash    = self._sha256(target_data)

        # Strategy 1: ZSTD dictionary
        try:
            delta_bytes, method = self._encode_zstd_dict(
                reference_data, target_data
            )
        except Exception:
            # Strategy 2: XOR + RLE fallback
            delta_bytes, method = self._encode_xor_rle(
                reference_data, target_data
            )

        ratio = len(target_data) / max(len(delta_bytes), 1)

        metadata = DeltaMetadata(
            reference_hash=reference_hash,
            target_hash=target_hash,
            target_size=len(target_data),
            reference_size=len(reference_data),
            delta_method=method,
            compression_ratio=round(ratio, 2),
        )

        return delta_bytes, metadata

    def decode(
        self,
        delta_bytes:    bytes,
        reference_data: bytes,
        metadata:       DeltaMetadata,
    ) -> bytes:
        """
        Reconstruct target from delta + reference.
        Verifies result against stored target_hash.
        """
        # Verify reference integrity
        ref_hash = self._sha256(reference_data)
        if ref_hash != metadata.reference_hash:
            raise DecompressionError(
                "Delta decode: reference file hash mismatch. "
                "Ensure you are using the correct reference."
            )

        if metadata.delta_method == "zstd_dict":
            target = self._decode_zstd_dict(delta_bytes, reference_data)
        elif metadata.delta_method == "xor_rle":
            target = self._decode_xor_rle(delta_bytes, reference_data)
        else:
            raise DecompressionError(
                f"Unknown delta method: {metadata.delta_method}"
            )

        # Verify reconstruction
        if self._sha256(target) != metadata.target_hash:
            raise DecompressionError(
                "Delta decode: reconstructed file hash mismatch. "
                "Compressed data may be corrupted."
            )

        return target

    # ------------------------------------------------------------------
    # ZSTD dictionary strategy
    # ------------------------------------------------------------------

    def _encode_zstd_dict(
        self,
        reference: bytes,
        target:    bytes,
    ) -> tuple[bytes, str]:
        """
        Train ZSTD on the reference data as a dictionary,
        then compress the target using that dictionary.

        The reference acts as a prediction model:
        bytes common to reference and target compress to near-zero.
        """
        # Build dictionary from reference
        # (ZSTD dict training works best with multiple samples,
        #  but a single large reference still provides a large benefit)
        cdict = zstd.ZstdCompressionDict(reference)

        compressor = zstd.ZstdCompressor(
            level=self.zstd_level,
            dict_data=cdict,
        )
        compressed = compressor.compress(target)
        return compressed, "zstd_dict"

    def _decode_zstd_dict(
        self, delta: bytes, reference: bytes
    ) -> bytes:
        cdict = zstd.ZstdCompressionDict(reference)
        dctx  = zstd.ZstdDecompressor(dict_data=cdict)
        return dctx.decompress(delta)

    # ------------------------------------------------------------------
    # XOR + RLE fallback
    # ------------------------------------------------------------------

    def _encode_xor_rle(
        self,
        reference: bytes,
        target:    bytes,
    ) -> tuple[bytes, str]:
        """
        Compute byte-level XOR between reference and target,
        then compress the result with standard ZSTD.

        XOR produces near-zero bytes wherever reference ≈ target,
        making subsequent compression very effective.
        """
        # Pad shorter array to match lengths
        max_len = max(len(reference), len(target))
        ref_padded = reference.ljust(max_len, b"\x00")
        tgt_padded = target.ljust(max_len, b"\x00")

        # XOR
        xor_bytes = bytes(
            a ^ b for a, b in zip(ref_padded, tgt_padded)
        )

        # Encode target length in header (needed for reconstruction)
        header = struct.pack(">I", len(target))

        # ZSTD compress XOR result
        compressor = zstd.ZstdCompressor(level=self.zstd_level)
        compressed_xor = compressor.compress(xor_bytes)

        return header + compressed_xor, "xor_rle"

    def _decode_xor_rle(
        self, delta: bytes, reference: bytes
    ) -> bytes:
        # Read header
        target_len = struct.unpack(">I", delta[:4])[0]
        compressed_xor = delta[4:]

        # Decompress XOR
        dctx = zstd.ZstdDecompressor()
        xor_bytes = dctx.decompress(compressed_xor)

        # Reconstruct target
        max_len = len(xor_bytes)
        ref_padded = reference.ljust(max_len, b"\x00")

        reconstructed = bytes(
            a ^ b for a, b in zip(ref_padded, xor_bytes)
        )

        # Trim to original target length
        return reconstructed[:target_len]

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Collection-level delta coordinator
# ---------------------------------------------------------------------------

class DeltaCollectionEncoder:
    """
    Encodes a collection of similar files using delta encoding.

    Algorithm:
        1. Select the largest file as the base reference
           (largest = most content = best prediction model)
        2. Encode all other files as deltas from the reference
        3. Return: {reference: full_data, deltas: [...]}

    For very large collections, Phase 5 will build a minimum
    spanning tree over the similarity graph and encode along
    tree edges for maximum efficiency.
    """

    def __init__(self) -> None:
        self._encoder = DeltaEncoder()

    def encode_collection(
        self, files: dict[str, bytes]
    ) -> dict:
        """
        files: {filename: file_bytes}

        Returns a bundle::

            {
                "reference_name": str,
                "reference_data": bytes,
                "deltas": {
                    filename: {
                        "delta": bytes,
                        "metadata": DeltaMetadata
                    }
                }
            }
        """
        if not files:
            raise CompressionError(
                "DeltaCollectionEncoder: no files provided."
            )

        # Select reference (largest file)
        reference_name = max(files, key=lambda k: len(files[k]))
        reference_data = files[reference_name]

        deltas: dict[str, dict] = {}

        for name, data in files.items():
            if name == reference_name:
                continue
            delta_bytes, metadata = self._encoder.encode(
                reference_data, data
            )
            deltas[name] = {
                "delta":    delta_bytes,
                "metadata": metadata,
            }

        return {
            "reference_name": reference_name,
            "reference_data": reference_data,
            "deltas":         deltas,
        }

    def decode_file(
        self,
        bundle:   dict,
        filename: str,
    ) -> bytes:
        """Reconstruct a single file from the bundle."""
        if filename == bundle["reference_name"]:
            return bundle["reference_data"]

        if filename not in bundle["deltas"]:
            raise DecompressionError(
                f"File '{filename}' not found in delta bundle."
            )

        delta_entry   = bundle["deltas"][filename]
        delta_bytes   = delta_entry["delta"]
        metadata      = delta_entry["metadata"]
        reference     = bundle["reference_data"]

        return self._encoder.decode(delta_bytes, reference, metadata)

    def decode_all(self, bundle: dict) -> dict[str, bytes]:
        """Reconstruct all files from the bundle."""
        result = {bundle["reference_name"]: bundle["reference_data"]}
        for name in bundle["deltas"]:
            result[name] = self.decode_file(bundle, name)
        return result
