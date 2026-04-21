"""
Tests for Delta Encoding.
"""

from __future__ import annotations

import os
import pytest

from airone.collection.delta import (
    DeltaCollectionEncoder,
    DeltaEncoder,
)
from airone.exceptions import DecompressionError


@pytest.fixture
def similar_pair():
    """Two similar byte strings (90% common)."""
    base    = b"AirOne delta test content " * 200
    variant = base[:4000] + b"CHANGED_SECTION " * 50 + base[4800:]
    return base, variant


@pytest.fixture
def identical_pair():
    """Two identical byte strings."""
    data = b"identical content " * 300
    return data, data


@pytest.fixture
def random_pair():
    """Two unrelated random byte strings."""
    ref = os.urandom(4096)
    tgt = os.urandom(4096)
    return ref, tgt


class TestDeltaEncoder:

    def test_encode_decode_similar(self, similar_pair):
        ref, tgt = similar_pair
        encoder = DeltaEncoder()

        delta, meta = encoder.encode(ref, tgt)
        restored = encoder.decode(delta, ref, meta)

        assert restored == tgt

    def test_encode_decode_identical(self, identical_pair):
        ref, tgt = identical_pair
        encoder = DeltaEncoder()

        delta, meta = encoder.encode(ref, tgt)
        restored = encoder.decode(delta, ref, meta)

        assert restored == tgt

    def test_encode_decode_random(self, random_pair):
        ref, tgt = random_pair
        encoder = DeltaEncoder()

        delta, meta = encoder.encode(ref, tgt)
        restored = encoder.decode(delta, ref, meta)

        assert restored == tgt

    def test_similar_delta_is_smaller(self, similar_pair):
        """Similar data should produce smaller delta than target itself."""
        ref, tgt = similar_pair
        encoder = DeltaEncoder()

        delta, meta = encoder.encode(ref, tgt)

        assert meta.compression_ratio > 1.0
        assert len(delta) < len(tgt)

    def test_wrong_reference_raises(self, similar_pair):
        ref, tgt = similar_pair
        encoder = DeltaEncoder()

        delta, meta = encoder.encode(ref, tgt)

        # Corrupt the reference
        bad_ref = b"completely wrong reference data " * 100
        with pytest.raises(DecompressionError, match="hash mismatch"):
            encoder.decode(delta, bad_ref, meta)

    def test_metadata_contains_hashes(self, similar_pair):
        ref, tgt = similar_pair
        encoder = DeltaEncoder()

        _, meta = encoder.encode(ref, tgt)

        assert len(meta.reference_hash) == 64   # SHA-256
        assert len(meta.target_hash)    == 64
        assert meta.target_size  == len(tgt)
        assert meta.reference_size == len(ref)

    def test_method_is_valid(self, similar_pair):
        ref, tgt = similar_pair
        encoder = DeltaEncoder()
        _, meta = encoder.encode(ref, tgt)

        assert meta.delta_method in ("zstd_dict", "xor_rle")


class TestDeltaCollectionEncoder:

    @pytest.fixture
    def file_collection(self):
        """A collection of 4 related files."""
        base = b"shared document template " * 500
        return {
            "report_jan.bin": base + b"January data " * 100,
            "report_feb.bin": base + b"February data " * 100,
            "report_mar.bin": base + b"March data " * 100,
            "report_apr.bin": base + b"April data " * 100,
        }

    def test_encode_decode_collection(self, file_collection):
        enc = DeltaCollectionEncoder()
        bundle = enc.encode_collection(file_collection)
        restored = enc.decode_all(bundle)

        for name, original in file_collection.items():
            assert restored[name] == original, (
                f"File '{name}' did not reconstruct correctly"
            )

    def test_reference_is_largest(self, file_collection):
        enc = DeltaCollectionEncoder()
        bundle = enc.encode_collection(file_collection)

        largest = max(file_collection, key=lambda k: len(file_collection[k]))
        assert bundle["reference_name"] == largest

    def test_deltas_exist_for_non_reference(self, file_collection):
        enc = DeltaCollectionEncoder()
        bundle = enc.encode_collection(file_collection)

        ref = bundle["reference_name"]
        for name in file_collection:
            if name != ref:
                assert name in bundle["deltas"]

    def test_decode_single_file(self, file_collection):
        enc = DeltaCollectionEncoder()
        bundle = enc.encode_collection(file_collection)

        for name, original in file_collection.items():
            restored = enc.decode_file(bundle, name)
            assert restored == original

    def test_empty_collection_raises(self):
        enc = DeltaCollectionEncoder()
        with pytest.raises(Exception):
            enc.encode_collection({})
