"""Tests for LSH Similarity Engine."""

from __future__ import annotations

import os
import pytest
import numpy as np

from airone.collection.lsh import (
    LSHIndex,
    MinHasher,
    ScalableCollectionAnalyser,
    SimilarPair,
)


@pytest.fixture
def identical_content():
    return b"AirOne identical content block " * 500


@pytest.fixture
def similar_content():
    base = b"AirOne similar content block " * 500
    variant = base[:8000] + b"CHANGED" * 100 + base[8700:]
    return base, variant


@pytest.fixture
def unrelated_content():
    return os.urandom(4096), os.urandom(4096)


class TestMinHasher:

    def test_identical_data_same_signature(self, identical_content):
        hasher = MinHasher(num_permutations=64)
        sig_a = hasher.signature(identical_content)
        sig_b = hasher.signature(identical_content)
        assert np.array_equal(sig_a, sig_b)

    def test_identical_jaccard_is_one(self, identical_content):
        hasher = MinHasher(num_permutations=64)
        sig = hasher.signature(identical_content)
        score = hasher.jaccard_estimate(sig, sig)
        assert score == pytest.approx(1.0)

    def test_similar_jaccard_is_high(self, similar_content):
        base, variant = similar_content
        hasher = MinHasher(num_permutations=128)
        sig_a  = hasher.signature(base)
        sig_b  = hasher.signature(variant)
        score  = hasher.jaccard_estimate(sig_a, sig_b)
        assert score > 0.4   # Similar content

    def test_unrelated_jaccard_is_low(self, unrelated_content):
        a, b   = unrelated_content
        hasher = MinHasher(num_permutations=128)
        sig_a  = hasher.signature(a)
        sig_b  = hasher.signature(b)
        score  = hasher.jaccard_estimate(sig_a, sig_b)
        assert score < 0.3

    def test_signature_shape(self, identical_content):
        hasher = MinHasher(num_permutations=64)
        sig    = hasher.signature(identical_content)
        assert sig.shape == (64,)
        assert sig.dtype == np.uint64

    def test_empty_data(self):
        hasher = MinHasher(num_permutations=32)
        sig = hasher.signature(b"")
        assert sig.shape == (32,)


class TestLSHIndex:

    def test_add_and_find_identical(self, identical_content):
        index = LSHIndex(num_permutations=64, num_bands=16)
        index.add("file_a", identical_content)
        index.add("file_b", identical_content)

        pairs = index.find_similar_pairs(threshold=0.9)
        assert len(pairs) > 0

    def test_find_similar_pair(self, similar_content):
        base, variant = similar_content
        index = LSHIndex(num_permutations=128, num_bands=32)
        index.add("base", base)
        index.add("variant", variant)

        pairs = index.find_similar_pairs(threshold=0.3)
        assert len(pairs) > 0
        names = {p.file_a for p in pairs} | {p.file_b for p in pairs}
        assert "base" in names or "variant" in names

    def test_unrelated_not_paired(self, unrelated_content):
        a, b = unrelated_content
        index = LSHIndex(num_permutations=128, num_bands=32)
        index.add("file_a", a)
        index.add("file_b", b)

        pairs = index.find_similar_pairs(threshold=0.8)
        assert len(pairs) == 0

    def test_indexed_count(self, identical_content):
        index = LSHIndex(num_permutations=64, num_bands=16)
        index.add("a", identical_content)
        index.add("b", identical_content)
        assert index.indexed_count == 2

    def test_permutations_divisibility_check(self):
        with pytest.raises(ValueError, match="divisible"):
            LSHIndex(num_permutations=100, num_bands=32)

    def test_query_similar(self, similar_content):
        base, variant = similar_content
        index = LSHIndex(num_permutations=128, num_bands=32)
        index.add("base", base)

        matches = index.query_similar(variant, threshold=0.3)
        assert isinstance(matches, list)

    def test_expected_threshold_is_float(self):
        index = LSHIndex(num_permutations=128, num_bands=32)
        t = index.expected_threshold()
        assert 0.0 < t < 1.0


class TestScalableCollectionAnalyser:

    @pytest.fixture
    def collection_files(self, tmp_path):
        base = b"shared content across all files " * 400
        paths = []
        for i in range(5):
            p = tmp_path / f"doc_{i}.bin"
            p.write_bytes(base + f"unique section {i} ".encode() * 50)
            paths.append(str(p))
        # One unrelated file
        unrelated = tmp_path / "unrelated.bin"
        unrelated.write_bytes(os.urandom(8192))
        paths.append(str(unrelated))
        return paths

    def test_ingest_files(self, collection_files):
        analyser = ScalableCollectionAnalyser()
        analyser.ingest_files(collection_files)
        assert analyser._index.indexed_count == len(collection_files)

    def test_find_similar_pairs(self, collection_files):
        analyser = ScalableCollectionAnalyser(
            num_permutations=64, num_bands=16
        )
        analyser.ingest_files(collection_files)
        pairs = analyser.find_similar_pairs(threshold=0.3)
        assert len(pairs) > 0

    def test_find_clusters(self, collection_files):
        analyser = ScalableCollectionAnalyser(
            num_permutations=64, num_bands=16
        )
        analyser.ingest_files(collection_files)
        clusters = analyser.find_clusters(threshold=0.3)
        assert len(clusters) >= 1
        # The 5 similar files should be in one cluster
        largest = max(clusters, key=lambda c: c.size)
        assert largest.size >= 3

    def test_is_duplicate_positive(self, tmp_path):
        content = b"duplicate detection test " * 300
        analyser = ScalableCollectionAnalyser(
            num_permutations=64, num_bands=16
        )
        analyser.ingest_bytes({"existing_file": content})
        assert analyser.is_duplicate(content, threshold=0.9)

    def test_is_duplicate_negative(self):
        analyser = ScalableCollectionAnalyser(
            num_permutations=64, num_bands=16
        )
        analyser.ingest_bytes({"file_a": b"content A " * 300})
        assert not analyser.is_duplicate(
            os.urandom(3000), threshold=0.8
        )
