"""
Tests for CollectionOptimizer and ContentAddressableStorage.
"""

from __future__ import annotations

import os
import pytest

from airone.collection.optimizer import (
    CollectionOptimizer,
    ContentAddressableStorage,
    SimilarityAnalyser,
)


@pytest.fixture
def identical_files(tmp_path) -> list[str]:
    """Three identical files."""
    content = b"AirOne test content " * 500
    paths = []
    for i in range(3):
        p = tmp_path / f"file_{i}.bin"
        p.write_bytes(content)
        paths.append(str(p))
    return paths


@pytest.fixture
def unique_files(tmp_path) -> list[str]:
    """Three completely different files."""
    paths = []
    for i in range(3):
        p = tmp_path / f"unique_{i}.bin"
        p.write_bytes(f"unique content {i} ".encode() * 200)
        paths.append(str(p))
    return paths


@pytest.fixture
def mixed_files(tmp_path) -> list[str]:
    """Two similar files + one different file."""
    shared = b"shared block content " * 300
    extra_a = b"unique to A " * 100
    extra_b = b"unique to B " * 100
    different = b"completely different " * 400

    paths = []
    for name, content in [
        ("a.bin", shared + extra_a),
        ("b.bin", shared + extra_b),
        ("c.bin", different),
    ]:
        p = tmp_path / name
        p.write_bytes(content)
        paths.append(str(p))
    return paths


class TestContentAddressableStorage:

    def test_add_single_file(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"Hello AirOne " * 100)
        cas = ContentAddressableStorage(block_size=256)
        recipe = cas.add_file(str(path))

        assert len(recipe.block_hashes) > 0
        assert recipe.file_size == os.path.getsize(str(path))

    def test_identical_files_share_blocks(self, identical_files):
        cas = ContentAddressableStorage(block_size=1024)
        recipes = cas.add_collection(identical_files)

        # All three files reference the same blocks
        assert cas.duplicate_block_count > 0
        # Unique blocks << total referenced blocks
        total_refs = sum(len(r.block_hashes) for r in recipes)
        assert cas.unique_block_count < total_refs

    def test_reconstruct_file(self, tmp_path):
        content = b"Reconstruct me " * 500
        path = tmp_path / "source.bin"
        path.write_bytes(content)

        cas = ContentAddressableStorage(block_size=256)
        recipe = cas.add_file(str(path))
        reconstructed = cas.reconstruct_file(recipe)

        assert reconstructed == content

    def test_unique_files_no_dedup(self, unique_files):
        cas = ContentAddressableStorage(block_size=1024)
        cas.add_collection(unique_files)
        # No duplicate blocks expected for completely different content
        assert cas.duplicate_block_count == 0


class TestSimilarityAnalyser:

    def test_identical_files_score_one(self, identical_files):
        analyser = SimilarityAnalyser()
        recipes = []
        for fp in identical_files:
            cas = ContentAddressableStorage(block_size=1024)
            recipe = cas.add_file(fp)
            recipes.append(recipe)

        score = analyser.similarity(recipes[0], recipes[1])
        assert score == pytest.approx(1.0)

    def test_different_files_score_zero(self, unique_files):
        analyser = SimilarityAnalyser()
        recipes = []
        for fp in unique_files:
            cas = ContentAddressableStorage(block_size=1024)
            recipe = cas.add_file(fp)
            recipes.append(recipe)

        score = analyser.similarity(recipes[0], recipes[1])
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self, mixed_files):
        analyser = SimilarityAnalyser()
        recipes = []
        for fp in mixed_files:
            cas = ContentAddressableStorage(block_size=512)
            recipe = cas.add_file(fp)
            recipes.append(recipe)

        # A and B share content → should be > 0
        score_ab = analyser.similarity(recipes[0], recipes[1])
        # A and C share nothing → should be 0
        score_ac = analyser.similarity(recipes[0], recipes[2])

        assert score_ab > 0
        assert score_ac == pytest.approx(0.0, abs=0.05)

    def test_similarity_matrix_shape(self, mixed_files):
        analyser = SimilarityAnalyser()
        recipes = []
        for fp in mixed_files:
            cas = ContentAddressableStorage(block_size=512)
            recipe = cas.add_file(fp)
            recipes.append(recipe)

        matrix = analyser.similarity_matrix(recipes)

        n = len(mixed_files)
        assert len(matrix) == n
        assert all(len(row) == n for row in matrix)
        # Diagonal must be 1.0
        for i in range(n):
            assert matrix[i][i] == pytest.approx(1.0)
        # Matrix must be symmetric
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == pytest.approx(matrix[j][i])


class TestCollectionOptimizer:

    def test_optimize_identical_files(self, identical_files):
        optimizer = CollectionOptimizer(block_size=1024)
        result = optimizer.optimize_collection(identical_files)

        assert result.stats.file_count == 3
        assert result.stats.dedup_ratio > 1.5   # Should save significant space
        assert result.stats.duplicate_blocks > 0

    def test_optimize_returns_recipes(self, mixed_files):
        optimizer = CollectionOptimizer(block_size=512)
        result = optimizer.optimize_collection(mixed_files)

        assert len(result.file_recipes) == len(mixed_files)

    def test_reconstruct_after_optimize(self, mixed_files):
        optimizer = CollectionOptimizer(block_size=512)
        result = optimizer.optimize_collection(mixed_files)

        for i, fp in enumerate(mixed_files):
            original = open(fp, "rb").read()
            reconstructed = optimizer.reconstruct_file(result, i)
            assert reconstructed == original, f"Reconstruction failed for file {i}"

    def test_stats_summary_is_string(self, identical_files):
        optimizer = CollectionOptimizer()
        result = optimizer.optimize_collection(identical_files)
        summary = result.stats.summary()
        assert isinstance(summary, str)
        assert "MB" in summary
