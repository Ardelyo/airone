"""
AirOne Collection Optimizer - Phase 3 Foundation

Provides the groundwork for cross-file compression optimization.

Phase 3 implements:
    - Content-addressable block deduplication
    - File similarity analysis (Jaccard on block sets)
    - Base profile extraction (common metadata across files)

Phase 4 will add:
    - Graph-based compression ordering
    - Delta encoding between similar files
    - Parallel processing pipeline
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BlockRecord:
    """One deduplicated block in the content-addressable store."""
    hash:       str
    data:       bytes
    size:       int
    ref_count:  int = 1      # How many files reference this block


@dataclass
class FileRecipe:
    """
    Describes how to reconstruct a file from its blocks.
    Analogous to a torrent file's piece list.
    """
    file_path:   str
    file_size:   int
    block_hashes: list[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """Statistics from a collection compression run."""
    file_count:          int
    total_original_size: int
    total_unique_size:   int
    block_count:         int
    duplicate_blocks:    int
    dedup_ratio:         float
    processing_time:     float

    @property
    def space_saved_bytes(self) -> int:
        return self.total_original_size - self.total_unique_size

    @property
    def space_saved_pct(self) -> float:
        if self.total_original_size == 0:
            return 0.0
        return 100.0 * self.space_saved_bytes / self.total_original_size

    def summary(self) -> str:
        return (
            f"Files        : {self.file_count}\n"
            f"Original     : {self.total_original_size / 1024 / 1024:.2f} MB\n"
            f"Unique data  : {self.total_unique_size / 1024 / 1024:.2f} MB\n"
            f"Saved        : {self.space_saved_bytes / 1024 / 1024:.2f} MB "
            f"({self.space_saved_pct:.1f}%)\n"
            f"Dedup ratio  : {self.dedup_ratio:.2f}x\n"
            f"Time         : {self.processing_time:.2f}s"
        )


@dataclass
class CollectionCompressed:
    """Output of a collection compression run."""
    block_store:   dict[str, BlockRecord]
    file_recipes:  list[FileRecipe]
    stats:         CollectionStats


# ---------------------------------------------------------------------------
# Content-Addressable Storage
# ---------------------------------------------------------------------------

class ContentAddressableStorage:
    """
    Splits files into fixed-size blocks, hashes each block,
    and stores only unique blocks.

    This is the foundation of collection-level deduplication.

    Block size trade-off:
        Small blocks  → better dedup, more overhead per block
        Large blocks  → less overhead, coarser dedup
        64 KB default balances both concerns for general files.
    """

    def __init__(self, block_size: int = 64 * 1024) -> None:
        self.block_size  = block_size
        self._store:     dict[str, BlockRecord] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_file(self, file_path: str) -> FileRecipe:
        """
        Read *file_path*, chunk it, store unique blocks,
        and return a recipe to reconstruct the file.
        """
        file_size = os.path.getsize(file_path)
        recipe = FileRecipe(file_path=file_path, file_size=file_size)

        for block in self._read_blocks(file_path):
            block_hash = self._hash_block(block)

            if block_hash in self._store:
                self._store[block_hash].ref_count += 1
            else:
                self._store[block_hash] = BlockRecord(
                    hash=block_hash,
                    data=block,
                    size=len(block),
                )

            recipe.block_hashes.append(block_hash)

        return recipe

    def add_collection(self, file_paths: list[str]) -> list[FileRecipe]:
        """Add multiple files. Returns one recipe per file."""
        return [self.add_file(fp) for fp in file_paths]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def reconstruct_file(self, recipe: FileRecipe) -> bytes:
        """Reconstruct file bytes from its recipe."""
        chunks = []
        for block_hash in recipe.block_hashes:
            record = self._store.get(block_hash)
            if record is None:
                raise KeyError(
                    f"Block {block_hash} not found in store. "
                    f"Store may be incomplete."
                )
            chunks.append(record.data)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def unique_block_count(self) -> int:
        return len(self._store)

    @property
    def total_unique_bytes(self) -> int:
        return sum(r.size for r in self._store.values())

    @property
    def duplicate_block_count(self) -> int:
        return sum(r.ref_count - 1 for r in self._store.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_blocks(self, file_path: str) -> Iterator[bytes]:
        with open(file_path, "rb") as fh:
            while True:
                block = fh.read(self.block_size)
                if not block:
                    break
                yield block

    @staticmethod
    def _hash_block(block: bytes) -> str:
        return hashlib.sha256(block).hexdigest()


# ---------------------------------------------------------------------------
# Similarity Analyser
# ---------------------------------------------------------------------------

class SimilarityAnalyser:
    """
    Measures pairwise similarity between files based on their block sets.
    Uses Jaccard similarity: |A ∩ B| / |A ∪ B|

    A score of 1.0 means identical files.
    A score of 0.0 means no blocks in common.
    """

    def similarity(
        self,
        recipe_a: FileRecipe,
        recipe_b: FileRecipe,
    ) -> float:
        set_a = set(recipe_a.block_hashes)
        set_b = set(recipe_b.block_hashes)

        intersection = len(set_a & set_b)
        union        = len(set_a | set_b)

        return intersection / union if union else 0.0

    def similarity_matrix(
        self, recipes: list[FileRecipe]
    ) -> list[list[float]]:
        """
        O(n²) pairwise similarity.
        Acceptable for collections up to ~1,000 files.
        Phase 4 will use LSH for larger collections.
        """
        n = len(recipes)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                score = self.similarity(recipes[i], recipes[j])
                matrix[i][j] = score
                matrix[j][i] = score
        return matrix


# ---------------------------------------------------------------------------
# Collection Optimizer
# ---------------------------------------------------------------------------

class CollectionOptimizer:
    """
    High-level API for compressing groups of related files.

    Usage::

        optimizer = CollectionOptimizer()
        result = optimizer.optimize_collection(
            ["scan1.dcm", "scan2.dcm", "scan3.dcm"]
        )
        print(result.stats.summary())
    """

    def __init__(self, block_size: int = 64 * 1024) -> None:
        self._cas        = ContentAddressableStorage(block_size)
        self._similarity = SimilarityAnalyser()

    def optimize_collection(
        self, file_paths: list[str]
    ) -> CollectionCompressed:
        """
        Ingest all files, deduplicate at block level,
        compute stats, and return the compressed collection.
        """
        start = time.perf_counter()

        total_original = sum(os.path.getsize(fp) for fp in file_paths)

        recipes = self._cas.add_collection(file_paths)

        elapsed = time.perf_counter() - start

        dedup_ratio = (
            total_original / max(self._cas.total_unique_bytes, 1)
        )

        stats = CollectionStats(
            file_count=len(file_paths),
            total_original_size=total_original,
            total_unique_size=self._cas.total_unique_bytes,
            block_count=self._cas.unique_block_count,
            duplicate_blocks=self._cas.duplicate_block_count,
            dedup_ratio=dedup_ratio,
            processing_time=elapsed,
        )

        return CollectionCompressed(
            block_store=self._cas._store,
            file_recipes=recipes,
            stats=stats,
        )

    def analyse_similarity(
        self, file_paths: list[str]
    ) -> dict:
        """
        Report pairwise similarity without compressing.
        Useful for understanding collection structure.
        """
        recipes = []
        for fp in file_paths:
            cas = ContentAddressableStorage(self._cas.block_size)
            recipe = cas.add_file(fp)
            recipes.append(recipe)

        matrix = self._similarity.similarity_matrix(recipes)
        names  = [os.path.basename(fp) for fp in file_paths]

        return {
            "files":  names,
            "matrix": matrix,
        }

    def reconstruct_file(
        self,
        collection: CollectionCompressed,
        file_index: int,
    ) -> bytes:
        """Reconstruct a file by index from a compressed collection."""
        recipe = collection.file_recipes[file_index]
        chunks = []
        for block_hash in recipe.block_hashes:
            record = collection.block_store.get(block_hash)
            if record is None:
                raise KeyError(f"Missing block: {block_hash}")
            chunks.append(record.data)
        return b"".join(chunks)
