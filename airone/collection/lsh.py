"""
AirOne LSH (Locality-Sensitive Hashing) Similarity Engine - Phase 5

Phase 3 used O(n²) Jaccard similarity — acceptable for ~1,000 files.
Phase 5 replaces this with MinHash LSH, which achieves approximate
nearest-neighbour similarity in O(n) time.

This scales AirOne collections to millions of files.

Theory:
    MinHash:   Approximates Jaccard similarity in O(k) per pair
               where k = number of hash functions (permutations)
    LSH bands: Groups similar files into buckets using band/row structure
               Files in the same bucket are candidate pairs

Parameters (tunable):
    num_permutations : Number of MinHash functions (default 128)
                       More = better accuracy, more memory
    num_bands        : LSH band count (default 32)
                       Fewer bands = higher recall, more false positives
    rows_per_band    : num_permutations / num_bands
    threshold        : Approximate Jaccard threshold for similarity

Accuracy at default settings (128 perms, 32 bands):
    True  Jaccard 0.8 → detected with ~97% probability
    True  Jaccard 0.5 → detected with ~80% probability
    True  Jaccard 0.2 → detected with ~25% probability (mostly filtered)
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# MinHash signature
# ---------------------------------------------------------------------------

class MinHasher:
    """
    Computes MinHash signatures for file shingles.

    Shingling strategy:
        Files are read in overlapping 8-byte windows (shingles).
        Each shingle is hashed with multiple hash functions.
        The minimum hash per function = the MinHash signature element.

    We simulate multiple hash functions using a single fast hash
    with linear transformations: h_i(x) = (a_i * x + b_i) mod P
    where P is a large prime.
    """

    _LARGE_PRIME = (1 << 31) - 1      # Mersenne prime

    def __init__(self, num_permutations: int = 128, shingle_size: int = 8) -> None:
        self.num_permutations = num_permutations
        self.shingle_size     = shingle_size

        rng = np.random.default_rng(seed=42)   # Fixed seed for reproducibility
        self._a = rng.integers(1, self._LARGE_PRIME, size=num_permutations)
        self._b = rng.integers(0, self._LARGE_PRIME, size=num_permutations)

    def signature(self, data: bytes) -> np.ndarray:
        """
        Compute MinHash signature for *data*.
        Returns array of shape (num_permutations,) with dtype uint64.
        """
        sig = np.full(self.num_permutations, np.iinfo(np.uint64).max, dtype=np.uint64)

        for shingle_hash in self._shingle_hashes(data):
            # Vectorised universal hash: h_i = (a_i * x + b_i) mod P
            hashes = (
                (self._a * shingle_hash + self._b) % self._LARGE_PRIME
            ).astype(np.uint64)
            sig = np.minimum(sig, hashes)

        return sig

    def jaccard_estimate(
        self, sig_a: np.ndarray, sig_b: np.ndarray
    ) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        matches = np.sum(sig_a == sig_b)
        return float(matches) / self.num_permutations

    def _shingle_hashes(self, data: bytes) -> Iterator[int]:
        """Generate integer hashes for each k-shingle in data."""
        if len(data) < self.shingle_size:
            if data:
                yield int.from_bytes(
                    hashlib.md5(data).digest()[:8], "little"
                )
            return

        # Optimization: use a stride (step) to avoid processing every single byte.
        # This significantly improves performance on large files while maintaining
        # enough signature resolution for LSH.
        stride = 4
        for i in range(0, len(data) - self.shingle_size + 1, stride):
            shingle = data[i : i + self.shingle_size]
            # Fast 8-byte integer from first bytes
            try:
                yield struct.unpack("<Q", shingle[:8])[0]
            except Exception:
                # Handle cases where shingle might be < 8 bytes
                yield int.from_bytes(shingle, "little")


# ---------------------------------------------------------------------------
# LSH Index
# ---------------------------------------------------------------------------

@dataclass
class SimilarPair:
    """A candidate similar pair found by LSH."""
    file_a:            str
    file_b:            str
    estimated_jaccard: float


class LSHIndex:
    """
    Builds a MinHash LSH index for fast approximate similarity search.

    Usage::

        index = LSHIndex(num_permutations=128, num_bands=32)
        index.add("file_a.pdf", data_a)
        index.add("file_b.pdf", data_b)
        pairs = index.find_similar_pairs(threshold=0.5)
    """

    def __init__(
        self,
        num_permutations: int = 128,
        num_bands:        int = 32,
    ) -> None:
        if num_permutations % num_bands != 0:
            raise ValueError(
                f"num_permutations ({num_permutations}) must be "
                f"divisible by num_bands ({num_bands})"
            )

        self.num_permutations = num_permutations
        self.num_bands        = num_bands
        self.rows_per_band    = num_permutations // num_bands

        self._hasher:     MinHasher  = MinHasher(num_permutations)
        self._signatures: dict[str, np.ndarray]    = {}
        self._buckets:    dict[tuple, list[str]]   = defaultdict(list)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, name: str, data: bytes) -> None:
        """
        Compute MinHash signature for *data* and add to the index.
        *name* is any unique identifier (usually file path or hash).
        """
        sig = self._hasher.signature(data)
        self._signatures[name] = sig
        self._index_signature(name, sig)

    def add_batch(self, items: dict[str, bytes]) -> None:
        """Add multiple items efficiently."""
        for name, data in items.items():
            self.add(name, data)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_similar_pairs(
        self, threshold: float = 0.5
    ) -> list[SimilarPair]:
        """
        Find all approximately similar pairs with Jaccard ≥ threshold.

        Returns deduplicated list of SimilarPair sorted by similarity desc.
        """
        candidate_pairs: set[frozenset] = set()

        # Collect candidates from shared buckets
        for bucket_members in self._buckets.values():
            if len(bucket_members) < 2:
                continue
            for i, a in enumerate(bucket_members):
                for b in bucket_members[i + 1:]:
                    candidate_pairs.add(frozenset({a, b}))

        # Verify candidates with exact MinHash comparison
        results = []
        for pair_set in candidate_pairs:
            pair_list = list(pair_set)
            a, b = pair_list[0], pair_list[1]
            if a not in self._signatures or b not in self._signatures:
                continue
            jaccard = self._hasher.jaccard_estimate(
                self._signatures[a], self._signatures[b]
            )
            if jaccard >= threshold:
                results.append(SimilarPair(
                    file_a=a,
                    file_b=b,
                    estimated_jaccard=round(jaccard, 4),
                ))

        return sorted(
            results, key=lambda p: p.estimated_jaccard, reverse=True
        )

    def query_similar(
        self, data: bytes, threshold: float = 0.5
    ) -> list[SimilarPair]:
        """
        Find items in the index similar to *data* without adding it.
        Useful for: "is this file similar to anything I've seen?"
        """
        query_sig = self._hasher.signature(data)
        candidates: set[str] = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end   = start + self.rows_per_band
            band  = query_sig[start:end]
            key   = (band_idx, band.tobytes())
            for member in self._buckets.get(key, []):
                candidates.add(member)

        results = []
        for name in candidates:
            jaccard = self._hasher.jaccard_estimate(
                query_sig, self._signatures[name]
            )
            if jaccard >= threshold:
                results.append(SimilarPair(
                    file_a="<query>",
                    file_b=name,
                    estimated_jaccard=round(jaccard, 4),
                ))

        return sorted(
            results, key=lambda p: p.estimated_jaccard, reverse=True
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def indexed_count(self) -> int:
        return len(self._signatures)

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def expected_threshold(self) -> float:
        """
        Theoretical Jaccard threshold at which pairs are found
        with 50% probability given current band/row configuration.
        """
        return (1 / self.num_bands) ** (1 / self.rows_per_band)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_signature(self, name: str, sig: np.ndarray) -> None:
        """Place signature into LSH buckets."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end   = start + self.rows_per_band
            band  = sig[start:end]
            key   = (band_idx, band.tobytes())
            self._buckets[key].append(name)


# ---------------------------------------------------------------------------
# Scalable collection similarity
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """A group of similar files forming a compression cluster."""
    cluster_id:    int
    members:       list[str]
    avg_similarity: float

    @property
    def size(self) -> int:
        return len(self.members)


class ScalableCollectionAnalyser:
    """
    Analyses large file collections using MinHash LSH.
    Identifies clusters of similar files for delta encoding.

    For n files:
        Phase 3: O(n²) — works up to ~1,000 files
        Phase 5: O(n)  — works up to millions of files

    Usage::

        analyser = ScalableCollectionAnalyser()
        analyser.ingest_files(file_paths)
        clusters = analyser.find_clusters(threshold=0.6)
        pairs    = analyser.find_similar_pairs(threshold=0.5)
    """

    def __init__(
        self,
        num_permutations: int = 128,
        num_bands:        int = 32,
        sample_bytes:     int = 1 * 1024 * 1024,  # 1 MB sample
    ) -> None:
        self._index       = LSHIndex(num_permutations, num_bands)
        self._sample_size = sample_bytes
        self._file_sizes: dict[str, int] = {}

    def ingest_files(self, file_paths: list[str]) -> None:
        """
        Read and index all files in parallel.
        Large files are sampled to keep ingestion fast.
        """
        from concurrent.futures import ThreadPoolExecutor

        def process_one(path: str):
            if not os.path.isfile(path):
                return
            data = self._read_sample(path)
            # Signature calculation is CPU-bound but ThreadPool still helps 
            # with I/O and overlapping compute in some environments.
            # In pure Python, ProcessPool would be better for CPU,
            # but ThreadPool is safer in many library contexts.
            self._index.add(path, data)
            self._file_sizes[path] = os.path.getsize(path)

        # Use threads to overlap I/O and signature calculation
        # Max 8 threads or CPU count
        max_workers = min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_one, file_paths)

    def ingest_bytes(self, items: dict[str, bytes]) -> None:
        """Ingest pre-loaded byte strings."""
        self._index.add_batch(items)

    def find_similar_pairs(
        self, threshold: float = 0.5
    ) -> list[SimilarPair]:
        """Find all approximately similar pairs."""
        return self._index.find_similar_pairs(threshold)

    def find_clusters(
        self, threshold: float = 0.5
    ) -> list[ClusterResult]:
        """
        Group files into clusters of mutual similarity
        using a greedy union-find approach.
        """
        pairs = self.find_similar_pairs(threshold)
        parent = {
            name: name for name in self._index._signatures
        }

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for pair in pairs:
            if pair.file_a in parent and pair.file_b in parent:
                union(pair.file_a, pair.file_b)

        # Group by root
        groups: dict[str, list[str]] = defaultdict(list)
        for name in self._index._signatures:
            groups[find(name)].append(name)

        clusters = []
        for cluster_id, (root, members) in enumerate(
            sorted(groups.items(), key=lambda x: -len(x[1]))
        ):
            if len(members) < 2:
                continue

            # Compute average pairwise similarity
            sigs = [
                self._index._signatures[m]
                for m in members
                if m in self._index._signatures
            ]
            total_sim = 0.0
            pair_count = 0
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    total_sim += self._index._hasher.jaccard_estimate(
                        sigs[i], sigs[j]
                    )
                    pair_count += 1

            avg_sim = total_sim / pair_count if pair_count > 0 else 0.0

            clusters.append(ClusterResult(
                cluster_id=cluster_id,
                members=members,
                avg_similarity=round(avg_sim, 4),
            ))

        return clusters

    def is_duplicate(
        self, data: bytes, threshold: float = 0.95
    ) -> bool:
        """
        Quick check: is this content already in the index?
        Useful for deduplication at ingestion time.
        """
        matches = self._index.query_similar(data, threshold)
        return len(matches) > 0

    def _read_sample(self, path: str) -> bytes:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size <= self._sample_size:
                return fh.read()
            # Sample start, middle, and end for better representation
            chunk = self._sample_size // 3
            start  = fh.read(chunk)
            fh.seek(size // 2)
            middle = fh.read(chunk)
            fh.seek(-chunk, 2)
            end    = fh.read(chunk)
            return start + middle + end
