"""
AirOne Entropy Analyser
Provides byte-level and block-level entropy measurements
used by the AnalysisEngine and StrategySelector.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass


@dataclass
class EntropyReport:
    """
    Entropy measurements for a file.

    Attributes
    ----------
    global_entropy:
        Shannon entropy (bits/byte) over the full sample.
        Range: 0 (perfectly repetitive) → 8 (perfectly random).
    block_entropies:
        Entropy measured in each sampled block.
    min_block_entropy:
        Lowest block entropy — reveals locally compressible regions.
    max_block_entropy:
        Highest block entropy.
    mean_block_entropy:
        Average across blocks.
    compressibility_estimate:
        Rough estimate of achievable compression ratio with a generic
        compressor (higher = more compressible).
    """
    global_entropy: float
    block_entropies: list[float]
    min_block_entropy: float
    max_block_entropy: float
    mean_block_entropy: float
    compressibility_estimate: float

    @property
    def is_highly_compressible(self) -> bool:
        return self.global_entropy < 4.0

    @property
    def is_random(self) -> bool:
        return self.global_entropy > 7.9


class EntropyAnalyser:
    """
    Computes Shannon entropy at file and block levels.

    Parameters
    ----------
    sample_bytes:
        How many bytes to read from the file.
        Default 4 MB balances accuracy and speed.
    block_size:
        Block size (bytes) for local entropy calculation.
    """

    def __init__(
        self,
        sample_bytes: int = 4 * 1024 * 1024,
        block_size: int = 64 * 1024,
    ) -> None:
        self.sample_bytes = sample_bytes
        self.block_size = block_size

    def analyse(self, file_path: str) -> EntropyReport:
        data = self._read_sample(file_path)

        global_entropy = self._shannon_entropy(data)
        block_entropies = self._block_entropies(data)

        if block_entropies:
            min_e = min(block_entropies)
            max_e = max(block_entropies)
            mean_e = sum(block_entropies) / len(block_entropies)
        else:
            min_e = max_e = mean_e = global_entropy

        return EntropyReport(
            global_entropy=global_entropy,
            block_entropies=block_entropies,
            min_block_entropy=min_e,
            max_block_entropy=max_e,
            mean_block_entropy=mean_e,
            compressibility_estimate=self._estimate_ratio(global_entropy),
        )

    # ------------------------------------------------------------------

    def _read_sample(self, file_path: str) -> bytes:
        file_size = os.path.getsize(file_path)
        read_size = min(file_size, self.sample_bytes)
        with open(file_path, "rb") as fh:
            return fh.read(read_size)

    @staticmethod
    def _shannon_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        counts = Counter(data)
        n = len(data)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _block_entropies(self, data: bytes) -> list[float]:
        entropies = []
        for start in range(0, len(data), self.block_size):
            block = data[start : start + self.block_size]
            if block:
                entropies.append(self._shannon_entropy(block))
        return entropies

    @staticmethod
    def _estimate_ratio(entropy: float) -> float:
        """
        Heuristic: maps entropy (0-8) to approximate compression ratio.

        entropy=0 → infinite compression (all same bytes)
        entropy=8 → ~1.0x (random, incompressible)
        """
        if entropy >= 8.0:
            return 1.0
        if entropy <= 0.0:
            return 100.0                   # theoretical max
        # Rough inverse relationship
        return round(8.0 / max(entropy, 0.1), 2)
