"""
AirOne Compression Orchestrator — Phase 2
Drives the full compression pipeline end-to-end:
    Analyse → Select Strategies → Execute → Verify → Package
"""

from __future__ import annotations

import time
from typing import Optional

from airone.analysis.engine import AnalysisEngine, AnalysisReport
from airone.compressors.base import CompressionResult
from airone.compressors.traditional.brotli import BrotliCompressor
from airone.compressors.traditional.lzma import LZMACompressor
from airone.compressors.procedural.gradient import GradientCompressor
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.compressors.semantic.pdf import PDFSemanticCompressor
from airone.core.file_format import AirFileFormat
from airone.core.verification import verify_lossless
from airone.exceptions import CompressionError, VerificationError
from airone.strategy.registry import StrategyRegistry
from airone.strategy.selector import StrategySelector


def _build_default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(ZstdCompressor())
    registry.register(BrotliCompressor())
    registry.register(LZMACompressor())
    registry.register(GradientCompressor())
    registry.register(PDFSemanticCompressor())
    return registry


class CompressionOrchestrator:
    """
    Manages the compression pipeline.

    Workflow per file
    -----------------
    1. Analyse → :class:`~airone.analysis.engine.AnalysisReport`
    2. Select strategies → ordered :class:`~airone.strategy.selector.StrategyCandidate` list
    3. Try strategies in order, collecting successful results
    4. Pick the result with the best compression ratio
    5. Verify lossless fidelity
    6. Write the `.air` container

    Early exit:  If a strategy achieves a ratio ≥ *excellent_ratio_threshold*
                 no further strategies are attempted.
    """

    def __init__(
        self,
        registry: Optional[StrategyRegistry] = None,
        excellent_ratio_threshold: float = 50.0,
        always_verify: bool = True,
    ) -> None:
        self._registry   = registry or _build_default_registry()
        self._analyser   = AnalysisEngine()
        self._selector   = StrategySelector(self._registry)
        self._excellent  = excellent_ratio_threshold
        self._verify     = always_verify

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_file(self, input_path: str, output_path: str) -> CompressionResult:
        """
        Compress *input_path* → *output_path* (.air).
        Returns the winning :class:`CompressionResult`.
        """
        # 1. Analyse
        analysis = self._analyser.analyse(input_path)

        # 2. Read raw bytes
        with open(input_path, "rb") as fh:
            raw_data = fh.read()

        # 3. Select strategy candidates
        candidates = self._selector.select(analysis)

        # 4. Try strategies
        best_result: Optional[CompressionResult] = None

        for candidate in candidates:
            try:
                compressor = self._registry.get(candidate.strategy_name)
                result = compressor.compress(raw_data, analysis)

                # Lossless verification
                if self._verify:
                    decompressed = compressor.decompress(
                        result.compressed_data, result.metadata
                    )
                    if not verify_lossless(raw_data, decompressed):
                        raise VerificationError(
                            f"Strategy '{candidate.strategy_name}' failed lossless check."
                        )

                # Track best
                if best_result is None or result.ratio > best_result.ratio:
                    best_result = result

                # Early exit on excellent ratio
                if result.ratio >= self._excellent:
                    break

            except (CompressionError, VerificationError):
                continue    # Try next strategy
            except Exception:
                continue

        if best_result is None:
            raise CompressionError(f"All strategies failed for: {input_path}")

        # 5. Write .air file
        AirFileFormat.write(output_path, best_result)

        return best_result

    def decompress_file(self, input_path: str, output_path: str) -> int:
        """
        Decompress *input_path* (.air) → *output_path*.
        Returns the number of bytes written.
        """
        container = AirFileFormat.read(input_path)
        strategy_name = container["metadata"]["strategy_name"] if "strategy_name" in container["metadata"] else container["metadata"].get("strategy", list(self._registry.list_names())[0])

        compressor = self._registry.get(strategy_name)
        raw_data = compressor.decompress(
            container["compressed_data"],
            container["metadata"],
        )

        with open(output_path, "wb") as fh:
            fh.write(raw_data)

        return len(raw_data)

    def analyse_file(self, file_path: str) -> AnalysisReport:
        """Public wrapper around the analysis engine."""
        return self._analyser.analyse(file_path)
