"""
AirOne Benchmark Suite - Phase 4

Provides systematic, reproducible benchmarks across:
    - All registered compression strategies
    - Multiple file types and sizes
    - Compression ratio, speed, and memory usage

Output formats: terminal table, JSON, CSV
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from airone.analysis.engine import AnalysisEngine
from airone.compressors.base import BaseCompressor, CompressionResult
from airone.core.verification import verify_lossless


@dataclass
class BenchmarkRecord:
    """Results for one (file, strategy) combination."""
    file_name:         str
    file_size:         int
    file_format:       str
    strategy:          str
    compressed_size:   int
    ratio:             float
    compression_ms:    float
    decompression_ms:  float
    peak_memory_kb:    float
    lossless_verified: bool
    error:             Optional[str] = None

    @property
    def compression_speed_mbps(self) -> float:
        if self.compression_ms <= 0:
            return 0.0
        return (self.file_size / 1024 / 1024) / (self.compression_ms / 1000)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all files and strategies."""
    records: list[BenchmarkRecord] = field(default_factory=list)
    total_time_s: float = 0.0

    def best_per_file(self) -> dict[str, BenchmarkRecord]:
        """Returns the best-ratio record for each file."""
        best: dict[str, BenchmarkRecord] = {}
        for r in self.records:
            if r.error:
                continue
            if r.file_name not in best or r.ratio > best[r.file_name].ratio:
                best[r.file_name] = r
        return best

    def strategy_summary(self) -> dict[str, dict]:
        """Average metrics per strategy."""
        from collections import defaultdict
        sums: dict[str, list] = defaultdict(list)
        for r in self.records:
            if not r.error:
                sums[r.strategy].append(r)

        summary = {}
        for strategy, records in sums.items():
            summary[strategy] = {
                "count":          len(records),
                "avg_ratio":      sum(r.ratio for r in records) / len(records),
                "avg_comp_ms":    sum(r.compression_ms for r in records) / len(records),
                "avg_decomp_ms":  sum(r.decompression_ms for r in records) / len(records),
                "all_lossless":   all(r.lossless_verified for r in records),
            }
        return summary

    def to_json(self) -> str:
        return json.dumps(
            {
                "records":       [asdict(r) for r in self.records],
                "total_time_s":  self.total_time_s,
                "best_per_file": {
                    k: asdict(v)
                    for k, v in self.best_per_file().items()
                },
                "strategy_summary": self.strategy_summary(),
            },
            indent=2,
        )

    def to_csv(self) -> str:
        buf = io.StringIO()
        if not self.records:
            return ""
        # Need to handle nested dicts if asdict is used directly
        # but here it is flat enough.
        fieldnames = [
            "file_name", "file_size", "file_format", "strategy",
            "compressed_size", "ratio", "compression_ms",
            "decompression_ms", "peak_memory_kb", "lossless_verified", "error"
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in self.records:
            row = asdict(r)
            writer.writerow(row)
        return buf.getvalue()

    def print_table(self) -> None:
        """Print a formatted terminal table."""
        if not self.records:
            print("No benchmark results.")
            return

        header = (
            f"{'File':<25} {'Strategy':<22} {'Ratio':>7} "
            f"{'Comp ms':>9} {'Decomp ms':>10} {'Lossless':>9}"
        )
        sep = "─" * len(header)

        print(f"\n{'AirOne Benchmark Results':^{len(header)}}")
        print(sep)
        print(header)
        print(sep)

        for r in sorted(self.records, key=lambda x: x.file_name):
            if r.error:
                status = f"ERROR: {r.error[:30]}"
                print(
                    f"{r.file_name:<25} {r.strategy:<22} "
                    f"{'—':>7} {'—':>9} {'—':>10} {status}"
                )
            else:
                lossless = "✓" if r.lossless_verified else "✗"
                print(
                    f"{r.file_name:<25} {r.strategy:<22} "
                    f"{r.ratio:>7.2f}x {r.compression_ms:>9.1f} "
                    f"{r.decompression_ms:>10.1f} {lossless:>9}"
                )

        print(sep)

        # Strategy summary
        summary = self.strategy_summary()
        print("\nStrategy Averages:")
        for strategy, stats in sorted(
            summary.items(), key=lambda x: -x[1]["avg_ratio"]
        ):
            print(
                f"  {strategy:<22} "
                f"avg ratio={stats['avg_ratio']:.2f}x  "
                f"lossless={'✓' if stats['all_lossless'] else '✗'}"
            )

        print(f"\nTotal benchmark time: {self.total_time_s:.2f}s\n")


class BenchmarkRunner:
    """
    Runs compression benchmarks on a set of files
    using multiple strategies.

    Usage::

        runner = BenchmarkRunner()
        report = runner.run(
            file_paths=["doc.pdf", "photo.png", "data.json"],
            strategies=["traditional_zstd", "traditional_brotli", "semantic_pdf"],
        )
        report.print_table()
        report.to_json()
    """

    def __init__(self, registry=None) -> None:
        from airone.orchestrator.orchestrator import _build_default_registry
        self._registry = registry or _build_default_registry()
        self._analyser = AnalysisEngine()

    def run(
        self,
        file_paths:  list[str],
        strategies:  Optional[list[str]] = None,
    ) -> BenchmarkReport:
        """
        Benchmark all (file, strategy) combinations.
        """
        all_strategies = self._registry.list_names()
        strategies = strategies or all_strategies
        report     = BenchmarkReport()
        start      = time.perf_counter()

        for file_path in file_paths:
            if not os.path.isfile(file_path):
                continue

            analysis = self._safe_analyse(file_path)

            with open(file_path, "rb") as fh:
                raw_data = fh.read()

            for strategy_name in strategies:
                record = self._benchmark_one(
                    file_path, raw_data, strategy_name, analysis
                )
                report.records.append(record)

        report.total_time_s = time.perf_counter() - start
        return report

    # ------------------------------------------------------------------

    def _benchmark_one(
        self,
        file_path:     str,
        raw_data:      bytes,
        strategy_name: str,
        analysis,
    ) -> BenchmarkRecord:
        file_name   = os.path.basename(file_path)
        file_format = (
            analysis.format.type if analysis else "UNKNOWN"
        )

        try:
            compressor = self._registry.get(strategy_name)

            if not compressor.can_handle(analysis):
                return BenchmarkRecord(
                    file_name=file_name,
                    file_size=len(raw_data),
                    file_format=file_format,
                    strategy=strategy_name,
                    compressed_size=0,
                    ratio=0.0,
                    compression_ms=0.0,
                    decompression_ms=0.0,
                    peak_memory_kb=0.0,
                    lossless_verified=False,
                    error="Strategy cannot handle this file type",
                )

            # Compression with memory tracking
            tracemalloc.start()
            t0 = time.perf_counter()
            result = compressor.compress(raw_data, analysis)
            comp_ms = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Decompression timing
            t0 = time.perf_counter()
            decompressed = compressor.decompress(
                result.compressed_data, result.metadata
            )
            decomp_ms = (time.perf_counter() - t0) * 1000

            # Lossless verification
            lossless = verify_lossless(raw_data, decompressed)

            return BenchmarkRecord(
                file_name=file_name,
                file_size=len(raw_data),
                file_format=file_format,
                strategy=strategy_name,
                compressed_size=result.compressed_size,
                ratio=result.ratio,
                compression_ms=comp_ms,
                decompression_ms=decomp_ms,
                peak_memory_kb=peak / 1024,
                lossless_verified=lossless,
            )

        except Exception as exc:
            return BenchmarkRecord(
                file_name=file_name,
                file_size=len(raw_data),
                file_format=file_format,
                strategy=strategy_name,
                compressed_size=0,
                ratio=0.0,
                compression_ms=0.0,
                decompression_ms=0.0,
                peak_memory_kb=0.0,
                lossless_verified=False,
                error=str(exc)[:100],
            )

    def _safe_analyse(self, file_path: str):
        try:
            return self._analyser.analyse(file_path)
        except Exception:
            return None
