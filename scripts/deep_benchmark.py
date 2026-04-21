#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirOne Deep Benchmark & Research Script
========================================
Generates synthetic test corpora across all file types and sizes,
runs every compression strategy, and produces a full research report.

Run:
    python scripts/deep_benchmark.py --output results/benchmark_$(date +%Y%m%d).json
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import struct
import sys
import time
import tracemalloc
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Make sure the package is importable from the project root ──────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from airone.benchmarks.runner import BenchmarkRecord, BenchmarkReport, BenchmarkRunner
from airone.collection.delta import DeltaCollectionEncoder
from airone.collection.lsh import LSHIndex, MinHasher, ScalableCollectionAnalyser
from airone.compressors.traditional.brotli import BrotliCompressor
from airone.compressors.traditional.lzma import LZMACompressor
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.core.streaming import StreamingCompressor

RESULTS_DIR = Path(__file__).parent.parent / "results"
CORPUS_DIR  = RESULTS_DIR / "corpus"


# ──────────────────────────────────────────────────────────────────────────────
# Corpus generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_repetitive_text(size: int) -> bytes:
    """Highly compressible English-like text."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "AirOne", "compress", "semantic", "intelligent", "lossless", "ratio",
        "document", "analysis", "strategy", "pipeline", "neural", "codec",
    ]
    lines = []
    while sum(len(l) for l in lines) < size:
        line = " ".join(random.choices(words, k=random.randint(8, 20)))
        lines.append(line)
    return "\n".join(lines).encode("utf-8")[:size]


def gen_low_entropy_binary(size: int) -> bytes:
    """Binary data with low entropy (run-length like)."""
    out = bytearray()
    while len(out) < size:
        byte_val = random.randint(0, 15)   # low range = low entropy
        run = random.randint(4, 32)
        out.extend([byte_val] * run)
    return bytes(out[:size])


def gen_high_entropy_binary(size: int) -> bytes:
    """Pseudo-random bytes — near-incompressible."""
    return os.urandom(size)


def gen_structured_json(size: int) -> bytes:
    """Repetitive JSON structure mimicking API log data."""
    record_template = {
        "timestamp": "2026-01-01T00:00:00Z",
        "user_id":   "usr_12345",
        "action":    "compress",
        "file_type": "PDF",
        "ratio":     0.0,
        "latency_ms": 0,
        "status": "success",
    }
    records = []
    while True:
        r = dict(record_template)
        r["ratio"]      = round(random.uniform(2.0, 50.0), 2)
        r["latency_ms"] = random.randint(10, 500)
        r["user_id"]    = f"usr_{random.randint(10000, 99999)}"
        records.append(r)
        payload = json.dumps({"records": records}, indent=2).encode()
        if len(payload) >= size:
            return payload[:size]


def gen_gradient_png(width: int = 512, height: int = 512) -> bytes:
    """A gradient PNG — ideal for procedural compression."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for x in range(width):
            for y in range(height):
                r = int(x / width * 255)
                g = int(y / height * 255)
                b = int((x + y) / (width + height) * 255)
                pixels[x, y] = (r, g, b)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return gen_low_entropy_binary(width * height * 3)


def gen_solid_png(width: int = 256, height: int = 256) -> bytes:
    """Solid colour PNG — extreme compressibility."""
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(42, 128, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return gen_low_entropy_binary(width * height * 3)


def gen_docx(text: str, n_paragraphs: int = 50) -> bytes:
    """Minimal DOCX with repetitive text content."""
    buf = io.BytesIO()
    paragraphs = "".join(
        f"<w:p><w:r><w:t>{text} paragraph {i}</w:t></w:r></w:p>"
        for i in range(n_paragraphs)
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>')
        zf.writestr("_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')
        zf.writestr("word/document.xml",
            f'<?xml version="1.0"?>'
            f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f'<w:body>{paragraphs}</w:body></w:document>')
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Corpus manifest
# ──────────────────────────────────────────────────────────────────────────────

CORPUS_FILES = [
    # (filename, generator_callable, kwargs)
    ("text_small.txt",        gen_repetitive_text,    {"size": 10_000}),
    ("text_medium.txt",       gen_repetitive_text,    {"size": 100_000}),
    ("text_large.txt",        gen_repetitive_text,    {"size": 1_000_000}),
    ("json_log_medium.json",  gen_structured_json,    {"size": 200_000}),
    ("json_log_large.json",   gen_structured_json,    {"size": 1_000_000}),
    ("binary_low_entropy.bin",gen_low_entropy_binary, {"size": 500_000}),
    ("binary_high_entropy.bin",gen_high_entropy_binary,{"size": 500_000}),
    ("gradient_512.png",      gen_gradient_png,       {"width": 512, "height": 512}),
    ("solid_256.png",         gen_solid_png,          {"width": 256, "height": 256}),
    ("report.docx",           gen_docx,               {"text": "Quarterly report AirOne compression platform", "n_paragraphs": 100}),
]


def build_corpus() -> list[str]:
    """Generate test corpus files, return list of paths."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, gen_fn, kwargs in CORPUS_FILES:
        out_path = CORPUS_DIR / filename
        if not out_path.exists():
            print(f"  Generating {filename}...", end="", flush=True)
            data = gen_fn(**kwargs)
            out_path.write_bytes(data)
            print(f" {len(data):,} bytes")
        else:
            print(f"  Using cached {filename} ({out_path.stat().st_size:,} bytes)")
        paths.append(str(out_path))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Delta encoding benchmark
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaBenchResult:
    collection_name: str
    n_files:         int
    total_original_bytes: int
    total_delta_bytes:    int
    savings_pct:     float
    avg_ratio:       float
    encode_ms:       float
    decode_ms:       float
    lossless:        bool


def benchmark_delta(tmp_dir: Path) -> list[DeltaBenchResult]:
    """Run delta encoding benchmarks on synthetic similar-file collections."""
    results = []
    encoder = DeltaCollectionEncoder()

    # Test case 1: monthly reports (high similarity)
    base_report = gen_repetitive_text(80_000)
    monthly = {
        f"report_{m}.txt": base_report + f"\n\nMonth: {m}\nDate: 2026-{m:02d}-01".encode()
        for m in range(1, 13)
    }
    total_orig = sum(len(v) for v in monthly.values())
    t0 = time.perf_counter()
    bundle = encoder.encode_collection(monthly)
    enc_ms = (time.perf_counter() - t0) * 1000

    total_delta = len(bundle["reference_data"]) + sum(
        len(e["delta"]) for e in bundle["deltas"].values()
    )

    t0 = time.perf_counter()
    restored = encoder.decode_all(bundle)
    dec_ms = (time.perf_counter() - t0) * 1000
    lossless = all(restored[k] == v for k, v in monthly.items())

    results.append(DeltaBenchResult(
        collection_name="Monthly Reports (12 files)",
        n_files=len(monthly),
        total_original_bytes=total_orig,
        total_delta_bytes=total_delta,
        savings_pct=round((1 - total_delta / total_orig) * 100, 1),
        avg_ratio=round(total_orig / max(total_delta, 1), 2),
        encode_ms=round(enc_ms, 1),
        decode_ms=round(dec_ms, 1),
        lossless=lossless,
    ))

    # Test case 2: versioned configs (very high similarity)
    base_cfg = b'{"version": "1.0", "host": "localhost", "port": 8080, "workers": 4}\n' * 500
    versions = {
        f"config_v{i}.json": base_cfg.replace(b'"1.0"', f'"{i}.0"'.encode())
                                       .replace(b"8080", str(8080 + i).encode())
        for i in range(1, 11)
    }
    total_orig = sum(len(v) for v in versions.values())
    t0 = time.perf_counter()
    bundle = encoder.encode_collection(versions)
    enc_ms = (time.perf_counter() - t0) * 1000
    total_delta = len(bundle["reference_data"]) + sum(
        len(e["delta"]) for e in bundle["deltas"].values()
    )
    t0 = time.perf_counter()
    restored = encoder.decode_all(bundle)
    dec_ms = (time.perf_counter() - t0) * 1000
    lossless = all(restored[k] == v for k, v in versions.items())

    results.append(DeltaBenchResult(
        collection_name="Versioned Configs (10 files)",
        n_files=len(versions),
        total_original_bytes=total_orig,
        total_delta_bytes=total_delta,
        savings_pct=round((1 - total_delta / total_orig) * 100, 1),
        avg_ratio=round(total_orig / max(total_delta, 1), 2),
        encode_ms=round(enc_ms, 1),
        decode_ms=round(dec_ms, 1),
        lossless=lossless,
    ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# LSH benchmark
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LSHBenchResult:
    n_files:         int
    n_similar_pairs: int
    precision:       float
    recall:          float
    index_ms:        float
    query_ms:        float


def benchmark_lsh() -> LSHBenchResult:
    """Benchmark LSH index speed and accuracy vs brute force."""
    base = b"shared content block for LSH benchmark test " * 400
    n = 200   # 200 files

    files: dict[str, bytes] = {}
    # 100 similar files
    for i in range(100):
        content = base + f"unique-{i}".encode() * 20
        files[f"similar_{i:03d}.txt"] = content
    # 100 unrelated files
    for i in range(100):
        files[f"random_{i:03d}.bin"] = os.urandom(len(base))

    analyser = ScalableCollectionAnalyser(num_permutations=128, num_bands=32)

    t0 = time.perf_counter()
    analyser.ingest_bytes(files)
    index_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    pairs = analyser.find_similar_pairs(threshold=0.5)
    query_ms = (time.perf_counter() - t0) * 1000

    # LSH true positives: only similar_* should pair with similar_*
    lsh_pairs = {
        frozenset({p.file_a, p.file_b}) for p in pairs
    }
    true_positives = sum(
        1 for p in lsh_pairs
        if all("similar_" in x for x in p)
    )
    false_positives = len(lsh_pairs) - true_positives
    # Max possible similar pairs = C(100,2) = 4950
    max_similar = 100 * 99 // 2

    precision = true_positives / max(len(lsh_pairs), 1)
    recall    = true_positives / max_similar

    return LSHBenchResult(
        n_files=n,
        n_similar_pairs=len(lsh_pairs),
        precision=round(precision, 4),
        recall=round(recall, 4),
        index_ms=round(index_ms, 1),
        query_ms=round(query_ms, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Streaming benchmark
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamingBenchResult:
    file_size_mb: float
    window_size_kb: int
    window_count:  int
    compress_ms:   float
    decompress_ms: float
    ratio:         float
    lossless:      bool


def benchmark_streaming(tmp_dir: Path) -> list[StreamingBenchResult]:
    """Benchmark streaming compressor across window sizes."""
    results = []

    # Generate a 10MB test file (mix of compressible and random)
    test_data = gen_repetitive_text(5_000_000) + gen_low_entropy_binary(5_000_000)
    test_path = tmp_dir / "stream_test.bin"
    test_path.write_bytes(test_data)
    file_size_mb = len(test_data) / 1024 / 1024

    for window_kb in [64, 256, 1024, 4096]:
        out_path  = tmp_dir / f"stream_{window_kb}kb.air"
        rest_path = tmp_dir / f"stream_{window_kb}kb_restored.bin"

        compressor = StreamingCompressor(window_size=window_kb * 1024)

        t0 = time.perf_counter()
        manifest = compressor.compress_file(str(test_path), str(out_path))
        comp_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        compressor.decompress_file(str(out_path), str(rest_path))
        decomp_ms = (time.perf_counter() - t0) * 1000

        compressed_size = out_path.stat().st_size
        ratio = len(test_data) / max(compressed_size, 1)
        lossless = rest_path.read_bytes() == test_data

        results.append(StreamingBenchResult(
            file_size_mb=round(file_size_mb, 2),
            window_size_kb=window_kb,
            window_count=manifest.window_count,
            compress_ms=round(comp_ms, 1),
            decompress_ms=round(decomp_ms, 1),
            ratio=round(ratio, 2),
            lossless=lossless,
        ))

        # Cleanup output files for disk space
        out_path.unlink(missing_ok=True)
        rest_path.unlink(missing_ok=True)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Per-strategy direct benchmarks (bypass orchestrator for exhaustive comparison)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyBenchResult:
    strategy:       str
    file_type:      str
    original_bytes: int
    compressed_bytes: int
    ratio:          float
    compression_ms: float
    decompression_ms: float
    peak_memory_kb: float
    throughput_mbps: float
    lossless:       bool


def benchmark_strategies_direct(corpus: list[tuple[str, bytes]]) -> list[StrategyBenchResult]:
    """Benchmark ZSTD, Brotli, LZMA head-to-head on each corpus file."""
    from airone.core.verification import verify_lossless

    codecs = {
        "ZSTD (level 3)":  ZstdCompressor(level=3),
        "ZSTD (level 19)": ZstdCompressor(level=19),
        "Brotli (q=4)":    BrotliCompressor(quality=4),
        "Brotli (q=11)":   BrotliCompressor(quality=11),
        "LZMA":            LZMACompressor(),
    }

    results = []
    for file_name, data in corpus:
        for codec_name, codec in codecs.items():
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                result = codec.compress(data)
                comp_ms = (time.perf_counter() - t0) * 1000
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                t0 = time.perf_counter()
                restored = codec.decompress(result.compressed_data, result.metadata)
                decomp_ms = (time.perf_counter() - t0) * 1000

                lossless = verify_lossless(data, restored)
                throughput = (len(data) / 1024 / 1024) / (comp_ms / 1000) if comp_ms > 0 else 0

                results.append(StrategyBenchResult(
                    strategy=codec_name,
                    file_type=file_name,
                    original_bytes=len(data),
                    compressed_bytes=len(result.compressed_data),
                    ratio=result.ratio,
                    compression_ms=round(comp_ms, 2),
                    decompression_ms=round(decomp_ms, 2),
                    peak_memory_kb=round(peak / 1024, 1),
                    throughput_mbps=round(throughput, 2),
                    lossless=lossless,
                ))
            except Exception as exc:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
                results.append(StrategyBenchResult(
                    strategy=codec_name,
                    file_type=file_name,
                    original_bytes=len(data),
                    compressed_bytes=0,
                    ratio=0.0,
                    compression_ms=0,
                    decompression_ms=0,
                    peak_memory_kb=0,
                    throughput_mbps=0,
                    lossless=False,
                ))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    w = 88
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def print_strategy_table(results: list[StrategyBenchResult]) -> None:
    print_section("Strategy-by-Strategy Compression Comparison")
    # Group by file type
    from collections import defaultdict
    by_file: dict[str, list[StrategyBenchResult]] = defaultdict(list)
    for r in results:
        by_file[r.file_type].append(r)

    for file_name, records in sorted(by_file.items()):
        print(f"\n  ── {file_name} ({records[0].original_bytes:,} bytes) ──")
        print(f"  {'Strategy':<22} {'Ratio':>7} {'Comp ms':>9} {'Decomp ms':>10} {'MB/s':>7} {'Mem KB':>8}  {'OK':>4}")
        print(f"  {'─'*22} {'─'*7} {'─'*9} {'─'*10} {'─'*7} {'─'*8}  {'─'*4}")
        for r in sorted(records, key=lambda x: -x.ratio):
            ok = "✓" if r.lossless else "✗"
            print(
                f"  {r.strategy:<22} {r.ratio:>7.2f}x "
                f"{r.compression_ms:>9.1f} {r.decompression_ms:>10.1f} "
                f"{r.throughput_mbps:>7.1f} {r.peak_memory_kb:>8.0f}  {ok:>4}"
            )


def print_delta_table(results: list[DeltaBenchResult]) -> None:
    print_section("Delta Encoding Performance")
    print(f"\n  {'Collection':<35} {'Files':>5} {'Orig MB':>8} {'Delta MB':>9} "
          f"{'Save%':>7} {'Ratio':>6} {'Enc ms':>8} {'Dec ms':>8} {'OK':>4}")
    print(f"  {'─'*35} {'─'*5} {'─'*8} {'─'*9} {'─'*7} {'─'*6} {'─'*8} {'─'*8} {'─'*4}")
    for r in results:
        ok = "✓" if r.lossless else "✗"
        print(
            f"  {r.collection_name:<35} {r.n_files:>5} "
            f"{r.total_original_bytes/1e6:>8.2f} {r.total_delta_bytes/1e6:>9.2f} "
            f"{r.savings_pct:>6.1f}% {r.avg_ratio:>6.2f}x "
            f"{r.encode_ms:>8.1f} {r.decode_ms:>8.1f} {ok:>4}"
        )


def print_lsh_table(result: LSHBenchResult) -> None:
    print_section("LSH Similarity Engine Performance")
    print(f"""
  Files indexed     : {result.n_files:,}
  Similar pairs found: {result.n_similar_pairs:,}
  Precision         : {result.precision:.1%}
  Recall            : {result.recall:.1%}
  Indexing time     : {result.index_ms:.1f} ms
  Query time        : {result.query_ms:.1f} ms
  Throughput        : {result.n_files / (result.index_ms / 1000):.0f} files/sec
""")


def print_streaming_table(results: list[StreamingBenchResult]) -> None:
    print_section("Streaming Compressor — Window Size Impact")
    print(f"\n  {'Window':>8} {'Windows':>7} {'Ratio':>7} {'Comp ms':>9} "
          f"{'Decomp ms':>10} {'Comp MB/s':>10} {'OK':>4}")
    print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*10} {'─'*4}")
    for r in results:
        ok = "✓" if r.lossless else "✗"
        comp_mbps = r.file_size_mb / (r.compress_ms / 1000) if r.compress_ms > 0 else 0
        print(
            f"  {r.window_size_kb:>6}KB {r.window_count:>7} {r.ratio:>7.2f}x "
            f"{r.compress_ms:>9.0f} {r.decompress_ms:>10.0f} {comp_mbps:>10.1f} {ok:>4}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import tempfile

    print("\n" + "█" * 88)
    print("  AirOne v1.0 — Deep Benchmark & Research Suite")
    print("█" * 88)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Build corpus ──────────────────────────────────────────────────────
    print_section("Generating Test Corpus")
    corpus_paths = build_corpus()

    # Load into memory for direct strategy comparisons
    direct_corpus: list[tuple[str, bytes]] = []
    for p in corpus_paths:
        path = Path(p)
        if path.stat().st_size < 2_000_000:    # skip huge files for direct tests
            direct_corpus.append((path.name, path.read_bytes()))

    # ── 2. BenchmarkRunner (orchestrated) ───────────────────────────────────
    print_section("Orchestrated Benchmark (All Registered Strategies)")
    runner = BenchmarkRunner()
    report = runner.run(
        corpus_paths,
        strategies=["traditional_zstd", "traditional_brotli", "traditional_lzma",
                    "procedural_gradient"],
    )
    report.print_table()

    # ── 3. Direct per-codec benchmarks ──────────────────────────────────────
    strategy_results = benchmark_strategies_direct(direct_corpus)
    print_strategy_table(strategy_results)

    # ── 4. Delta encoding ────────────────────────────────────────────────────
    print_section("Delta Encoding Benchmarks")
    with tempfile.TemporaryDirectory() as tmp:
        delta_results = benchmark_delta(Path(tmp))
    print_delta_table(delta_results)

    # ── 5. LSH similarity ────────────────────────────────────────────────────
    lsh_result = benchmark_lsh()
    print_lsh_table(lsh_result)

    # ── 6. Streaming ─────────────────────────────────────────────────────────
    print_section("Streaming Compressor Benchmarks")
    with tempfile.TemporaryDirectory() as tmp:
        streaming_results = benchmark_streaming(Path(tmp))
    print_streaming_table(streaming_results)

    # ── 7. Save full JSON report ─────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    full_report = {
        "airone_version": "1.0.0",
        "run_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_files":   [str(p) for p in corpus_paths],
        "orchestrated":   json.loads(report.to_json()),
        "strategy_direct": [asdict(r) for r in strategy_results],
        "delta":          [asdict(r) for r in delta_results],
        "lsh":            asdict(lsh_result),
        "streaming":      [asdict(r) for r in streaming_results],
    }

    report_path.write_text(json.dumps(full_report, indent=2, default=str))
    print(f"\n  ✓ JSON report saved to: {report_path}")

    # ── 8. Print final summary ────────────────────────────────────────────────
    print_section("Executive Summary")
    summary = report.strategy_summary()
    best_strategy = max(summary.items(), key=lambda x: x[1]["avg_ratio"])
    best_delta    = max(delta_results, key=lambda r: r.avg_ratio)
    best_stream   = max(streaming_results, key=lambda r: r.ratio)

    print(f"""
  ┌─ Compression Engine ──────────────────────────────────────────────────────┐
  │  Best overall strategy   : {best_strategy[0]:<40} │
  │  Best avg ratio          : {best_strategy[1]['avg_ratio']:.2f}x{'':<37} │
  │  All lossless?           : {'YES' if best_strategy[1]['all_lossless'] else 'NO':<40} │
  ├─ Delta Encoding ──────────────────────────────────────────────────────────┤
  │  Best collection savings : {best_delta.savings_pct:.1f}%{'':<38} │
  │  Best delta ratio        : {best_delta.avg_ratio:.2f}x — {best_delta.collection_name:<29} │
  ├─ LSH Engine ──────────────────────────────────────────────────────────────┤
  │  Files indexed           : {lsh_result.n_files:<40} │
  │  Precision               : {lsh_result.precision:.1%}{'':<38} │
  │  Recall                  : {lsh_result.recall:.1%}{'':<38} │
  ├─ Streaming ────────────────────────────────────────────────────────────────┤
  │  Best streaming ratio    : {best_stream.ratio:.2f}x at {best_stream.window_size_kb}KB window{'':<23} │
  └───────────────────────────────────────────────────────────────────────────┘
""")
    print(f"  Total benchmark time: {report.total_time_s:.1f}s\n")


if __name__ == "__main__":
    main()
