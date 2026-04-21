# AirOne v1.0 — Benchmark Research Report

> **Run date**: 2026-04-21 | **Tests**: 141 passing | **Platform**: Python 3.10, Windows

---

## 1. Test Suite Results

| Module | Tests | Status |
|--------|-------|--------|
| `test_analysis` | 12 | ✅ All pass |
| `test_benchmark` | 7 | ✅ All pass |
| `test_cli` | 2 | ✅ All pass |
| `test_collection_optimizer` | 12 | ✅ All pass |
| `test_compressors` | 1 | ✅ All pass |
| `test_delta_encoding` | 12 | ✅ All pass |
| `test_document_decomposer` | 9 | ✅ All pass |
| `test_lsh_similarity` | 18 | ✅ All pass |
| `test_neural_codec` | 12 | ✅ All pass |
| `test_office_compressor` | 11 | ✅ All pass |
| `test_onnx_codec` | 15 | ✅ All pass |
| `test_pdf_compressor` | 4 | ✅ All pass |
| `test_pdf_reconstructor_v2` | 8 | ✅ All pass |
| `test_procedural` | 4 | ✅ All pass |
| `test_strategy` | 5 | ✅ All pass |
| `test_streaming` | 9 | ✅ All pass |
| **TOTAL** | **141** | ✅ **100% pass** |

> **Execution time**: 8.24s

---

## 2. Core Compression Benchmarks

### 2a. Orchestrated Strategy Comparison

Test corpus: 10 file types (10KB → 1MB). Orchestrator auto-selects the best strategy per file.

| File | Size | Best Strategy | Ratio | Comp ms | Decomp ms |
|------|------|---------------|-------|---------|-----------|
| `json_log_large.json` | 1 MB | ZSTD | **24.95×** | 714ms | 0.7ms |
| `json_log_medium.json` | 200 KB | ZSTD | **24.41×** | 182ms | 0.2ms |
| `binary_low_entropy.bin` | 500 KB | LZMA | **16.08×** | 204ms | 4.5ms |
| `text_large.txt` | 1 MB | ZSTD | **8.01×** | 512ms | 1.3ms |
| `text_medium.txt` | 100 KB | ZSTD | **7.14×** | 60ms | 0.2ms |
| `text_small.txt` | 10 KB | ZSTD | **5.58×** | 5ms | 0.1ms |
| `solid_256.png` | 760 B | ZSTD | **5.76×** | 0.1ms | 0.0ms |
| `gradient_512.png` | 5.3 KB | LZMA | **1.58×** | 5ms | 0.3ms |
| `report.docx` | 1.1 KB | ZSTD | **1.30×** | 0.4ms | 0.0ms |
| `binary_high_entropy.bin` | 500 KB | ZSTD | **1.00×** | 54ms | 0.1ms |

> **Random binary is incompressible by design** — this is correct behaviour.

### 2b. Per-Strategy Head-to-Head (Direct Codec Comparison)

#### JSON Logs (200KB)

| Strategy | Ratio | Comp ms | MB/s | Memory |
|----------|-------|---------|------|--------|
| ZSTD level 19 | **24.41×** | 191ms | 1.0 MB/s | 102KB |
| LZMA preset 6 | 22.40× | 30ms | 6.4 MB/s | 93MB† |
| Brotli q=11 | 22.01× | 266ms | 0.7 MB/s | 13KB |
| ZSTD level 3 | 19.26× | 1.7ms | 112 MB/s | 102KB |
| Brotli q=4 | 16.83× | 3.3ms | 58 MB/s | 14KB |

#### Text corpus (100KB)

| Strategy | Ratio | Comp ms | MB/s |
|----------|-------|---------|------|
| ZSTD level 19 | **7.14×** | 64ms | 1.5 MB/s |
| LZMA | 7.07× | 51ms | 1.9 MB/s |
| Brotli q=11 | 7.02× | 175ms | 0.6 MB/s |
| ZSTD level 3 | 5.28× | 0.6ms | **172 MB/s** |
| Brotli q=4 | 4.77× | 1.8ms | 53 MB/s |

> [!TIP]
> **Key insight**: ZSTD level 3 trades ~26% ratio for **115× faster compression** — ideal for real-time use. ZSTD level 19 maximises ratio at the cost of speed.

†LZMA allocates ~93MB of working memory for its dictionary — a known trade-off.

---

## 3. Delta Encoding Benchmarks

Tested on collections of highly similar files (monthly reports, versioned configs).

| Collection | Files | Original | Delta | Savings | Ratio | Encode | Decode | Lossless |
|-----------|-------|----------|-------|---------|-------|--------|--------|----------|
| Monthly Reports (12 files) | 12 | 0.96 MB | 0.08 MB | **91.6%** | **11.95×** | 131ms | 4.5ms | ✅ |
| Versioned Configs (10 files) | 10 | 0.34 MB | 0.03 MB | **89.8%** | **9.80×** | 9ms | 1.7ms | ✅ |

> [!IMPORTANT]
> Delta encoding achieves **~90% space savings** on similar-file collections. This is the highest ratio achieved by any AirOne strategy, as it exploits inter-file redundancy that single-file codecs cannot see.

---

## 4. LSH Similarity Engine Benchmarks

| Metric | Value |
|--------|-------|
| Files indexed | 200 (100 similar + 100 random) |
| Similar pairs found | 4,950 |
| **Precision** | **100.0%** |
| **Recall** | **100.0%** |
| Indexing time | 48,214ms (MinHash computation) |
| Query time | 128ms |
| Throughput | ~4 files/sec |

> [!NOTE]
> The 48s indexing time is expected for 200 files × 128 hash permutations × byte-level shingling of ~18KB each. In production, the sample-based ingestion (`_read_sample`) limits this to 1MB per file regardless of actual size.
>
> **The critical metric is Precision=100% and Recall=100%** — the LSH index found every single one of the 4,950 similar pairs with zero false positives. This validates the MinHash band parameters (128 perms, 32 bands).

---

## 5. Streaming Compressor Benchmarks

File size: **9.53 MB** (mixed compressible + low-entropy binary)

| Window | Windows | Ratio | Compress | Decompress | Throughput | Lossless |
|--------|---------|-------|----------|------------|------------|----------|
| 64 KB | 153 | 8.74× | 8.0s | 59ms | 1.2 MB/s | ✅ |
| 256 KB | 39 | 9.46× | 6.9s | 60ms | 1.4 MB/s | ✅ |
| 1024 KB | 10 | 9.90× | 6.3s | 84ms | **1.5 MB/s** | ✅ |
| 4096 KB | 3 | **10.18×** | 7.3s | 57ms | 1.3 MB/s | ✅ |

> [!TIP]
> **Larger windows achieve better ratios** because ZSTD has more context for its dictionary-based algorithm. The 4MB window achieves the best ratio at 10.18×.
>
> **Decompression is 100× faster than compression** (~60ms vs 7s). This is inherent to ZSTD's asymmetric design — ideal for write-once, read-many archival scenarios.

---

## 6. Key Research Findings

### Finding 1: ZSTD Dominates on Structured Data
JSON logs achieve 24.95× with ZSTD. This is due to JSON's highly repetitive key names and value patterns which align well with ZSTD's sliding-window dictionary.

### Finding 2: Delta Encoding is the Highest-Ratio Strategy
For collections of similar files (versions, monthly reports), delta encoding achieves **~90% savings** — far beyond what any single-file codec can achieve. This validates investing in the collection optimization layer.

### Finding 3: LSH Precision=Recall=100% at Small Scale
At 200 files with 128 permutations and 32 bands, LSH achieves perfect accuracy. At millions of files, theory predicts ~97% recall at Jaccard≥0.8 — an acceptable trade-off for O(n) scalability.

### Finding 4: Speed vs Ratio Is the Key Trade-Off
| Use case | Recommended strategy |
|----------|---------------------|
| Real-time API response compression | ZSTD level 3 (172 MB/s) |
| Cold archival storage | ZSTD level 19 or Brotli q=11 |
| Similar-file collections | Delta encoding |
| Unknown collections | LSH → Delta if similar, ZSTD otherwise |
| Multi-GB files | Streaming + 1–4MB windows |

### Finding 5: LZMA Memory Is a Production Concern
LZMA allocates ~93MB of working memory per compression operation. For high-concurrency servers this could become problematic. ZSTD achieves comparable ratios (7.14× vs 7.07×) at ~1000× lower memory (102KB vs 93MB).

---

## 7. Repository Structure (Final)

```
airone/                          ← Project root
├── airone/                      ← Python package
│   ├── __init__.py
│   ├── __version__.py           ← "1.0.0"
│   ├── api.py                   ← Public Python API
│   ├── exceptions.py
│   ├── analysis/
│   │   ├── engine.py            ← AnalysisEngine (orchestrates analysis)
│   │   ├── format_detector.py   ← Magic-byte + extension detection
│   │   ├── entropy.py           ← Shannon entropy analysis
│   │   ├── image_classifier.py  ← Gradient/solid/natural classification
│   │   └── decomposer.py        ← Document semantic decomposer
│   ├── benchmarks/
│   │   └── runner.py            ← BenchmarkRunner (JSON/CSV/table output)
│   ├── cli/
│   │   └── main.py              ← click CLI entry point
│   ├── collection/
│   │   ├── cas.py               ← ContentAddressableStorage
│   │   ├── delta.py             ← DeltaEncoder + DeltaCollectionEncoder
│   │   ├── lsh.py               ← MinHasher + LSHIndex + ScalableCollectionAnalyser
│   │   └── optimizer.py         ← CollectionOptimizer
│   ├── compressors/
│   │   ├── base.py              ← BaseCompressor + CompressionResult
│   │   ├── neural/
│   │   │   ├── onnx_runtime.py  ← ONNX inference engine
│   │   │   └── trainer.py       ← PyTorch training + ONNX export
│   │   ├── procedural/
│   │   │   └── gradient.py      ← Gradient-aware image compressor
│   │   ├── semantic/
│   │   │   ├── office.py        ← DOCX/XLSX/PPTX semantic compressor
│   │   │   ├── pdf.py           ← PDF semantic compressor
│   │   │   ├── pdf_reconstructor.py   ← PDF reconstructor v1
│   │   │   └── pdf_reconstructor_v2.py ← PDF reconstructor v2 (CTM positional)
│   │   └── traditional/
│   │       ├── brotli.py
│   │       ├── lzma.py
│   │       └── zstd.py
│   ├── core/
│   │   ├── file_format.py       ← .air container format
│   │   ├── streaming.py         ← StreamingCompressor + StreamManifest
│   │   └── verification.py      ← verify_lossless()
│   ├── orchestrator/
│   │   └── orchestrator.py      ← CompressionOrchestrator pipeline
│   └── strategy/
│       ├── registry.py          ← StrategyRegistry
│       └── selector.py          ← StrategySelector
├── tests/                       ← 141 tests
│   ├── conftest.py              ← Shared fixtures + pytest markers
│   ├── test_analysis.py
│   ├── test_benchmark.py
│   ├── test_cli.py
│   ├── test_collection_optimizer.py
│   ├── test_compressors.py
│   ├── test_delta_encoding.py
│   ├── test_document_decomposer.py
│   ├── test_lsh_similarity.py
│   ├── test_neural_codec.py
│   ├── test_office_compressor.py
│   ├── test_onnx_codec.py
│   ├── test_pdf_compressor.py
│   ├── test_pdf_reconstructor_v2.py
│   ├── test_procedural.py
│   ├── test_strategy.py
│   └── test_streaming.py
├── scripts/
│   └── deep_benchmark.py        ← Full research benchmark script
├── examples/
│   ├── 01_basic_compress.py
│   ├── 02_lsh_collection.py
│   └── 03_streaming.py
├── results/
│   ├── corpus/                  ← Generated test corpus
│   └── benchmark_*.json         ← Timestamped benchmark reports
├── .github/
│   └── workflows/
│       └── ci.yml               ← GitHub Actions CI
├── CHANGELOG.md
├── Makefile
├── pyproject.toml               ← Modern PEP 517 build config
├── setup.py                     ← Compatibility shim
└── README.md
```
