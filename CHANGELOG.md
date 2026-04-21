# Changelog

All notable changes to AirOne are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2026-04-21 — Production Release

### Added
- **PDF Positional Reconstructor v2**: CTM-based pixel-perfect element placement
- **MinHash LSH Similarity Engine**: O(n) scaling to millions of files with 100% precision/recall on benchmarks
- **Office Document Semantic Compressor**: DOCX, XLSX, PPTX decomposition with Brotli XML + media deduplication
- **Streaming Compressor**: Window-based O(1)-memory compression with random-access decompression
- **Deep Benchmark Suite**: `scripts/deep_benchmark.py` — full corpus, delta, LSH, and streaming benchmarks
- **141 passing tests** across all architectural layers

### Changed
- `ZstdCompressor` now accepts `level` parameter at construction (default 19)
- Strategy registry `_build_default_registry` is now importable for testing

---

## [0.4.0] - 2026-04-20 — Phase 4: Neural Codecs

### Added
- ONNX Runtime inference engine for domain-specific neural codecs
- PyTorch training pipeline with ONNX export
- Delta encoding (`DeltaEncoder`, `DeltaCollectionEncoder`) using ZSTD patch-from
- Benchmark runner with JSON/CSV/terminal output
- Phase 4 test suite (95 tests)

---

## [0.3.0] - 2026-04-20 — Phase 3: Semantic Documents

### Added
- `PDFDecomposer`: page-by-page semantic decomposition
- `PDFSemanticCompressor`: component-aware PDF compression
- `ContentAddressableStorage`: SHA-256 block deduplication
- `SimilarityAnalyser`: O(n²) Jaccard similarity (replaced by LSH in v1.0)
- `CollectionOptimizer`: CAS + similarity for collection-level compression
- Brotli and LZMA traditional codecs
- Extended CLI `analyse` command

---

## [0.2.0] - 2026-04-19 — Phase 2: Intelligence Layer

### Added
- `FormatDetector`: magic-byte + extension file type identification
- `ImageClassifier`: gradient vs solid vs natural image classification
- `EntropyAnalyser`: Shannon entropy measurement for compressibility estimation
- `GradientCompressor`: procedural image codec
- `StrategySelector`: entropy + format-aware codec selection
- Phase 2 test suite

---

## [0.1.0] - 2026-04-19 — Phase 1: Foundation

### Added
- `.air` container format (msgpack header + zstd body)
- `ZstdCompressor` — universal baseline codec
- `CompressionOrchestrator` — end-to-end pipeline
- `verify_lossless` — round-trip fidelity checking
- Initial CLI (`compress`, `decompress`)
- Foundation test suite
