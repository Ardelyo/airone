"""
Tests for the Benchmark Runner.
"""

from __future__ import annotations

import os
import pytest

from airone.benchmarks.runner import BenchmarkRunner, BenchmarkReport


@pytest.fixture
def sample_files(tmp_path):
    """Create a variety of test files."""
    files = {}

    # Text file
    text_path = tmp_path / "document.txt"
    text_path.write_text("AirOne benchmark test. " * 500)
    files["text"] = str(text_path)

    # Binary file (repetitive)
    binary_path = tmp_path / "data.bin"
    binary_path.write_bytes(b"\x00\x01\x02\x03" * 2048)
    files["binary"] = str(binary_path)

    # Gradient PNG
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        intensity = int(x / 99 * 255)
        for y in range(100):
            img.putpixel((x, y), (intensity, intensity, intensity))
    png_path = tmp_path / "gradient.png"
    img.save(png_path, format="PNG")
    files["png"] = str(png_path)

    return files


class TestBenchmarkRunner:

    def test_run_produces_records(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            list(sample_files.values()),
            strategies=["traditional_zstd"],
        )

        assert len(report.records) > 0

    def test_all_records_have_strategy(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd", "traditional_lzma"],
        )

        strategies = {r.strategy for r in report.records}
        assert "traditional_zstd" in strategies

    def test_lossless_verified(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd"],
        )

        successful = [r for r in report.records if not r.error]
        assert all(r.lossless_verified for r in successful)

    def test_best_per_file(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd", "traditional_lzma"],
        )

        best = report.best_per_file()
        assert len(best) >= 1

    def test_to_json(self, sample_files):
        import json
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd"],
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "records" in parsed
        assert "strategy_summary" in parsed

    def test_to_csv(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd"],
        )

        csv_str = report.to_csv()
        assert "traditional_zstd" in csv_str
        assert "file_name" in csv_str

    def test_strategy_summary(self, sample_files):
        runner = BenchmarkRunner()
        report = runner.run(
            [sample_files["text"]],
            strategies=["traditional_zstd"],
        )

        summary = report.strategy_summary()
        assert "traditional_zstd" in summary
        assert summary["traditional_zstd"]["avg_ratio"] > 1.0
