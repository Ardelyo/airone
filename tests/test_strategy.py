"""Tests for Strategy Selector and Registry."""

import pytest
from airone.strategy.registry import StrategyRegistry
from airone.strategy.selector import StrategySelector
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.compressors.procedural.gradient import GradientCompressor
from airone.exceptions import StrategyError


@pytest.fixture
def registry():
    r = StrategyRegistry()
    r.register(ZstdCompressor())
    r.register(GradientCompressor())
    return r


@pytest.fixture
def selector(registry):
    return StrategySelector(registry)


class TestStrategyRegistry:

    def test_register_and_get(self, registry):
        compressor = registry.get("traditional_zstd")
        assert compressor is not None
        assert compressor.name == "traditional_zstd"

    def test_list_names(self, registry):
        names = registry.list_names()
        assert "traditional_zstd" in names
        assert "procedural_gradient" in names

    def test_missing_strategy_raises(self, registry):
        with pytest.raises(StrategyError):
            registry.get("nonexistent_strategy")


class TestStrategySelector:

    def _make_basic_report(self):
        """Minimal analysis report for testing."""
        from unittest.mock import MagicMock
        from airone.analysis.format_detector import FileCategory, FileFormat
        from airone.analysis.entropy import EntropyReport

        report = MagicMock()
        report.file_size = 1024 * 1024
        report.is_image = False
        report.is_document = False
        report.format = FileFormat(
            type="TXT",
            mime_type="text/plain",
            category=FileCategory.TEXT,
        )
        report.entropy = EntropyReport(
            global_entropy=5.0,
            block_entropies=[5.0],
            min_block_entropy=5.0,
            max_block_entropy=5.0,
            mean_block_entropy=5.0,
            compressibility_estimate=2.0,
        )
        report.image_classification = None
        return report

    def test_fallback_always_zstd(self, selector):
        report = self._make_basic_report()
        candidates = selector.select(report)
        names = [c.strategy_name for c in candidates]
        assert "traditional_zstd" in names

    def test_candidates_are_ranked(self, selector):
        report = self._make_basic_report()
        candidates = selector.select(report)
        priorities = [c.priority for c in candidates]
        assert priorities == sorted(priorities)
