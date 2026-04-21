import pytest
import os
import tempfile
from unittest.mock import MagicMock

from airone.orchestrator.orchestrator import CompressionOrchestrator, AirOneConfig
from airone.analysis.engine import AnalysisReport
from airone.strategy.registry import StrategyRegistry
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.compressors.traditional.lzma import LZMACompressor

@pytest.fixture
def memory_test_registry():
    r = StrategyRegistry()
    r.register(ZstdCompressor())
    r.register(LZMACompressor())
    return r

def test_orchestrator_skips_lzma_on_low_memory_budget(memory_test_registry):
    """
    Verify that if the budget is lower than LZMA requirements (100MB),
    the orchestrator skips it and falls back to ZSTD.
    """
    # Create a TINY budget
    config = AirOneConfig(memory_budget_mb=50) # LZMA requires 100
    orch = CompressionOrchestrator(registry=memory_test_registry, config=config)
    
    # Mock analysis and selector to prefer LZMA
    from airone.strategy.selector import StrategyCandidate
    orch._selector.select = MagicMock(return_value=[
        StrategyCandidate("traditional_lzma", 10.0, 1, "Preferred but heavy"),
        StrategyCandidate("traditional_zstd", 5.0, 10, "Fallback")
    ])
    orch._analyser.analyse = MagicMock(return_value=MagicMock(spec=AnalysisReport))
    
    # Create a dummy file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello world" * 1000)
        in_path = f.name
        
    with tempfile.NamedTemporaryFile(suffix=".air", delete=False) as f:
        out_path = f.name
        
    try:
        result = orch.compress_file(in_path, out_path)
        # Should have skipped LZMA (50 < 100) and picked ZSTD
        assert result.strategy_name == "traditional_zstd"
    finally:
        os.remove(in_path)
        os.remove(out_path)

def test_orchestrator_allows_lzma_on_high_memory_budget(memory_test_registry):
    """
    Verify that if the budget is high, LZMA is allowed.
    """
    # Create a LARGE budget
    config = AirOneConfig(memory_budget_mb=1024)
    orch = CompressionOrchestrator(registry=memory_test_registry, config=config)
    
    # Mock to prefer LZMA
    from airone.strategy.selector import StrategyCandidate
    orch._selector.select = MagicMock(return_value=[
        StrategyCandidate("traditional_lzma", 10.0, 1, "Preferred"),
        StrategyCandidate("traditional_zstd", 5.0, 10, "Fallback")
    ])
    orch._analyser.analyse = MagicMock(return_value=MagicMock(spec=AnalysisReport))
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello world" * 1000)
        in_path = f.name
        
    with tempfile.NamedTemporaryFile(suffix=".air", delete=False) as f:
        out_path = f.name
        
    try:
        result = orch.compress_file(in_path, out_path)
        # Should have picked LZMA (if physical memory also allows)
        # We assume the test machine has > 100MB free.
        assert result.strategy_name == "traditional_lzma"
    finally:
        os.remove(in_path)
        os.remove(out_path)
