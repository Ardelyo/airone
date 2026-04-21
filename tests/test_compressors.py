import pytest
from airone.compressors.traditional.zstd import ZstdCompressor
from airone.analysis.engine import AnalysisEngine

def test_zstd_compressor(tmp_path):
    compressor = ZstdCompressor()
    data = b"hello world" * 100
    
    test_file = tmp_path / "dummy_path.txt"
    test_file.write_bytes(data)
    
    analysis = AnalysisEngine().analyse(str(test_file))
    result = compressor.compress(data, analysis)
    
    assert result.original_size == 1100
    assert result.compressed_size < 1100
    
    decompressed = compressor.decompress(result.compressed_data, result.metadata)
    assert decompressed == data
