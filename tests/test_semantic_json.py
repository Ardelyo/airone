import json
import pytest
from airone.compressors.semantic.json_semantic import SemanticJSONCompressor
from airone.analysis.engine import AnalysisReport
from airone.analysis.format_detector import FileFormat, FileCategory

@pytest.fixture
def compressor():
    return SemanticJSONCompressor()

@pytest.fixture
def mock_json_analysis():
    class MockAnalysis:
        def __init__(self):
            self.format = FileFormat("JSON", "application/json", FileCategory.DATA)
            self.file_size = 10000
    return MockAnalysis()

def test_json_semantic_can_handle(compressor, mock_json_analysis):
    assert compressor.can_handle(mock_json_analysis) is True

def test_json_semantic_roundtrip(compressor, mock_json_analysis):
    data = [
        {"id": i, "name": f"user_{i}", "age": 20 + (i % 50), "active": i % 2 == 0}
        for i in range(100)
    ]
    json_bytes = json.dumps(data).encode("utf-8")
    
    # Compress
    result = compressor.compress(json_bytes, mock_json_analysis)
    assert result.compressed_size < len(json_bytes)
    
    # Decompress
    decompressed = compressor.decompress(result.compressed_data, result.metadata)
    decoded = json.loads(decompressed.decode("utf-8"))
    
    # Note: Columnar reconstruction might change key order in some JSON libs, 
    # but here we use a list of keys to maintain order if possible.
    assert len(decoded) == len(data)
    for orig, res in zip(data, decoded):
        assert orig == res

def test_json_semantic_null_handling(compressor, mock_json_analysis):
    data = [
        {"id": 1, "note": "hello"},
        {"id": 2, "note": None},
        {"id": 3, "note": "world"}
    ]
    # Need at least _MIN_ROWS (10) and _MIN_SIZE (2KB) for the compressor to accept it
    data = data * 50 
    json_bytes = json.dumps(data).encode("utf-8")
    
    result = compressor.compress(json_bytes, mock_json_analysis)
    decompressed = compressor.decompress(result.compressed_data, result.metadata)
    decoded = json.loads(decompressed.decode("utf-8"))
    
    assert decoded[1]["note"] is None
    assert decoded[4]["note"] is None

def test_json_semantic_mixed_types_fallback(compressor, mock_json_analysis):
    # If it's too irregular, it should ideally still handle it as strings or error out
    # if it doesn't meet the uniformity threshold.
    data = [{"a": 1}] * 5 + [{"b": 2}] * 5
    json_bytes = json.dumps(data).encode("utf-8")
    
    # This might fail or fallback depending on _infer_schema return
    try:
        result = compressor.compress(json_bytes, mock_json_analysis)
        decompressed = compressor.decompress(result.compressed_data, result.metadata)
        decoded = json.loads(decompressed.decode("utf-8"))
        assert len(decoded) == 10
    except Exception as e:
        # If it raises a CompressionError for lack of uniformity, that's also valid
        from airone.exceptions import CompressionError
        assert isinstance(e, CompressionError)
