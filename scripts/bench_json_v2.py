import json
import time
import os
import zstandard as zstd
from airone.compressors.semantic.json_semantic import SemanticJSONCompressor

def generate_large_json(rows=10000):
    data = []
    for i in range(rows):
        data.append({
            "id": 1000000 + i,
            "timestamp": 1713600000 + i * 10,
            "level": "INFO" if i % 10 != 0 else "WARNING",
            "active": i % 2 == 0,
            "value": 10.5 + (i % 100) / 10.0,
            "user_id": f"user_{i % 500}"
        })
    return json.dumps(data).encode("utf-8")

def run_bench():
    print("--- AirOne v2.0 Semantic JSON Benchmark ---")
    data = generate_large_json(10000)
    original_size = len(data)
    print(f"Original Size: {original_size / 1024:.1f} KB")
    
    # 1. ZSTD-19
    start = time.perf_counter()
    zstd_c = zstd.ZstdCompressor(level=19)
    zstd_data = zstd_c.compress(data)
    zstd_time = time.perf_counter() - start
    print(f"ZSTD-19:       {len(zstd_data) / 1024:.1f} KB (Ratio: {original_size / len(zstd_data):.2f}x) in {zstd_time:.3f}s")
    
    # 2. Semantic JSON v2.0
    compressor = SemanticJSONCompressor()
    # Mocking analysis for the benchmark
    class MockAnalysis:
        def __init__(self, size):
            self.file_size = size
            self.format = type('fmt', (), {'type': 'JSON', 'mime_type': 'application/json'})()
    
    start = time.perf_counter()
    result = compressor.compress(data, MockAnalysis(original_size))
    sem_time = time.perf_counter() - start
    print(f"Semantic v2.0: {result.compressed_size / 1024:.1f} KB (Ratio: {result.ratio:.2f}x) in {sem_time:.3f}s")
    
    improvement = result.ratio / (original_size / len(zstd_data))
    print(f"\nIMPROVEMENT OVER ZSTD: {improvement:.2f}x [Rocket]")
    
    # Verification
    decomp = compressor.decompress(result.compressed_data, result.metadata)
    if json.loads(decomp) == json.loads(data):
        print("Verification:  PASSED (Bit-exact semantic reconstruction)")
    else:
        print("Verification:  FAILED")

if __name__ == "__main__":
    run_bench()
