#!/usr/bin/env python3
"""
AirOne Utility: Output Compressed Corpus
========================================
Compresses the benchmark test corpus and saves the .air files 
to disk for inspection.
"""
import sys
import os
from pathlib import Path

# Fix python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airone.orchestrator.orchestrator import CompressionOrchestrator

CORPUS_DIR     = Path("results/corpus")
COMPRESSED_DIR = Path("results/compressed_corpus")

def main():
    if not CORPUS_DIR.exists():
        print(f"Error: Corpus directory {CORPUS_DIR} not found. Run benchmark first.")
        return

    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    orch = CompressionOrchestrator()
    
    print(f"Compressing corpus to {COMPRESSED_DIR}...")
    
    for file_path in CORPUS_DIR.glob("*"):
        if file_path.is_dir():
            continue
            
        output_path = COMPRESSED_DIR / (file_path.name + ".air")
        
        print(f"  {file_path.name:<25} -> {output_path.name}")
        try:
            result = orch.compress_file(str(file_path), str(output_path))
            print(f"    {result.original_size:,} -> {result.compressed_size:,} bytes ({result.ratio:.2f}x)")
        except Exception as exc:
            print(f"    FAILED: {exc}")

    print("\nDone. You can find the compressed files in results/compressed_corpus/")

if __name__ == "__main__":
    main()
