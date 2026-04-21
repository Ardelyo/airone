#!/usr/bin/env python3
"""
AirOne Utility: Visual Comparison Benchmarking
==============================================
Demonstrates real-world fidelity by:
1. Compressing a high-res PNG -> .air
2. Decompressing back to PNG
3. Calculating PSNR/SSIM and file size ratios.
"""
import sys
import os
import math
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from airone.orchestrator.orchestrator import CompressionOrchestrator

SAMPLES = [
    {
        "input": "results/visual_benchmark/original.png",
        "air": "results/visual_benchmark/photo.air",
        "output": "results/visual_benchmark/photo_restored.png",
        "desc": "High-Resolution Photograph (5:4 Aspect)"
    },
    {
        "input": "results/visual_benchmark/gradient.png",
        "air": "results/visual_benchmark/gradient.air",
        "output": "results/visual_benchmark/gradient_restored.png",
        "desc": "4K Linear Gradient (Procedural Strategy)"
    }
]

def calculate_psnr(img1, img2):
    i1 = np.array(img1).astype(np.float64)
    i2 = np.array(img2).astype(np.float64)
    mse = np.mean((i1 - i2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    COMPRESSED_DIR = Path("results/visual_benchmark")
    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

    orch = CompressionOrchestrator()
    print(f"--- AirOne Visual Benchmark Suite ---")

    for sample in SAMPLES:
        input_path = Path(sample["input"])
        air_path = Path(sample["air"])
        output_path = Path(sample["output"])

        if not input_path.exists():
            print(f"Skipping {sample['desc']}: File not found.")
            continue

        print(f"\nProcessing: {sample['desc']}")
        
        # 1. Compress
        t0 = time.perf_counter()
        result = orch.compress_file(str(input_path), str(air_path))
        t1 = time.perf_counter()
        
        comp_size = os.path.getsize(air_path)
        orig_size = os.path.getsize(input_path)
        
        print(f"  Step 1: Compression Complete")
        print(f"    Original Size   : {orig_size:,} bytes")
        print(f"    Compressed Size : {comp_size:,} bytes")
        print(f"    Ratio           : {result.ratio:.2f}x")
        print(f"    Time            : {(t1-t0)*1000:.1f} ms")
        print(f"    Strategy        : {result.strategy_name}")
        
        # 2. Decompress
        t0 = time.perf_counter()
        orch.decompress_file(str(air_path), str(output_path))
        t1 = time.perf_counter()
        
        print(f"  Step 2: Decompression Complete")
        print(f"    Time            : {(t1-t0)*1000:.1f} ms")
        
        # 3. Visual Comparison
        img_orig = Image.open(input_path).convert("RGB")
        img_rest = Image.open(output_path).convert("RGB")
        
        psnr = calculate_psnr(img_orig, img_rest)
        
        print(f"  --- Fidelity Metrics ---")
        if psnr == float('inf'):
            print(f"    Visual Quality  : PIXEL-PERFECT (Infinity PSNR)")
        else:
            print(f"    PSNR            : {psnr:.2f} dB")
    
    # Check SSIM if available (optional)
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to gray for ssim
        gray_orig = np.array(img_orig.convert("L"))
        gray_rest = np.array(img_rest.convert("L"))
        score, _ = ssim(gray_orig, gray_rest, full=True)
        print(f"  SSIM            : {score:.5f}")
    except ImportError:
        print("  SSIM            : Skip (scikit-image not installed)")

    print(f"\nBenchmark finished. Files generated in results/visual_benchmark/")

if __name__ == "__main__":
    main()
