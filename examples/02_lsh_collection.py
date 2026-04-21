#!/usr/bin/env python3
"""
AirOne Example: LSH Collection Similarity
==========================================
Shows how to use the MinHash LSH engine to find
duplicate and near-duplicate files in a large collection.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from airone.collection.lsh import ScalableCollectionAnalyser


def main():
    # Build a synthetic collection with intentional duplicates
    base = b"Monthly financial report for department: "
    files = {}

    # 10 similar monthly reports
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
        content = base + (
            f"Month={month}. Revenue=12345. Costs=6789. Net=5556. "
            "Details follow. All figures in USD. Approved by CFO. "
        ).encode() * 200
        files[f"report_{month}.txt"] = content

    # 5 completely different files
    for i in range(5):
        files[f"random_{i}.bin"] = os.urandom(8192)

    print(f"Indexing {len(files)} files with MinHash LSH...\n")

    analyser = ScalableCollectionAnalyser(
        num_permutations=64,   # Lower for demo speed; use 128 for production
        num_bands=16,
    )
    analyser.ingest_bytes(files)

    # Find similar pairs
    pairs = analyser.find_similar_pairs(threshold=0.5)
    print(f"Similar pairs found: {len(pairs)}")
    for p in pairs[:5]:
        print(f"  {os.path.basename(p.file_a):<25} <-> {os.path.basename(p.file_b):<25} Jaccard={p.estimated_jaccard:.2f}")

    # Find clusters
    clusters = analyser.find_clusters(threshold=0.5)
    print(f"\nClusters found: {len(clusters)}")
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id}: {cluster.size} files, avg_sim={cluster.avg_similarity:.2f}")
        for m in cluster.members[:3]:
            print(f"    - {os.path.basename(m)}")
        if cluster.size > 3:
            print(f"    ... and {cluster.size - 3} more")


if __name__ == "__main__":
    main()
