import pytest
import tempfile
import os

from scripts.deep_benchmark import CORPUS_DIR, gen_gradient_png
from airone.orchestrator.orchestrator import CompressionOrchestrator
from airone.compressors.procedural.gradient import GradientCompressor

def test_gradient_routing():
    """Verify that a gradient image routes to procedural_gradient correctly."""
    gradient_path = CORPUS_DIR / "gradient_4096.png"
    if not gradient_path.exists():
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        gradient_path.write_bytes(gen_gradient_png(width=4096, height=4096))

    orch = CompressionOrchestrator()
    analysis = orch.analyse_file(str(gradient_path))
    
    # Verify the image classification identifies it as a gradient
    from airone.analysis.image_classifier import ContentType
    assert analysis.image_classification is not None
    assert analysis.image_classification.content_type == ContentType.GRADIENT
    assert analysis.image_classification.content_confidence > 0.70

    # Ensure it's handled by compressor correctly
    compressor = GradientCompressor()
    assert compressor.can_handle(analysis) is True

    # Compress end to end and assert the strategy names
    with tempfile.NamedTemporaryFile(suffix=".air", delete=False) as f:
        out_path = f.name
        
    try:
        result = orch.compress_file(str(gradient_path), out_path)
        assert result.strategy_name == "procedural_gradient"
        # On a large image: gradient_params are ~200 bytes vs MB of raw pixels
        assert result.ratio > 200.0
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
