"""
Tests for ONNX Neural Codec infrastructure.
Tests interface correctness and graceful failure without model weights.
"""

from __future__ import annotations

import pytest
import numpy as np

from airone.compressors.neural.onnx_runtime import (
    ArchitecturalONNXCodec,
    ImagePreprocessor,
    MedicalONNXCodec,
    SatelliteONNXCodec,
    UIScreenshotONNXCodec,
    find_model,
    models_available,
)
from airone.exceptions import CompressionError


class TestModelDiscovery:

    def test_find_model_returns_none_when_missing(self):
        """No model weights in test env — should return None."""
        result = find_model("medical", "encoder")
        assert result is None

    def test_models_available_false_without_weights(self):
        assert models_available("medical") is False
        assert models_available("ui_screenshot") is False

    def test_find_model_checks_all_paths(self, tmp_path, monkeypatch):
        """If a model file exists in the search path, it should be found."""
        import os
        monkeypatch.setenv("AIRONE_MODELS_DIR", str(tmp_path))

        # Create dummy model file
        model_file = tmp_path / "medical_encoder.onnx"
        model_file.write_bytes(b"dummy onnx model")

        # Re-import or reload is tricky with standard imports, 
        # but we can rely on the fact that the function reads the env var 
        # inside the list definition if we trigger a re-eval or just 
        # check if monkeypatch works for the search.
        
        from importlib import reload
        import airone.compressors.neural.onnx_runtime as rt
        # We need to reach inside to update the global search paths
        rt._MODEL_SEARCH_PATHS[0] = tmp_path

        result = rt.find_model("medical", "encoder")
        assert result is not None
        assert result.exists()


class TestImagePreprocessor:

    @pytest.fixture
    def sample_png_bytes(self, tmp_path):
        from PIL import Image
        import io
        img = Image.new("RGB", (64, 64), color=(128, 64, 192))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_to_tensor_shape(self, sample_png_bytes):
        preprocessor = ImagePreprocessor(target_size=(64, 64))
        tensor, shape, mode = preprocessor.to_tensor(sample_png_bytes)

        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_roundtrip_preserves_shape(self, sample_png_bytes):
        preprocessor = ImagePreprocessor(target_size=(64, 64))
        tensor, orig_shape, orig_mode = preprocessor.to_tensor(
            sample_png_bytes
        )
        arr = preprocessor.from_tensor(tensor, orig_shape, orig_mode)

        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.uint8

    def test_no_resize_preserves_dimensions(self, sample_png_bytes):
        """Without target_size, original dimensions are preserved."""
        preprocessor = ImagePreprocessor(target_size=None)
        tensor, orig_shape, _ = preprocessor.to_tensor(sample_png_bytes)

        assert orig_shape == (64, 64)


class TestONNXCodecInterface:

    def test_medical_codec_name(self):
        codec = MedicalONNXCodec()
        assert codec.name == "neural_medical"
        assert codec.domain == "medical"

    def test_ui_codec_name(self):
        codec = UIScreenshotONNXCodec()
        assert codec.name == "neural_ui_screenshot"

    def test_satellite_codec_name(self):
        codec = SatelliteONNXCodec()
        assert codec.name == "neural_satellite"

    def test_architectural_codec_name(self):
        codec = ArchitecturalONNXCodec()
        assert codec.name == "neural_architectural"

    def test_cannot_handle_without_models(self):
        """Without model weights, can_handle should return False."""
        codec = MedicalONNXCodec()
        assert codec.can_handle(None) is False

    def test_compress_raises_without_models(self):
        codec = MedicalONNXCodec()
        with pytest.raises(CompressionError, match="model weights not found"):
            codec.compress(b"fake image data")

    def test_estimate_ratio_positive(self):
        codec = MedicalONNXCodec()
        ratio = codec.estimate_ratio(None)
        assert ratio > 1.0

    def test_target_size_set(self):
        codec = MedicalONNXCodec()
        assert codec.target_size == (512, 512)

    def test_latent_dims_set(self):
        codec = MedicalONNXCodec()
        assert codec.latent_dims == 256
