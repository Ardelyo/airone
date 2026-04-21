"""
Tests for Neural Codec infrastructure.
Validates interface correctness without requiring model weights.
"""

from __future__ import annotations

import pytest

from airone.compressors.neural.codec import (
    MedicalNeuralCodec,
    NeuralCodecSystem,
    SUPPORTED_DOMAINS,
    UIScreenshotNeuralCodec,
)
from airone.exceptions import CompressionError


class TestNeuralCodecSystem:

    def test_supported_domains_defined(self):
        assert "medical" in SUPPORTED_DOMAINS
        assert "ui_screenshot" in SUPPORTED_DOMAINS
        assert "satellite" in SUPPORTED_DOMAINS
        assert "architectural" in SUPPORTED_DOMAINS

    def test_get_medical_codec(self):
        codec = NeuralCodecSystem.get("medical")
        assert codec is not None
        assert codec.name == "neural_medical"

    def test_get_ui_codec(self):
        codec = NeuralCodecSystem.get("ui_screenshot")
        assert codec.name == "neural_ui_screenshot"

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            NeuralCodecSystem.get("nonexistent_domain")

    def test_available_domains_list(self):
        domains = NeuralCodecSystem.available_domains()
        assert isinstance(domains, list)
        assert len(domains) >= 2


class TestMedicalNeuralCodecInterface:

    def test_name_is_correct(self):
        codec = MedicalNeuralCodec()
        assert codec.name == "neural_medical"

    def test_domain_is_correct(self):
        codec = MedicalNeuralCodec()
        assert codec.domain == "medical"

    def test_estimate_ratio_positive(self):
        codec = MedicalNeuralCodec()
        ratio = codec.estimate_ratio(None)
        assert ratio > 1.0

    def test_compress_raises_not_implemented(self):
        """
        Phase 3: compress() should raise NotImplementedError
        because no model weights are loaded.
        """
        codec = MedicalNeuralCodec()
        with pytest.raises(NotImplementedError):
            codec.compress(b"fake image data")

    def test_domain_info_has_status(self):
        codec = MedicalNeuralCodec()
        info = codec.domain_info
        assert "status" in info
        assert "typical_ratio" in info


class TestUIScreenshotNeuralCodecInterface:

    def test_name_is_correct(self):
        codec = UIScreenshotNeuralCodec()
        assert codec.name == "neural_ui_screenshot"

    def test_compress_raises_not_implemented(self):
        codec = UIScreenshotNeuralCodec()
        with pytest.raises(NotImplementedError):
            codec.compress(b"fake screenshot data")
