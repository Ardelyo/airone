"""Tests for the Phase 2 Analysis Engine."""

import os
import pytest
from PIL import Image
from airone.analysis.engine import AnalysisEngine
from airone.analysis.format_detector import FileCategory, FormatDetector
from airone.analysis.entropy import EntropyAnalyser
from airone.analysis.image_classifier import (
    ImageClassifier, ContentType, GenerationMethod,
)


@pytest.fixture
def sample_png(tmp_path):
    """A simple 100×100 red PNG."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = tmp_path / "sample.png"
    img.save(path, format="PNG")
    return str(path)


@pytest.fixture
def gradient_png(tmp_path):
    """A 200×200 horizontal gradient PNG (black → white)."""
    img = Image.new("RGB", (200, 200))
    for x in range(200):
        intensity = round(x / 199 * 255)
        for y in range(200):
            img.putpixel((x, y), (intensity, intensity, intensity))
    path = tmp_path / "gradient.png"
    img.save(path, format="PNG")
    return str(path)


@pytest.fixture
def sample_text(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello AirOne! " * 100)
    return str(path)


# -----------------------------------------------------------------------
# FormatDetector
# -----------------------------------------------------------------------

class TestFormatDetector:

    def test_detects_png(self, sample_png):
        detector = FormatDetector()
        fmt = detector.detect(sample_png)
        assert fmt.type == "PNG"
        assert fmt.category == FileCategory.IMAGE
        assert fmt.is_image is True

    def test_detects_text_by_extension(self, sample_text):
        detector = FormatDetector()
        fmt = detector.detect(sample_text)
        assert fmt.type == "TXT"

    def test_raises_on_missing_file(self):
        from airone.exceptions import FormatError
        detector = FormatDetector()
        with pytest.raises(FormatError):
            detector.detect("/does/not/exist.abc")


# -----------------------------------------------------------------------
# EntropyAnalyser
# -----------------------------------------------------------------------

class TestEntropyAnalyser:

    def test_solid_colour_has_low_entropy(self, sample_png):
        """Solid red image should have very low entropy."""
        analyser = EntropyAnalyser()
        report = analyser.analyse(sample_png)
        assert report.global_entropy < 4.0

    def test_text_has_moderate_entropy(self, sample_text):
        analyser = EntropyAnalyser()
        report = analyser.analyse(sample_text)
        assert 2.0 < report.global_entropy < 7.0

    def test_high_entropy_low_compressibility(self, tmp_path):
        """Random bytes → entropy ≈ 8."""
        import os
        random_file = tmp_path / "random.bin"
        random_file.write_bytes(os.urandom(64 * 1024))
        analyser = EntropyAnalyser()
        report = analyser.analyse(str(random_file))
        assert report.global_entropy > 7.5
        assert report.is_random is True


# -----------------------------------------------------------------------
# ImageClassifier
# -----------------------------------------------------------------------

class TestImageClassifier:

    def test_gradient_detected(self, gradient_png):
        clf = ImageClassifier()
        result = clf.classify(gradient_png)
        assert result.content_type == ContentType.GRADIENT
        assert result.content_confidence >= 0.4

    def test_solid_colour_classified(self, sample_png):
        clf = ImageClassifier()
        result = clf.classify(sample_png)
        # Solid red = essentially a gradient/texture — not a photograph
        assert result.content_type != ContentType.PHOTOGRAPH


# -----------------------------------------------------------------------
# AnalysisEngine (integration)
# -----------------------------------------------------------------------

class TestAnalysisEngine:

    def test_analyse_png(self, sample_png):
        engine = AnalysisEngine()
        report = engine.analyse(sample_png)
        assert report.file_size > 0
        assert report.format.type == "PNG"
        assert report.format.is_image is True
        assert report.image_classification is not None
        assert "traditional_zstd" in report.strategy_hints

    def test_analyse_text(self, sample_text):
        engine = AnalysisEngine()
        report = engine.analyse(sample_text)
        assert report.format.type == "TXT"
        assert report.image_classification is None

    def test_summary_returns_string(self, sample_png):
        engine = AnalysisEngine()
        report = engine.analyse(sample_png)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "PNG" in summary

    def test_gradient_hint_generated(self, gradient_png):
        engine = AnalysisEngine()
        report = engine.analyse(gradient_png)
        assert "procedural_gradient" in report.strategy_hints
