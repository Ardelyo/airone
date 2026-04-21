"""
AirOne Image Classifier
Classifies images by content type, domain, and generation method
using statistical analysis and heuristics.

Intentionally ML-free in Phase 2 — pure numerical/statistical approach.
Neural classification is a Phase 3 concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from PIL import Image, ImageStat


class ContentType(str, Enum):
    PHOTOGRAPH   = "photograph"
    SCREENSHOT   = "screenshot"
    DIAGRAM      = "diagram"
    LOGO         = "logo"
    CHART        = "chart"
    ILLUSTRATION = "illustration"
    TEXTURE      = "texture"
    GRADIENT     = "gradient"
    MIXED        = "mixed"
    UNKNOWN      = "unknown"


class ImageDomain(str, Enum):
    MEDICAL       = "medical"
    SATELLITE     = "satellite"
    ARCHITECTURAL = "architectural"
    UI            = "ui"
    ARTISTIC      = "artistic"
    GENERAL       = "general"


class GenerationMethod(str, Enum):
    NATURAL          = "natural"       # Real-world photo
    GRADIENT         = "gradient"      # Solid gradient
    FRACTAL          = "fractal"       # Fractal/mathematical
    VECTOR_RENDER    = "vector_render" # Rendered from vector
    SYNTHETIC        = "synthetic"     # Noise / procedural
    AI_GENERATED     = "ai_generated"  # AI image (Stable Diffusion etc.)
    SCREENSHOT       = "screenshot"    # OS/app screenshot


@dataclass
class ImageFeatures:
    """Raw feature measurements used for classification."""

    width: int = 0
    height: int = 0
    num_channels: int = 3

    unique_color_ratio: float = 0.0    # unique_colors / total_pixels
    palette_size: int = 0              # number of unique colours (capped sample)

    mean_brightness: float = 0.0
    brightness_std: float = 0.0

    edge_density: float = 0.0          # edges / total_pixels

    has_transparency: bool = False
    is_grayscale: bool = False

    # Spatial uniformity: how uniform are pixel values in sub-regions?
    spatial_uniformity: float = 0.0    # 0 = chaotic, 1 = perfectly uniform

    # Horizontal/vertical stripe presence (common in screenshots / UIs)
    horizontal_stripe_score: float = 0.0
    vertical_stripe_score: float = 0.0

    # Colour entropy
    color_entropy: float = 0.0


@dataclass
class ImageClassification:
    """Classification output."""
    content_type: ContentType
    domain: ImageDomain
    generation_method: GenerationMethod

    # 0-1 confidence for each top-level decision
    content_confidence: float = 0.0
    domain_confidence: float = 0.0
    generation_confidence: float = 0.0

    features: Optional[ImageFeatures] = None
    notes: list[str] = field(default_factory=list)


class ImageClassifier:
    """
    Classifies images using statistical feature extraction.
    No external ML models required.

    Usage::

        clf = ImageClassifier()
        result = clf.classify("photo.jpg")
        print(result.content_type)   # ContentType.PHOTOGRAPH
    """

    # Maximum pixels sampled when counting unique colours (performance guard)
    _MAX_SAMPLE_PIXELS = 50_000

    def classify(self, image_path: str) -> ImageClassification:
        image = self._load(image_path)
        features = self._extract_features(image)

        content_type, c_conf   = self._classify_content(features)
        domain,       d_conf   = self._classify_domain(features, content_type)
        gen_method,   g_conf   = self._classify_generation(features, content_type)

        return ImageClassification(
            content_type=content_type,
            domain=domain,
            generation_method=gen_method,
            content_confidence=c_conf,
            domain_confidence=d_conf,
            generation_confidence=g_conf,
            features=features,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load(self, path: str) -> Image.Image:
        img = Image.open(path)
        img.load()          # force decode
        return img

    def _extract_features(self, image: Image.Image) -> ImageFeatures:
        f = ImageFeatures()
        f.width, f.height = image.size
        f.has_transparency = image.mode in ("RGBA", "LA", "PA")
        f.is_grayscale     = image.mode in ("L", "LA")
        f.num_channels     = len(image.getbands())

        # Work in RGB(A) for consistent stats
        rgb = image.convert("RGB")

        # --- Colour stats ---
        stat = ImageStat.Stat(rgb)
        means = stat.mean          # per-channel means
        stddevs = stat.stddev

        f.mean_brightness = sum(means) / len(means)
        f.brightness_std  = sum(stddevs) / len(stddevs)

        # --- Unique colours (sampled) ---
        f.palette_size, f.unique_color_ratio = self._sample_unique_colors(rgb)

        # --- Colour entropy ---
        f.color_entropy = self._compute_color_entropy(rgb)

        # --- Edge density ---
        f.edge_density = self._compute_edge_density(rgb)

        # --- Spatial uniformity ---
        f.spatial_uniformity = self._compute_spatial_uniformity(rgb)

        # --- Stripe detection ---
        f.horizontal_stripe_score = self._detect_stripes(rgb, axis="horizontal")
        f.vertical_stripe_score   = self._detect_stripes(rgb, axis="vertical")

        return f

    def _sample_unique_colors(self, rgb_image: Image.Image) -> tuple[int, float]:
        pixels = list(rgb_image.getdata())
        total = len(pixels)

        if total > self._MAX_SAMPLE_PIXELS:
            step = total // self._MAX_SAMPLE_PIXELS
            pixels = pixels[::step]

        unique = len(set(pixels))
        ratio = unique / len(pixels) if pixels else 0.0
        return unique, ratio

    def _compute_color_entropy(self, rgb_image: Image.Image) -> float:
        """Shannon entropy over the flattened pixel values."""
        from collections import Counter
        pixels = list(rgb_image.getdata())
        if len(pixels) > self._MAX_SAMPLE_PIXELS:
            step = len(pixels) // self._MAX_SAMPLE_PIXELS
            pixels = pixels[::step]

        counts = Counter(pixels)
        total = len(pixels)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(p)
        return entropy

    def _compute_edge_density(self, rgb_image: Image.Image) -> float:
        """
        Approximates edge density using a simple horizontal-difference filter.
        Returns fraction of pixels considered 'edges'.
        """
        gray = rgb_image.convert("L")
        width, height = gray.size
        pixels = gray.load()

        edge_count = 0
        threshold = 20
        sample_rows = range(0, height, max(1, height // 100))

        for y in sample_rows:
            for x in range(width - 1):
                diff = abs(int(pixels[x, y]) - int(pixels[x + 1, y]))
                if diff > threshold:
                    edge_count += 1

        sampled_pixels = len(sample_rows) * (width - 1)
        return edge_count / sampled_pixels if sampled_pixels else 0.0

    def _compute_spatial_uniformity(self, rgb_image: Image.Image) -> float:
        """
        Divide image into 4×4 grid of blocks.
        Measure variance of per-block mean brightness.
        Low variance → high uniformity.
        """
        gray = rgb_image.convert("L")
        w, h = gray.size
        block_w, block_h = max(1, w // 4), max(1, h // 4)

        block_means = []
        for row in range(4):
            for col in range(4):
                box = (col * block_w, row * block_h,
                       min((col + 1) * block_w, w),
                       min((row + 1) * block_h, h))
                region = gray.crop(box)
                stat = ImageStat.Stat(region)
                block_means.append(stat.mean[0])

        if len(block_means) < 2:
            return 1.0

        mean_of_means = sum(block_means) / len(block_means)
        variance = sum((m - mean_of_means) ** 2 for m in block_means) / len(block_means)
        # Normalise: variance of 0 → 1.0, variance of 10000 → 0.0
        uniformity = 1.0 - min(1.0, variance / 10_000)
        return uniformity

    def _detect_stripes(self, rgb_image: Image.Image, axis: str) -> float:
        """
        Returns a score 0-1 indicating presence of horizontal or vertical stripes.
        High score is a strong indicator of UI / screenshot content.
        """
        gray = rgb_image.convert("L")
        w, h = gray.size
        pixels = gray.load()

        identical_lines = 0

        if axis == "horizontal":
            total_lines = min(h - 1, 200)
            step = max(1, h // total_lines)
            rows = range(0, h - 1, step)
            for y in rows:
                row_a = [pixels[x, y]     for x in range(0, w, max(1, w // 50))]
                row_b = [pixels[x, y + 1] for x in range(0, w, max(1, w // 50))]
                diff = sum(abs(a - b) for a, b in zip(row_a, row_b))
                if diff < len(row_a) * 5:   # very similar successive rows
                    identical_lines += 1
            return identical_lines / len(rows) if rows else 0.0

        else:  # vertical
            total_cols = min(w - 1, 200)
            step = max(1, w // total_cols)
            cols = range(0, w - 1, step)
            for x in cols:
                col_a = [pixels[x, y]     for y in range(0, h, max(1, h // 50))]
                col_b = [pixels[x + 1, y] for y in range(0, h, max(1, h // 50))]
                diff = sum(abs(a - b) for a, b in zip(col_a, col_b))
                if diff < len(col_a) * 5:
                    identical_lines += 1
            return identical_lines / len(cols) if cols else 0.0

    # ------------------------------------------------------------------
    # Classification rules
    # ------------------------------------------------------------------

    def _classify_content(self, f: ImageFeatures) -> tuple[ContentType, float]:
        """
        Rule-based content classification using extracted features.
        Returns (ContentType, confidence).
        """
        scores: dict[ContentType, float] = {t: 0.0 for t in ContentType}

        # --- GRADIENT ---
        # Gradients typically have a very high stripe score (horizontal or vertical) and very low edge density
        if (f.horizontal_stripe_score > 0.95 or f.vertical_stripe_score > 0.95) and f.edge_density < 0.05:
            return ContentType.GRADIENT, 1.0

        # --- SCREENSHOT ---
        # Stripe patterns, limited palette, sharp edges, transparency unlikely
        stripe_score = max(f.horizontal_stripe_score, f.vertical_stripe_score)
        if stripe_score > 0.4:
            scores[ContentType.SCREENSHOT] += 2.0 * stripe_score
        if f.unique_color_ratio < 0.05 and f.edge_density > 0.1:
            scores[ContentType.SCREENSHOT] += 1.5
        if not f.has_transparency and f.color_entropy < 12:
            scores[ContentType.SCREENSHOT] += 0.5

        # --- LOGO / ICON ---
        # Very few colours, has transparency, small dimensions
        small = (f.width <= 512 and f.height <= 512)
        if f.palette_size < 256 and f.has_transparency and small:
            scores[ContentType.LOGO] += 2.5
        elif f.palette_size < 64:
            scores[ContentType.LOGO] += 1.0

        # --- DIAGRAM / CHART ---
        # Limited palette, high edge density (from lines), no transparency
        if f.palette_size < 512 and f.edge_density > 0.15:
            scores[ContentType.DIAGRAM] += 1.5
        if f.unique_color_ratio < 0.02 and f.edge_density > 0.2:
            scores[ContentType.CHART] += 1.5

        # --- PHOTOGRAPH ---
        # High unique colour ratio, high entropy, moderate edge density
        if f.unique_color_ratio > 0.30 and f.color_entropy > 15:
            scores[ContentType.PHOTOGRAPH] += 3.0
        if f.brightness_std > 40:
            scores[ContentType.PHOTOGRAPH] += 1.0

        # --- TEXTURE ---
        # High uniformity + high entropy (structured repetition)
        if f.spatial_uniformity > 0.7 and f.color_entropy > 10:
            scores[ContentType.TEXTURE] += 2.0

        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score == 0.0:
            return ContentType.UNKNOWN, 0.0

        # Normalise to rough confidence [0, 1]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0
        return best, min(1.0, confidence)

    def _classify_domain(
        self,
        f: ImageFeatures,
        content_type: ContentType,
    ) -> tuple[ImageDomain, float]:
        """
        Identify specialised domain.
        Currently heuristic; Phase 3 will replace with a trained model.
        """
        # UI domain — Screenshots with stripe patterns
        if content_type == ContentType.SCREENSHOT:
            stripe = max(f.horizontal_stripe_score, f.vertical_stripe_score)
            return ImageDomain.UI, min(1.0, stripe * 1.5)

        # Grayscale → possible medical or satellite SAR
        if f.is_grayscale and f.unique_color_ratio > 0.1:
            return ImageDomain.MEDICAL, 0.45  # low confidence; needs neural model

        # Very large images with high entropy can be satellite
        if f.width > 2000 and f.height > 2000 and f.color_entropy > 16:
            return ImageDomain.SATELLITE, 0.35

        return ImageDomain.GENERAL, 0.90

    def _classify_generation(
        self,
        f: ImageFeatures,
        content_type: ContentType,
    ) -> tuple[GenerationMethod, float]:
        """
        Determine how the image was likely produced.
        """
        if content_type == ContentType.GRADIENT:
            return GenerationMethod.GRADIENT, 0.90

        if content_type == ContentType.SCREENSHOT:
            return GenerationMethod.SCREENSHOT, 0.88

        if content_type == ContentType.LOGO and f.has_transparency:
            return GenerationMethod.VECTOR_RENDER, 0.75

        if f.unique_color_ratio > 0.40 and f.color_entropy > 16:
            return GenerationMethod.NATURAL, 0.85

        if f.unique_color_ratio < 0.001 and f.spatial_uniformity > 0.85:
            return GenerationMethod.SYNTHETIC, 0.70

        return GenerationMethod.NATURAL, 0.55
