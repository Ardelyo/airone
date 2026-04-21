"""
AirOne Neural Codec Infrastructure - Phase 3 Scaffolding

Defines the full interface for domain-specific neural compression codecs.
No model weights are loaded in Phase 3 — all methods raise
NotImplementedError with clear guidance.

Phase 4 will:
    1. Integrate ONNX Runtime for inference (no PyTorch required at runtime)
    2. Ship pre-trained encoder/decoder weights for 'medical' and 'ui' domains
    3. Implement the full encode → quantise → residual → compress pipeline

Architecture per domain
-----------------------
    Encoder  : Image  →  Latent  (learned compact representation)
    Decoder  : Latent →  Image   (approximate reconstruction)
    Residual : Original - Decoded (makes reconstruction exact / lossless)
    Codec    : Compress residual with ZSTD (typically very sparse)

The lossless guarantee comes entirely from storing the residual.
The neural network improves compression by making the residual small.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from airone.compressors.base import BaseCompressor, CompressionResult
from airone.exceptions import CompressionError, DecompressionError


# ---------------------------------------------------------------------------
# Supported domains
# ---------------------------------------------------------------------------

SUPPORTED_DOMAINS = {
    "medical": {
        "description": "CT scans, MRI, X-Ray, Ultrasound (DICOM)",
        "typical_ratio": 35.0,
        "input_modes": ["L", "RGB"],
        "status": "Phase 4",
    },
    "satellite": {
        "description": "Optical, SAR, Multispectral imagery",
        "typical_ratio": 45.0,
        "input_modes": ["RGB", "RGBA"],
        "status": "Phase 4",
    },
    "ui_screenshot": {
        "description": "Web, iOS, Android, Desktop screenshots",
        "typical_ratio": 28.0,
        "input_modes": ["RGB", "RGBA"],
        "status": "Phase 4",
    },
    "architectural": {
        "description": "Building plans, technical drawings",
        "typical_ratio": 80.0,
        "input_modes": ["L", "RGB"],
        "status": "Phase 4",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatentRepresentation:
    """
    The compressed intermediate state produced by the encoder.
    Stored alongside the residual in the .air container.
    """
    domain: str
    shape: tuple          # Original image shape (H, W, C)
    dtype: str            # Original dtype, e.g. "uint8"

    # Quantised latent vector — compact but not yet lossless
    latent_bytes: bytes = b""

    # Compressed residual (original − decoded) — makes it lossless
    residual_bytes: bytes = b""

    # Model version used for encoding
    # Must match decoder version exactly
    model_version: str = "0.0.0"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class NeuralCodecBase(BaseCompressor):
    """
    Abstract base for all neural compression codecs.

    Subclasses must implement:
        _encode(image_array) → latent
        _decode(latent)      → image_array
        _load_models()       → None

    The lossless pipeline (residual computation, compression,
    verification) is handled here in the base class and must
    NOT be overridden.
    """

    def __init__(self, domain: str) -> None:
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Supported: {list(SUPPORTED_DOMAINS.keys())}"
            )
        self.domain = domain
        self._models_loaded = False

    @property
    def name(self) -> str:
        return f"neural_{self.domain}"

    @property
    def domain_info(self) -> dict:
        return SUPPORTED_DOMAINS[self.domain]

    def can_handle(self, analysis) -> bool:
        if analysis is None:
            return False
        ic = getattr(analysis, "image_classification", None)
        if ic is None:
            return False

        from airone.analysis.image_classifier import ImageDomain
        domain_map = {
            "medical":        ImageDomain.MEDICAL,
            "satellite":      ImageDomain.SATELLITE,
            "ui_screenshot":  ImageDomain.UI,
            "architectural":  ImageDomain.ARCHITECTURAL,
        }
        expected = domain_map.get(self.domain)
        return ic.domain == expected and ic.domain_confidence > 0.60

    def estimate_ratio(self, analysis) -> float:
        return self.domain_info["typical_ratio"]

    def compress(self, data: bytes, analysis=None) -> CompressionResult:
        self._ensure_models_loaded()
        start = time.perf_counter()

        import numpy as np
        from PIL import Image
        import io

        # Load image
        image = Image.open(io.BytesIO(data))
        image.load()
        original_array = np.array(image)

        # Encode → latent
        latent = self._encode(original_array)

        # Decode latent → approximate reconstruction
        reconstructed_array = self._decode(latent)

        # Compute exact residual for lossless guarantee
        residual = original_array.astype(np.int16) - reconstructed_array.astype(np.int16)

        # Compress residual (expected to be sparse / near-zero)
        from airone.compressors.traditional.zstd import ZstdCompressor
        zstd = ZstdCompressor()
        residual_result = zstd.compress(residual.tobytes())

        # Pack latent representation
        latent_repr = LatentRepresentation(
            domain=self.domain,
            shape=original_array.shape,
            dtype=str(original_array.dtype),
            latent_bytes=self._serialise_latent(latent),
            residual_bytes=residual_result.compressed_data,
            model_version=self._model_version(),
        )

        import msgpack
        payload = msgpack.packb(
            {
                "domain":         latent_repr.domain,
                "shape":          list(latent_repr.shape),
                "dtype":          latent_repr.dtype,
                "latent":         latent_repr.latent_bytes,
                "residual":       latent_repr.residual_bytes,
                "model_version":  latent_repr.model_version,
            },
            use_bin_type=True,
        )

        return CompressionResult(
            compressed_data=payload,
            original_size=len(data),
            compressed_size=len(payload),
            strategy_name=self.name,
            execution_time=time.perf_counter() - start,
            metadata={"domain": self.domain},
        )

    def decompress(self, compressed_data: bytes, metadata: dict) -> bytes:
        self._ensure_models_loaded()

        import msgpack
        import numpy as np
        from PIL import Image
        import io

        bundle = msgpack.unpackb(compressed_data, raw=False)
        shape  = tuple(bundle["shape"])
        dtype  = bundle["dtype"]

        # Decode latent → approximate image
        latent = self._deserialise_latent(bundle["latent"])
        reconstructed = self._decode(latent)

        # Restore exact original via residual
        from airone.compressors.traditional.zstd import ZstdCompressor
        zstd = ZstdCompressor()
        residual_raw = zstd.decompress(bundle["residual"], {})
        residual = np.frombuffer(residual_raw, dtype=np.int16).reshape(shape)

        original = (reconstructed.astype(np.int16) + residual).astype(dtype)

        # Encode as PNG for return
        image = Image.fromarray(original)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_models(self) -> None:
        """Load encoder and decoder model weights."""

    @abstractmethod
    def _encode(self, image_array) -> object:
        """Encode image_array → latent representation."""

    @abstractmethod
    def _decode(self, latent) -> object:
        """Decode latent → approximate image_array."""

    @abstractmethod
    def _serialise_latent(self, latent) -> bytes:
        """Serialise latent to bytes for storage."""

    @abstractmethod
    def _deserialise_latent(self, data: bytes) -> object:
        """Deserialise stored bytes → latent."""

    @abstractmethod
    def _model_version(self) -> str:
        """Return the version string of the loaded model."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_models_loaded(self) -> None:
        if not self._models_loaded:
            self._load_models()
            self._models_loaded = True


# ---------------------------------------------------------------------------
# Concrete stub: Medical domain
# ---------------------------------------------------------------------------

class MedicalNeuralCodec(NeuralCodecBase):
    """
    Neural codec for medical imaging (CT, MRI, X-Ray).
    Phase 3: Raises NotImplementedError — weights not yet available.
    Phase 4: Will load ONNX models trained on medical imaging datasets.
    """

    def __init__(self) -> None:
        super().__init__("medical")

    def _load_models(self) -> None:
        raise NotImplementedError(
            "MedicalNeuralCodec: model weights are not available yet. "
            "This codec will be fully implemented in Phase 4."
        )

    def _encode(self, image_array) -> object:
        raise NotImplementedError

    def _decode(self, latent) -> object:
        raise NotImplementedError

    def _serialise_latent(self, latent) -> bytes:
        raise NotImplementedError

    def _deserialise_latent(self, data: bytes) -> object:
        raise NotImplementedError

    def _model_version(self) -> str:
        return "0.0.0-stub"


# ---------------------------------------------------------------------------
# Concrete stub: UI Screenshot domain
# ---------------------------------------------------------------------------

class UIScreenshotNeuralCodec(NeuralCodecBase):
    """
    Neural codec for UI screenshots.
    Phase 3: Stub only.
    Phase 4: Will load ONNX models trained on UI screenshot datasets.
    """

    def __init__(self) -> None:
        super().__init__("ui_screenshot")

    def _load_models(self) -> None:
        raise NotImplementedError(
            "UIScreenshotNeuralCodec: model weights not available. "
            "Phase 4 will ship pre-trained weights."
        )

    def _encode(self, image_array) -> object:
        raise NotImplementedError

    def _decode(self, latent) -> object:
        raise NotImplementedError

    def _serialise_latent(self, latent) -> bytes:
        raise NotImplementedError

    def _deserialise_latent(self, data: bytes) -> object:
        raise NotImplementedError

    def _model_version(self) -> str:
        return "0.0.0-stub"


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

class NeuralCodecSystem:
    """
    Central registry for all neural codecs.
    The orchestrator queries this to find available neural strategies.
    """

    _codecs: dict[str, NeuralCodecBase] = {
        "medical":       MedicalNeuralCodec(),
        "ui_screenshot": UIScreenshotNeuralCodec(),
    }

    @classmethod
    def get(cls, domain: str) -> NeuralCodecBase:
        if domain not in cls._codecs:
            raise KeyError(
                f"No neural codec registered for domain '{domain}'. "
                f"Available: {list(cls._codecs.keys())}"
            )
        return cls._codecs[domain]

    @classmethod
    def available_domains(cls) -> list[str]:
        return list(cls._codecs.keys())
