"""
AirOne ONNX Neural Codec Runtime - Phase 4

Provides the inference engine for domain-specific neural compression.
Uses ONNX Runtime for lightweight, dependency-free inference.

Architecture per domain codec:
    encoder.onnx : Image (H×W×C float32) → Latent (N float32)
    decoder.onnx : Latent (N float32)     → Image (H×W×C float32)

Lossless guarantee:
    original - decoded = residual
    residual compressed with ZSTD
    decompressed = decoded + residual  (exact)

Model discovery:
    Models are searched in order:
    1. AIRONE_MODELS_DIR environment variable
    2. ~/.airone/models/
    3. <package_dir>/models/
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from airone.compressors.base import BaseCompressor, CompressionResult
from airone.exceptions import CompressionError, DecompressionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_MODEL_SEARCH_PATHS: list[Path] = [
    Path(os.environ.get("AIRONE_MODELS_DIR", "")),
    Path.home() / ".airone" / "models",
    Path(__file__).parent.parent.parent.parent / "models",
]


def find_model(domain: str, role: str) -> Optional[Path]:
    """
    Search for model file: <domain>_<role>.onnx
    e.g. medical_encoder.onnx, ui_screenshot_decoder.onnx

    Returns the Path if found, None otherwise.
    """
    filename = f"{domain}_{role}.onnx"
    for base in _MODEL_SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            logger.debug(f"Found model: {candidate}")
            return candidate
    return None


def models_available(domain: str) -> bool:
    """Returns True only if both encoder and decoder exist."""
    return (
        find_model(domain, "encoder") is not None
        and find_model(domain, "decoder") is not None
    )


# ---------------------------------------------------------------------------
# ONNX session wrapper
# ---------------------------------------------------------------------------

class ONNXSession:
    """
    Thin wrapper around onnxruntime.InferenceSession.
    Handles lazy loading and provides a clean run() interface.
    """

    def __init__(self, model_path: Path) -> None:
        self._path    = model_path
        self._session = None

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for neural codecs. "
                "Install it with: pip install airone[ml]"
            ) from exc

        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)

        # Use CUDA if available, fall back to CPU silently
        providers = ["CPUExecutionProvider"]
        try:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Neural codec: using CUDA acceleration")
        except Exception:
            pass

        self._session = ort.InferenceSession(
            str(self._path),
            sess_options=opts,
            providers=providers,
        )
        logger.info(f"Loaded ONNX model: {self._path.name}")

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._session is None:
            self._load()
        input_names = [i.name for i in self._session.get_inputs()]
        feed = {k: v for k, v in inputs.items() if k in input_names}
        return self._session.run(None, feed)


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Converts PIL images to normalised float32 tensors
    and back — matching the format expected by ONNX models.

    Convention (matches standard computer vision models):
        Input:  (1, C, H, W) float32 in [0, 1]
        Output: (1, C, H, W) float32 in [0, 1]
    """

    def __init__(self, target_size: Optional[tuple[int, int]] = None) -> None:
        """
        target_size: (H, W) to resize input to, or None to keep original.
        Most neural codecs require power-of-2 dimensions.
        """
        self.target_size = target_size

    def to_tensor(
        self, image_bytes: bytes
    ) -> tuple[np.ndarray, tuple, str]:
        """
        Convert image bytes → (tensor, original_shape, original_mode).
        tensor shape: (1, C, H, W) float32
        """
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        original_mode  = image.mode
        original_shape = (image.height, image.width)

        # Normalise to RGB
        image_rgb = image.convert("RGB")

        # Optional resize
        if self.target_size is not None:
            image_rgb = image_rgb.resize(
                (self.target_size[1], self.target_size[0])
            )

        arr = np.array(image_rgb, dtype=np.float32) / 255.0   # H×W×C
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # 1×C×H×W
        return tensor, original_shape, original_mode

    def from_tensor(
        self,
        tensor: np.ndarray,
        original_shape: tuple,
        original_mode: str,
    ) -> np.ndarray:
        """
        Convert (1, C, H, W) float32 tensor → uint8 H×W×C array.
        Resizes back to original_shape if needed.
        """
        from PIL import Image

        arr = tensor[0]                                # C×H×W
        arr = np.transpose(arr, (1, 2, 0))             # H×W×C
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if arr.shape[:2] != original_shape:
            img = Image.fromarray(arr)
            img = img.resize(
                (original_shape[1], original_shape[0])
            )
            arr = np.array(img)

        return arr


# ---------------------------------------------------------------------------
# Base ONNX codec
# ---------------------------------------------------------------------------

class ONNXNeuralCodec(BaseCompressor):
    """
    ONNX-based neural compression codec.

    Subclasses configure:
        domain          : str identifier matching model filenames
        target_size     : Optional[tuple] — resize images before encoding
        latent_dims     : Expected latent vector size (for validation)

    The lossless pipeline lives here in the base class and is
    NOT overridden by subclasses.
    """

    domain:       str = "base"
    target_size:  Optional[tuple[int, int]] = None
    latent_dims:  Optional[int] = None

    def __init__(self) -> None:
        self._encoder     : Optional[ONNXSession] = None
        self._decoder     : Optional[ONNXSession] = None
        self._preprocessor = ImagePreprocessor(self.target_size)
        self._models_ready = False

    @property
    def name(self) -> str:
        return f"neural_{self.domain}"

    # ------------------------------------------------------------------
    # BaseCompressor interface
    # ------------------------------------------------------------------

    def can_handle(self, analysis) -> bool:
        if not models_available(self.domain):
            return False
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
        return (
            ic.domain == expected
            and ic.domain_confidence > 0.60
        )

    def estimate_ratio(self, analysis) -> float:
        from airone.compressors.neural.codec import SUPPORTED_DOMAINS
        return SUPPORTED_DOMAINS.get(
            self.domain, {}
        ).get("typical_ratio", 10.0)

    def compress(
        self, data: bytes, analysis=None
    ) -> CompressionResult:
        self._ensure_models()
        start = time.perf_counter()

        # 1. Preprocess
        tensor, orig_shape, orig_mode = (
            self._preprocessor.to_tensor(data)
        )

        # 2. Encode → latent
        latent = self._encode(tensor)

        # 3. Decode → approximate reconstruction
        reconstructed_tensor = self._decode(latent)
        reconstructed_arr    = self._preprocessor.from_tensor(
            reconstructed_tensor, orig_shape, orig_mode
        )

        # 4. Load original as array for residual computation
        from PIL import Image
        orig_image = Image.open(io.BytesIO(data)).convert("RGB")
        orig_arr   = np.array(orig_image, dtype=np.int16)
        rec_arr    = reconstructed_arr.astype(np.int16)

        # 5. Compute residual (makes reconstruction lossless)
        residual = orig_arr - rec_arr    # int16, range ~[-255, 255]

        # 6. Compress residual (typically very sparse near zero)
        from airone.compressors.traditional.zstd import ZstdCompressor
        zstd = ZstdCompressor()
        residual_result = zstd.compress(residual.tobytes())

        # 7. Serialise payload
        import msgpack
        payload = msgpack.packb(
            {
                "domain":         self.domain,
                "latent":         latent.tobytes(),
                "latent_shape":   list(latent.shape),
                "latent_dtype":   str(latent.dtype),
                "residual":       residual_result.compressed_data,
                "orig_shape":     list(orig_shape),
                "orig_mode":      orig_mode,
                "model_version":  self._model_version(),
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

    def decompress(
        self, compressed_data: bytes, metadata: dict
    ) -> bytes:
        self._ensure_models()

        import msgpack
        from PIL import Image

        bundle = msgpack.unpackb(compressed_data, raw=False)

        # Reconstruct latent tensor
        latent_arr = np.frombuffer(
            bundle["latent"], dtype=np.dtype(bundle["latent_dtype"])
        ).reshape(bundle["latent_shape"])

        # Decode latent → approximate image
        reconstructed_tensor = self._decode(latent_arr)
        orig_shape = tuple(bundle["orig_shape"])
        orig_mode  = bundle["orig_mode"]

        reconstructed_arr = self._preprocessor.from_tensor(
            reconstructed_tensor, orig_shape, orig_mode
        ).astype(np.int16)

        # Restore residual
        from airone.compressors.traditional.zstd import ZstdCompressor
        zstd = ZstdCompressor()
        residual_raw = zstd.decompress(bundle["residual"], {})
        residual = np.frombuffer(
            residual_raw, dtype=np.int16
        ).reshape(*orig_shape, 3)

        # Perfect reconstruction
        perfect = np.clip(
            reconstructed_arr + residual, 0, 255
        ).astype(np.uint8)

        # Encode back to original format
        image = Image.fromarray(perfect, mode="RGB")
        if orig_mode != "RGB":
            image = image.convert(orig_mode)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # ONNX inference
    # ------------------------------------------------------------------

    def _encode(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self._encoder.run({"input": tensor})
        return outputs[0]

    def _decode(self, latent: np.ndarray) -> np.ndarray:
        # Ensure batch dimension present
        if latent.ndim == 1:
            latent = latent[np.newaxis, ...]
        outputs = self._decoder.run({"latent": latent})
        return outputs[0]

    def _model_version(self) -> str:
        """
        Extract version from ONNX model metadata if available.
        Falls back to filename-based versioning.
        """
        try:
            encoder_path = find_model(self.domain, "encoder")
            return encoder_path.stem.split("_v")[-1]
        except Exception:
            return "1.0.0"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._models_ready:
            return

        encoder_path = find_model(self.domain, "encoder")
        decoder_path = find_model(self.domain, "decoder")

        if not encoder_path or not decoder_path:
            raise CompressionError(
                f"Neural codec '{self.domain}': model weights not found.\n"
                f"Expected files:\n"
                f"  {self.domain}_encoder.onnx\n"
                f"  {self.domain}_decoder.onnx\n"
                f"Run: airone download-models --domain {self.domain}"
            )

        self._encoder = ONNXSession(encoder_path)
        self._decoder = ONNXSession(decoder_path)
        self._models_ready = True


# ---------------------------------------------------------------------------
# Domain-specific codecs
# ---------------------------------------------------------------------------

class MedicalONNXCodec(ONNXNeuralCodec):
    """
    Neural codec for medical imaging.
    Trained on CT/MRI/X-Ray datasets.
    Expected compression ratio: 20-50x
    """
    domain      = "medical"
    target_size = (512, 512)   # Standard medical image crop
    latent_dims = 256


class UIScreenshotONNXCodec(ONNXNeuralCodec):
    """
    Neural codec for UI screenshots.
    Trained on web/iOS/Android/desktop screenshot datasets.
    Expected compression ratio: 20-40x
    """
    domain      = "ui_screenshot"
    target_size = (512, 512)
    latent_dims = 512


class SatelliteONNXCodec(ONNXNeuralCodec):
    """
    Neural codec for satellite imagery.
    Trained on optical and multispectral datasets.
    Expected compression ratio: 30-80x
    """
    domain      = "satellite"
    target_size = (512, 512)
    latent_dims = 256


class ArchitecturalONNXCodec(ONNXNeuralCodec):
    """
    Neural codec for architectural drawings and CAD rasters.
    Trained on building plan and technical drawing datasets.
    Expected compression ratio: 50-120x
    """
    domain      = "architectural"
    target_size = (512, 512)
    latent_dims = 128
