"""
AirOne Neural Codec Trainer - Phase 4

Trains encoder/decoder pairs and exports them to ONNX format.
Requires: pip install airone[train]
    torch, torchvision, onnx

The training pipeline:
    1. Load domain-specific image dataset
    2. Train autoencoder (encoder + decoder)
    3. Validate lossless residual on held-out set
    4. Export encoder.onnx + decoder.onnx
    5. Run compression benchmark on validation set

Architecture: Convolutional autoencoder
    Encoder: Conv2d blocks → bottleneck latent
    Decoder: ConvTranspose2d blocks → reconstructed image

Loss function:
    L = MSE(reconstructed, original)
      + λ_sparsity × L1(latent)
      + λ_perceptual × perceptual_loss(reconstructed, original)

This encourages:
    - Accurate reconstruction (MSE)
    - Compact latent (L1 sparsity → better residual compression)
    - Visually faithful reconstruction (perceptual)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters for a training run."""

    domain: str

    # Data
    dataset_path:    str  = ""
    val_split:       float = 0.1
    image_size:      tuple = (512, 512)

    # Architecture
    latent_dim:      int   = 256
    encoder_channels: list = field(
        default_factory=lambda: [32, 64, 128, 256]
    )

    # Training
    num_epochs:      int   = 50
    batch_size:      int   = 16
    learning_rate:   float = 1e-3
    weight_decay:    float = 1e-4

    # Loss weights
    lambda_sparsity: float = 0.01
    lambda_perceptual: float = 0.1

    # Output
    output_dir:      str   = "./models"
    model_version:   str   = "1.0.0"

    # Hardware
    device:          str   = "auto"   # "auto", "cpu", "cuda"


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def _build_encoder(config: TrainingConfig):
    """
    Convolutional encoder: Image → Latent

    Architecture:
        Input  : (B, 3, H, W) float32 in [0,1]
        Conv blocks downsampling by 2x each stage
        Output : (B, latent_dim) float32
    """
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch required for training. "
            "Install with: pip install airone[train]"
        ) from exc

    channels = [3] + config.encoder_channels

    layers = []
    for in_ch, out_ch in zip(channels[:-1], channels[1:]):
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    # Global average pool + linear projection to latent
    layers += [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(config.encoder_channels[-1], config.latent_dim),
    ]

    return nn.Sequential(*layers)


def _build_decoder(config: TrainingConfig):
    """
    Convolutional decoder: Latent → Image

    Architecture:
        Input  : (B, latent_dim) float32
        Linear projection → reshape → ConvTranspose2d upsampling
        Output : (B, 3, H, W) float32 in [0,1]
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch required for training. "
            "Install with: pip install airone[train]"
        ) from exc

    class Decoder(nn.Module):
        def __init__(self, config: TrainingConfig) -> None:
            super().__init__()
            channels  = list(reversed(config.encoder_channels))
            init_size = config.image_size[0] // (2 ** len(channels))

            self.init_size = init_size
            self.fc = nn.Linear(
                config.latent_dim,
                channels[0] * init_size * init_size,
            )

            layers = []
            for i in range(len(channels) - 1):
                layers += [
                    nn.ConvTranspose2d(
                        channels[i], channels[i + 1],
                        kernel_size=4, stride=2, padding=1,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            layers += [
                nn.ConvTranspose2d(
                    channels[-1], 3,
                    kernel_size=4, stride=2, padding=1,
                ),
                nn.Sigmoid(),
            ]
            self.deconv = nn.Sequential(*layers)
            self._channels0 = channels[0]

        def forward(self, latent):
            x = self.fc(latent)
            x = x.view(
                x.size(0),
                self._channels0,
                self.init_size,
                self.init_size,
            )
            return self.deconv(x)

    return Decoder(config)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DomainImageDataset:
    """
    Simple image folder dataset.
    Loads all images from a directory tree.
    """

    def __init__(
        self,
        root: str,
        image_size: tuple,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp"),
    ) -> None:
        self.image_size = image_size
        self.paths = []

        root_path = Path(root)
        for ext in extensions:
            self.paths.extend(root_path.rglob(f"*{ext}"))

        if not self.paths:
            raise FileNotFoundError(
                f"No images found in: {root}"
            )
        logger.info(f"Dataset: {len(self.paths)} images in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as T
        except ImportError as exc:
            raise ImportError(
                "torchvision required for training."
            ) from exc

        transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
        ])

        img = Image.open(self.paths[idx]).convert("RGB")
        return transform(img)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Outcomes from a completed training run."""
    domain:            str
    encoder_path:      str
    decoder_path:      str
    best_loss:         float
    epochs_trained:    int
    training_time_s:   float
    val_avg_residual:  float    # Mean absolute residual on validation set


class NeuralCodecTrainer:
    """
    Trains and exports ONNX neural codecs.

    Usage::

        trainer = NeuralCodecTrainer()
        result = trainer.train(TrainingConfig(
            domain="medical",
            dataset_path="/data/ct_scans/",
            num_epochs=50,
        ))
        print(f"Models saved: {result.encoder_path}")
    """

    def train(self, config: TrainingConfig) -> TrainingResult:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, random_split
        except ImportError as exc:
            raise ImportError(
                "PyTorch required for training. "
                "pip install airone[train]"
            ) from exc

        device = self._resolve_device(config.device)
        logger.info(f"Training on: {device}")

        # Dataset
        dataset = DomainImageDataset(
            config.dataset_path, config.image_size
        )
        val_size   = max(1, int(len(dataset) * config.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size
        )

        # Models
        encoder = _build_encoder(config).to(device)
        decoder = _build_decoder(config).to(device)

        # Optimiser
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs
        )

        best_loss   = float("inf")
        start       = time.perf_counter()
        best_state  = None

        # Training loop
        for epoch in range(1, config.num_epochs + 1):
            encoder.train()
            decoder.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)

                # Forward
                latent = encoder(batch)
                reconstructed = decoder(latent)

                # Loss
                mse_loss      = nn.functional.mse_loss(reconstructed, batch)
                sparsity_loss = torch.mean(torch.abs(latent))
                loss = mse_loss + config.lambda_sparsity * sparsity_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(len(train_loader), 1)

            # Validation
            if avg_loss < best_loss:
                best_loss  = avg_loss
                best_state = {
                    "encoder": {
                        k: v.clone()
                        for k, v in encoder.state_dict().items()
                    },
                    "decoder": {
                        k: v.clone()
                        for k, v in decoder.state_dict().items()
                    },
                }

            if epoch % 10 == 0 or epoch == config.num_epochs:
                logger.info(
                    f"Epoch {epoch}/{config.num_epochs}  "
                    f"loss={avg_loss:.6f}  best={best_loss:.6f}"
                )

        # Restore best weights
        if best_state:
            encoder.load_state_dict(best_state["encoder"])
            decoder.load_state_dict(best_state["decoder"])

        # Compute mean residual on validation set
        val_residual = self._compute_val_residual(
            encoder, decoder, val_loader, device
        )

        # Export ONNX
        os.makedirs(config.output_dir, exist_ok=True)
        enc_path, dec_path = self._export_onnx(
            encoder, decoder, config, device
        )

        return TrainingResult(
            domain=config.domain,
            encoder_path=enc_path,
            decoder_path=dec_path,
            best_loss=best_loss,
            epochs_trained=config.num_epochs,
            training_time_s=time.perf_counter() - start,
            val_avg_residual=val_residual,
        )

    # ------------------------------------------------------------------

    def _compute_val_residual(
        self, encoder, decoder, val_loader, device
    ) -> float:
        try:
            import torch
        except ImportError:
            return 0.0

        encoder.eval()
        decoder.eval()
        total_residual = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                latent        = encoder(batch)
                reconstructed = decoder(latent)
                residual = (batch - reconstructed).abs().mean().item()
                total_residual += residual
                count += 1

        return total_residual / max(count, 1)

    def _export_onnx(
        self,
        encoder,
        decoder,
        config: TrainingConfig,
        device,
    ) -> tuple[str, str]:
        try:
            import torch
            import onnx
        except ImportError as exc:
            raise ImportError(
                "onnx required for export. pip install airone[train]"
            ) from exc

        H, W = config.image_size
        dummy_image  = torch.randn(1, 3, H, W).to(device)
        dummy_latent = torch.randn(1, config.latent_dim).to(device)

        enc_path = os.path.join(
            config.output_dir,
            f"{config.domain}_encoder.onnx",
        )
        dec_path = os.path.join(
            config.output_dir,
            f"{config.domain}_decoder.onnx",
        )

        encoder.eval()
        decoder.eval()

        # Export encoder
        torch.onnx.export(
            encoder,
            dummy_image,
            enc_path,
            opset_version=17,
            input_names=["input"],
            output_names=["latent"],
            dynamic_axes={
                "input":  {0: "batch"},
                "latent": {0: "batch"},
            },
        )

        # Export decoder
        torch.onnx.export(
            decoder,
            dummy_latent,
            dec_path,
            opset_version=17,
            input_names=["latent"],
            output_names=["output"],
            dynamic_axes={
                "latent": {0: "batch"},
                "output": {0: "batch"},
            },
        )

        logger.info(f"Exported encoder → {enc_path}")
        logger.info(f"Exported decoder → {dec_path}")

        return enc_path, dec_path

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        if device_str != "auto":
            return device_str
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
