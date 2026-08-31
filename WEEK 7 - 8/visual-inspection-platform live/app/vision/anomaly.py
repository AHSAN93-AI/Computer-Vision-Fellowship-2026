"""
Unsupervised anomaly model (`anomaly_v1`) -- a convolutional autoencoder
trained (per Requirement 24) primarily on NORMAL samples. At inference time
we compare the reconstruction to the input; regions the autoencoder cannot
reconstruct well are the anomaly heatmap, and the overall reconstruction
error becomes the anomaly score.

Architecture reconstructed from anomaly_autoencoder.pt's state_dict:
conv_encoder.{0,2,4} are stride-2 convs (64x64 -> 32x32 -> 16x16 -> 8x8,
64 channels = 4096 features), to_latent/from_latent bottleneck to 16 dims,
conv_decoder.{0,2,4} are the mirrored transposed convs back to 64x64x1.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from app.config import ANOMALY_MODEL_PATH, IMG_SIZE, IN_CHANNELS

_LATENT_DIM = 16
_BOTTLENECK_CHANNELS = 64
_BOTTLENECK_SIZE = 8  # 64 / 2 / 2 / 2


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, latent_dim: int = _LATENT_DIM):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 0
            nn.ReLU(inplace=True),                                          # 1
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # 2
            nn.ReLU(inplace=True),                                          # 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 4
            nn.ReLU(inplace=True),                                          # 5
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),                                                    # 0
            nn.Linear(_BOTTLENECK_CHANNELS * _BOTTLENECK_SIZE * _BOTTLENECK_SIZE, latent_dim),  # 1
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, _BOTTLENECK_CHANNELS * _BOTTLENECK_SIZE * _BOTTLENECK_SIZE),  # 0
            nn.ReLU(inplace=True),                                          # 1
        )
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 0
            nn.ReLU(inplace=True),                                          # 1
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2
            nn.ReLU(inplace=True),                                          # 3
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4
            nn.Sigmoid(),                                                    # 5
        )

    def forward(self, x):
        z = self.to_latent(self.conv_encoder(x))
        y = self.from_latent(z)
        y = y.view(-1, _BOTTLENECK_CHANNELS, _BOTTLENECK_SIZE, _BOTTLENECK_SIZE)
        recon = self.conv_decoder(y)
        return recon


_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> ConvAutoencoder:
    global _model
    if _model is None:
        model = ConvAutoencoder()
        state_dict = torch.load(ANOMALY_MODEL_PATH, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _model = model
    return _model


def anomaly_map(image_tensor: torch.Tensor, threshold: float):
    """
    image_tensor: float tensor, shape (1, IN_CHANNELS, IMG_SIZE, IMG_SIZE), values in [0, 1].
    threshold: anomaly_score_threshold from decision settings.

    Returns a dict:
        {
          "anomaly_score": float,             # mean reconstruction error, roughly 0-1
          "is_anomalous": bool,
          "error_map": np.ndarray (H, W) float32, per-pixel reconstruction error
          "thresholded_mask": np.ndarray (H, W) bool, per-pixel mask used for area estimation
          "inference_time_ms": float,
        }
    """
    model = load_model()
    start = time.perf_counter()
    with torch.no_grad():
        recon = model(image_tensor.to(_device))
        error = torch.abs(recon - image_tensor.to(_device))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    error_map = error.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    anomaly_score = float(error_map.mean())

    # Per-pixel threshold for localization: a pixel is "anomalous" if its
    # local error is well above the map's own mean+std -- this adapts to
    # each image rather than using one global pixel cutoff.
    pixel_cutoff = error_map.mean() + error_map.std()
    thresholded_mask = error_map > max(pixel_cutoff, 1e-6)

    return {
        "anomaly_score": anomaly_score,
        "is_anomalous": anomaly_score >= threshold,
        "error_map": error_map,
        "thresholded_mask": thresholded_mask,
        "inference_time_ms": elapsed_ms,
    }
