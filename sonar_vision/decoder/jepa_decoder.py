"""JEPA-based video decoder for SonarVision.

Joint-Embedding Predictive Architecture for temporal video prediction.
Replaces the deterministic decoder with a learned predictor that
produces temporally consistent video frames.

Architecture:
  Encoder (depth → latent) → Predictor (latent_t → latent_t+1) → Decoder (latent → frame)

Based on JEPA Perception Lab findings:
  - Law 141: Even tiny models learn useful representations
  - Law 153: Raw deltas carry more information than smoothed signals
  - Law 145: Feature weighting matters enormously
"""

import math
import logging
from typing import Optional, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger("sonar-vision.jepa")


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required for JEPA decoder. Install with: pip install torch")


class JEPAEncoder(nn.Module):
    """Depth data encoder — maps sonar returns to latent space.

    Architecture:
      Conv1D → GELU → Conv1D → GELU → Linear → Latent
    """

    def __init__(self, input_dim: int = 32, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        _check_torch()
        self.conv1 = nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.norm(x)
        return x


class JEPAPredictor(nn.Module):
    """Predicts next latent from current latent + action context.

    The predictor module that drives temporal consistency.
    Uses residual connections for stable gradient flow.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, num_steps: int = 1):
        super().__init__()
        _check_torch()
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        residual = z
        delta = self.net(z)
        return residual + delta  # Residual connection — predict the delta (Law 153)


class JEPADecoder(nn.Module):
    """Decodes latent vectors back to video frames.

    Architecture:
      Latent → Linear → ConvTranspose1D → ConvTranspose1D → Frame
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, output_dim: int = 384):
        super().__init__()
        _check_torch()
        self.fc = nn.Linear(latent_dim, hidden_dim * 16)
        self.deconv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(hidden_dim // 2, 1, kernel_size=4, stride=2, padding=1)
        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        batch = x.shape[0]
        x = x.view(batch, -1, 16)  # (B, C, T)
        x = self.deconv1(x)
        x = F.gelu(x)
        x = self.deconv2(x)
        # Adapt to target output dimension
        if x.shape[-1] != self.output_dim:
            x = F.interpolate(x, size=self.output_dim, mode='linear', align_corners=False)
        return x


class SonarJEPA(nn.Module):
    """Full JEPA model for SonarVision — depth in, video frame out.

    Combines encoder, predictor, and decoder into a single forward pass.
    Supports multi-step prediction for temporal consistency.
    """

    def __init__(
        self,
        input_dim: int = 32,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 384,
        num_prediction_steps: int = 1,
    ):
        super().__init__()
        _check_torch()
        self.latent_dim = latent_dim
        self.encoder = JEPAEncoder(input_dim, latent_dim, hidden_dim)
        self.predictor = JEPAPredictor(latent_dim, hidden_dim, num_prediction_steps)
        self.decoder = JEPADecoder(latent_dim, hidden_dim, output_dim)

    def encode(self, depth_data: torch.Tensor) -> torch.Tensor:
        """Encode depth data to latent space."""
        return self.encoder(depth_data)

    def predict(self, z: torch.Tensor, steps: int = 1) -> List[torch.Tensor]:
        """Roll out multi-step predictions in latent space."""
        predictions = []
        current = z
        for _ in range(steps):
            current = self.predictor(current)
            predictions.append(current)
        return predictions

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video frame."""
        return self.decoder(z)

    def forward(self, depth_data: torch.Tensor) -> torch.Tensor:
        z = self.encode(depth_data)
        z_next = self.predict(z, steps=1)[0]
        frame = self.decode(z_next)
        return frame

    def compute_loss(
        self,
        depth_data: torch.Tensor,
        target_frame: torch.Tensor,
        lambda_recon: float = 1.0,
        lambda_latent: float = 0.01,
    ) -> torch.Tensor:
        """Compute JEPA loss: reconstruction + latent regularization."""
        z = self.encode(depth_data)
        z_next = self.predict(z, steps=1)[0]
        pred_frame = self.decode(z_next)

        recon_loss = F.mse_loss(pred_frame, target_frame)
        latent_loss = torch.norm(z_next - z, p=2, dim=-1).mean()
        return lambda_recon * recon_loss + lambda_latent * latent_loss


class JEPADecoderInference:
    """Inference wrapper for the JEPA decoder — drop-in replacement for VideoDecoder."""

    def __init__(self, latent_dim: int = 16, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        self.model: Optional[SonarJEPA] = None
        if HAS_TORCH:
            self.model = SonarJEPA(latent_dim=latent_dim)
            self.model.eval()

    def decode(self, latent: np.ndarray) -> List[np.ndarray]:
        """Decode a batch of latent vectors to video frames.

        Args:
            latent: numpy array of shape (batch, latent_dim) or (batch, channels, time)

        Returns:
            List of numpy arrays, each shape (height, width)
        """
        if self.model is None:
            log.warning("PyTorch not available — returning latent as frame")
            return [latent[0] if latent.ndim > 1 else latent]

        with torch.no_grad():
            z = torch.from_numpy(latent).float().to(self.device)

            # If latent is multi-channel, project to latent_dim
            if z.ndim > 2 or z.shape[-1] != self.latent_dim:
                z = z.mean(dim=tuple(range(1, z.ndim - 1))) if z.ndim > 2 else z
                if z.shape[-1] != self.latent_dim:
                    z = z[..., :self.latent_dim]

            frame = self.model.decode(z)
            frames = []
            for i in range(frame.shape[0]):
                f = frame[i, 0].cpu().numpy()
                f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                frames.append(f)

            return frames


# Benchmark: JEPA vs Deterministic Decoder
def benchmark_decoders(depth_sample: np.ndarray, iterations: int = 100) -> dict:
    """Compare JEPA vs deterministic decoder performance."""
    import time

    results = {}

    # JEPA decoder
    if HAS_TORCH:
        jepa = JEPADecoderInference()
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = jepa.decode(depth_sample)
            times.append((time.perf_counter() - t0) * 1000)

        results["jepa_mean_ms"] = float(np.mean(times))
        results["jepa_std_ms"] = float(np.std(times))
    else:
        results["jepa"] = "skipped (no torch)"

    # Deterministic decoder (numpy baseline)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = depth_sample  # identity decode
        times.append((time.perf_counter() - t0) * 1000)

    results["deterministic_mean_ms"] = float(np.mean(times))
    results["deterministic_std_ms"] = float(np.std(times))
    results["iterations"] = iterations

    return results
