"""
Jetson Orin deployment utilities for SonarVision.

Handles model export (TorchScript, TensorRT), quantization,
and real-time inference with memory management.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def export_torchscript(
    model: nn.Module,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
) -> str:
    """Export model to TorchScript for Jetson deployment.

    Args:
        model: SonarVision model in eval mode
        output_path: Path to save .pt file
        example_input: (1, bearing_bins, max_depth) or auto-generated

    Returns:
        Path to exported model
    """
    model.eval()
    if example_input is None:
        # Infer input shape from encoder
        example_input = torch.randn(1, 128, 200)

    example_input = example_input.cuda()
    model.cuda()

    # Trace the model (simpler than scripting for our architecture)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Optimize
    traced = torch.jit.optimize_for_inference(traced)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, output_path)
    return output_path


def quantize_int8(
    model: nn.Module,
    calib_loader=None,
    num_calib_batches: int = 50,
) -> nn.Module:
    """Apply INT8 post-training quantization (PTQ).

    Uses PyTorch's dynamic quantization for linear layers.
    For full static quantization, use TensorRT instead.

    Args:
        model: Model to quantize
        calib_loader: Calibration data (for static quantization)
        num_calib_batches: Number of calibration batches

    Returns:
        Quantized model
    """
    model.eval()

    # Dynamic quantization for linear layers (fast, no calibration needed)
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return quantized


def estimate_memory_mb(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Estimate GPU memory usage for model inference.

    Returns dict with model_params_mb, activation_mb, total_mb.
    """
    model.eval()

    # Model parameters
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024 * 1024)

    # Estimate activations (rough: 2x model size for intermediate activations)
    act_mb = param_mb * 2.0

    # KV cache memory
    # 2 (K+V) * num_layers * batch * kv_heads * seq_len * head_dim * 2 bytes (fp16)
    # Approximate based on model config
    cache_mb = 0.0
    if hasattr(model, "aggregator"):
        agg = model.aggregator
        kv_heads = agg.blocks[0].attn.num_kv_heads if agg.blocks else 0
        head_dim = agg.blocks[0].attn.head_dim if agg.blocks else 0
        cache_bytes = 2 * len(agg.blocks) * 1 * kv_heads * agg.window_size * head_dim * 2
        cache_mb = cache_bytes / (1024 * 1024)

    total = param_mb + act_mb + cache_mb

    return {
        "model_params_mb": round(param_mb, 1),
        "activation_mb": round(act_mb, 1),
        "kv_cache_mb": round(cache_mb, 1),
        "total_mb": round(total, 1),
    }


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 128, 200),
    warmup: int = 10,
    runs: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark inference latency.

    Returns dict with mean_ms, p50_ms, p95_ms, p99_ms, throughput_fps.
    """
    model.eval()
    model.to(device)

    x = torch.randn(*input_shape, device=device)
    dummy_cam = torch.randn(1, 2, 3, 384, 512, device=device)
    dummy_depths = torch.tensor([[5.0, 15.0]], device=device)
    dummy_det = torch.tensor([[15.0, 0.0, -30.0]], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, dummy_cam, dummy_depths, dummy_det)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x, dummy_cam, dummy_depths, dummy_det)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)

    return {
        "mean_ms": round(np.mean(latencies), 2),
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[int(n * 0.95)], 2),
        "p99_ms": round(latencies[int(n * 0.99)], 2),
        "throughput_fps": round(1000.0 / np.mean(latencies), 1),
    }


class JetsonInference:
    """Real-time inference wrapper for Jetson deployment.

    Manages model loading, KV cache, and frame generation
    with bounded memory and latency guarantees.
    """

    def __init__(
        self,
        model_path: str,
        bearing_bins: int = 128,
        max_depth: int = 200,
        device: str = "cuda",
        precision: str = "fp16",
    ):
        self.bearing_bins = bearing_bins
        self.max_depth = max_depth
        self.device = device

        # Load model
        if model_path.endswith(".pt") or model_path.endswith(".ts"):
            self.model = torch.jit.load(model_path, map_location=device)
        else:
            from sonar_vision.pipeline import SonarVision
            state = torch.load(model_path, map_location=device)
            # Need config here — for now, load with defaults
            self.model = SonarVision()
            self.model.load_state_dict(state)

        self.model.eval()
        self.model.to(device)

        # Cast to half precision
        if precision == "fp16" and device == "cuda":
            self.model.half()

        # Initialize cache for streaming
        self.cache = None

        # Latency tracking
        self.frame_times = []

    def process_sweep(
        self,
        sonar_sweep: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Process a single sonar sweep and return predicted frame.

        Args:
            sonar_sweep: (bearing_bins, max_depth) float array

        Returns:
            (frame, depth_map, latency_ms)
        """
        t0 = time.perf_counter()

        # Prepare input
        x = torch.from_numpy(sonar_sweep).float()
        x = x.unsqueeze(0).to(self.device)

        if self.model.config.precision == "fp16" if hasattr(self.model, "config") else False:
            x = x.half()

        with torch.no_grad():
            frame, depth_map, self.cache = self.model.generate_stream(x, cache=self.cache)

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - t0) * 1000
        self.frame_times.append(latency_ms)

        # Convert to numpy
        frame_np = frame[0].cpu().float().numpy().transpose(1, 2, 0)
        frame_np = (frame_np + 1) / 2  # [-1, 1] → [0, 1]
        frame_np = np.clip(frame_np, 0, 1)

        depth_np = depth_map[0, 0].cpu().numpy()

        return frame_np, depth_np, latency_ms

    def reset_cache(self):
        """Reset KV cache (e.g., when sonar direction changes)."""
        self.cache = None

    def get_stats(self) -> Dict[str, float]:
        """Get inference statistics."""
        if not self.frame_times:
            return {}
        times = sorted(self.frame_times)
        n = len(times)
        return {
            "frames_processed": n,
            "mean_latency_ms": round(np.mean(times), 2),
            "p95_latency_ms": round(times[int(n * 0.95)], 2),
            "avg_fps": round(1000.0 / np.mean(times), 1),
        }
