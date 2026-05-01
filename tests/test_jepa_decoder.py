"""Tests for JEPA-based video decoder."""

import pytest
import numpy as np
from sonar_vision.decoder.jepa_decoder import (
    JEPADecoderInference, SonarJEPA, benchmark_decoders,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestJEPADecoderInference:
    def test_decode_numpy_latent(self):
        decoder = JEPADecoderInference()
        latent = np.random.randn(1, 16).astype(np.float32)
        frames = decoder.decode(latent)
        assert len(frames) > 0
        assert isinstance(frames[0], np.ndarray)

    def test_decode_batch(self):
        decoder = JEPADecoderInference()
        latent = np.random.randn(4, 16).astype(np.float32)
        frames = decoder.decode(latent)
        assert len(frames) == 4

    def test_decode_fallback_no_torch(self):
        decoder = JEPADecoderInference()
        latent = np.random.randn(3, 5).astype(np.float32)
        frames = decoder.decode(latent)
        # Should work even without torch (returns latent as frame)
        assert len(frames) == 3


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
class TestSonarJEPA:
    def test_forward(self):
        model = SonarJEPA(input_dim=32, latent_dim=16)
        depth = torch.randn(1, 1, 32)
        frame = model(depth)
        assert frame.shape[-1] >= 1

    def test_encode_decode_roundtrip(self):
        model = SonarJEPA(latent_dim=16)
        depth = torch.randn(1, 1, 32)
        z = model.encode(depth)
        assert z.shape[-1] == 16
        frame = model.decode(z)
        assert frame.ndim >= 2

    def test_multi_step_prediction(self):
        model = SonarJEPA(latent_dim=16)
        z = torch.randn(1, 16)
        preds = model.predict(z, steps=5)
        assert len(preds) == 5
        assert all(p.shape == z.shape for p in preds)


class TestBenchmark:
    def test_benchmark_runs(self):
        depth = np.random.randn(1, 32).astype(np.float32)
        results = benchmark_decoders(depth, iterations=10)
        assert "deterministic_mean_ms" in results
        assert "iterations" in results
        assert results["iterations"] == 10
