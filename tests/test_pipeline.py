"""Tests for SonarVision pipeline."""

import torch
import pytest


class TestSonarVisionPipeline:
    def test_model_creation(self):
        from sonar_vision.pipeline import SonarVision
        model = SonarVision(
            max_depth=50, bearing_bins=32, embed_dim=128
        )
        # Count params to make sure it's not empty
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_encoder_output(self):
        from sonar_vision.encoder.sonar_encoder import SonarEncoder
        enc = SonarEncoder(
            max_depth=50, bearing_bins=32, patch_size=14, embed_dim=128, num_layers=1
        )
        x = torch.randn(1, 32, 50)
        tokens, info = enc(x)
        assert tokens.shape[0] == 1
        assert tokens.shape[2] == 128

    def test_gct_output(self):
        from sonar_vision.aggregator.gct import StreamingGCTAggregator
        agg = StreamingGCTAggregator(
            embed_dim=128, num_heads=4, num_layers=1, gqa_ratio=2, window_size=8
        )
        tokens = torch.randn(1, 8, 128)
        out, _ = agg(tokens)
        assert out.shape == (1, 8, 128)

    def test_kv_cache_lifecycle(self):
        from sonar_vision.aggregator.gct import StreamingGCTAggregator
        agg = StreamingGCTAggregator(
            embed_dim=128, num_heads=4, num_layers=1, gqa_ratio=2, window_size=16
        )
        cache = agg.init_cache(batch_size=1)
        assert cache.seq_len == 0

        tokens = torch.randn(1, 4, 128)
        out, cache = agg(tokens, cache=cache)
        assert cache.seq_len > 0

        cache.reset()
        assert cache.seq_len == 0

    def test_depth_weight_logic(self):
        """Verify the depth-weighted supervision concept."""
        import math
        sigma = 3.0
        cam_depths = [5.0, 10.0, 15.0, 20.0]
        target_depth = 15.2

        weights = [math.exp(-(d - target_depth)**2 / (2 * sigma**2)) for d in cam_depths]

        # Camera at 15m (index 2) should dominate
        assert weights[2] > 0.9
        assert weights[2] > weights[0]
        assert weights[2] > weights[3]

    def test_water_physics(self):
        from sonar_vision.water.physics import WaterColumnModel
        water = WaterColumnModel()
        speed = water.sound_speed(torch.tensor([10.0]))
        assert 1480 < speed.item() < 1500

    def test_config_roundtrip(self):
        import tempfile, os
        from sonar_vision.config import SonarVisionConfig
        cfg = SonarVisionConfig()
        cfg.name = "test"
        cfg.encoder.embed_dim = 512

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            cfg.to_yaml(f.name)
            path = f.name

        try:
            loaded = SonarVisionConfig.from_yaml(path)
            assert loaded.name == "test"
            assert loaded.encoder.embed_dim == 512
        finally:
            os.unlink(path)

    def test_deploy_memory_estimate(self):
        from sonar_vision.pipeline import SonarVision
        from sonar_vision.deploy import estimate_memory_mb
        model = SonarVision(max_depth=50, bearing_bins=32, embed_dim=128)
        mem = estimate_memory_mb(model, (1, 32, 50))
        assert mem["model_params_mb"] > 0
        assert mem["total_mb"] > 0
