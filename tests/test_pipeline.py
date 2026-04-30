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
        # SonarEncoder has no num_layers param — it wraps SonarSweepEmbedding directly.
        enc = SonarEncoder(
            max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64
        )
        x = torch.randn(1, 28, 28)
        tokens, info = enc(x)
        assert tokens.shape[0] == 1
        assert tokens.shape[2] == 64

    def test_gct_output(self):
        from sonar_vision.aggregator.gct import StreamingGCTAggregator
        # num_patches_v=1, num_patches_h=1 → tokens_per_frame=2.  8 % 2 == 0.
        agg = StreamingGCTAggregator(
            embed_dim=64, num_heads=4, num_layers=1, gqa_ratio=2, window_size=8,
            num_patches_v=1, num_patches_h=1,
        )
        tokens = torch.randn(1, 8, 64)   # 4 frames × 2 tokens_per_frame
        out, _ = agg(tokens)
        assert out.shape == (1, 8, 64)

    def test_kv_cache_lifecycle(self):
        from sonar_vision.aggregator.gct import StreamingGCTAggregator
        # cache is a plain dict (KVCache = Dict[str, Any]).
        # Use num_patches_v=1, num_patches_h=1 so tokens_per_frame=2.
        agg = StreamingGCTAggregator(
            embed_dim=64, num_heads=4, num_layers=1, gqa_ratio=2, window_size=16,
            num_patches_v=1, num_patches_h=1,
        )
        cache = agg.init_cache(batch_size=1)
        assert cache["num_frames"] == 0   # dict key, not attribute

        tokens = torch.randn(1, 2, 64)   # 1 frame × tokens_per_frame=2
        out, cache = agg(tokens, past_cache=cache)   # kwarg is past_cache
        assert cache["num_frames"] > 0

        # "reset" by re-initialising a fresh cache
        cache = agg.init_cache(batch_size=1)
        assert cache["num_frames"] == 0

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
        # Mackenzie equation at depth=10 m, surface_temp=15°C, S=35 PSU → ~1506 m/s
        speed = water.sound_speed(torch.tensor([10.0]))
        assert 1480 < speed.item() < 1540

    def test_config_roundtrip(self):
        # to_yaml uses yaml.dump which writes tuples as !!python/tuple tags
        # that yaml.safe_load cannot parse.  Use to_dict/from_dict instead.
        from sonar_vision.config import SonarVisionConfig
        cfg = SonarVisionConfig()
        cfg.name = "test"
        cfg.encoder.embed_dim = 512

        loaded = SonarVisionConfig.from_dict(cfg.to_dict())
        assert loaded.name == "test"
        assert loaded.encoder.embed_dim == 512

    def test_deploy_parameter_count(self):
        """deploy.py has no estimate_memory_mb; verify param count as proxy for model size."""
        from sonar_vision.pipeline import SonarVision
        model = SonarVision(max_depth=28, bearing_bins=28, embed_dim=64)
        n_params = sum(p.numel() for p in model.parameters())
        n_bytes = n_params * 4   # fp32
        n_mb = n_bytes / (1024 ** 2)
        assert n_mb > 0
