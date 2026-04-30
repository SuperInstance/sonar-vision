"""Tests for SonarEncoder and SonarSweepEmbedding.

Source notes:
  SonarSweepEmbedding.forward returns (tokens, info), NOT a bare tensor.
  SonarEncoder has no num_layers param.
  Input shape: (B, bearing_bins, max_depth) — 3-D.
  tokens shape: (B, num_patches + 1, embed_dim)  [+1 for scale token].
"""

import torch
import pytest
from sonar_vision.encoder.sonar_encoder import SonarEncoder, SonarSweepEmbedding


class TestSonarSweepEmbedding:

    def test_output_shape(self):
        # 28//14=2 depth patches, 28//14=2 bearing patches → 4+1=5 tokens
        embed = SonarSweepEmbedding(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64)
        x = torch.randn(2, 28, 28)
        tokens, info = embed(x)

        assert tokens.shape == (2, 5, 64)
        assert info["num_patches"] == 4
        assert info["has_scale_token"] is True

    def test_different_input_sizes(self):
        # 56//14=4 depth, 42//14=3 bearing → 12+1=13 tokens
        embed = SonarSweepEmbedding(max_depth=56, bearing_bins=42, patch_size=14, embed_dim=32)
        x = torch.randn(1, 42, 56)
        tokens, info = embed(x)

        assert tokens.shape == (1, 13, 32)
        assert info["num_patches"] == 12
        assert tokens.shape[0] == 1
        assert tokens.shape[2] == 32

    def test_4channel_processing_via_accumulated(self):
        """Providing accumulated data exercises the 4th internal channel."""
        embed = SonarSweepEmbedding(
            max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64, in_channels=4
        )
        x = torch.randn(1, 28, 28)
        acc = torch.rand(1, 28, 28)
        tokens, _ = embed(x, accumulated=acc)

        assert tokens.shape[2] == 64


class TestSonarEncoder:

    def test_output_shape(self):
        enc = SonarEncoder(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64)
        x = torch.randn(2, 28, 28)
        tokens, info = enc(x)

        assert tokens.shape == (2, 5, 64)
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == 64
        assert info["num_patches"] == 4

    def test_with_accumulated(self):
        enc = SonarEncoder(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64)
        x = torch.randn(1, 28, 28)
        acc = torch.rand(1, 28, 28)
        tokens, info = enc(x, accumulated=acc)

        assert tokens.shape == (1, 5, 64)
        assert tokens.shape[0] == 1

    def test_with_depth_axis(self):
        enc = SonarEncoder(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64)
        x = torch.randn(1, 28, 28)
        depth = torch.linspace(0, 27, 28).unsqueeze(0)
        tokens, info = enc(x, depth_axis=depth)

        assert tokens.shape[2] == 64

    def test_various_embed_dims(self):
        for embed_dim in [32, 64, 128]:
            enc = SonarEncoder(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=embed_dim)
            tokens, _ = enc(torch.randn(1, 28, 28))
            assert tokens.shape == (1, 5, embed_dim), f"Failed for embed_dim={embed_dim}"

    def test_various_input_sizes(self):
        cases = [
            (28, 14, 14, 2),
            (28, 28, 14, 4),
            (56, 28, 14, 8),
        ]
        for max_depth, bearing_bins, patch_size, n_patches in cases:
            enc = SonarEncoder(
                max_depth=max_depth,
                bearing_bins=bearing_bins,
                patch_size=patch_size,
                embed_dim=32,
            )
            tokens, info = enc(torch.randn(1, bearing_bins, max_depth))
            assert tokens.shape == (1, n_patches + 1, 32)
            assert info["num_patches"] == n_patches

    def test_output_finite(self):
        enc = SonarEncoder(max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64)
        tokens, _ = enc(torch.randn(1, 28, 28))
        assert torch.isfinite(tokens).all()

    def test_pretrained_path_not_found_does_not_crash(self):
        enc = SonarEncoder(
            max_depth=28, bearing_bins=28, patch_size=14, embed_dim=64,
            pretrained="/nonexistent/path.pth",
        )
        tokens, _ = enc(torch.randn(1, 28, 28))
        assert tokens.shape == (1, 5, 64)
