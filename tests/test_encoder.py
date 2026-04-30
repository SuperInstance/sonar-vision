"""Tests for SonarEncoder and SonarSweepEmbedding."""

import torch
import pytest

from sonar_vision.encoder.sonar_encoder import SonarEncoder, SonarSweepEmbedding


class TestSonarSweepEmbedding:
    def test_output_shape(self):
        embed = SonarSweepEmbedding(
            max_depth=200, bearing_bins=128, patch_size=14, embed_dim=512
        )
        x = torch.randn(2, 128, 200)
        out = embed(x)
        # Should produce (B, num_patches, embed_dim) + (B, embed_dim) for scale token
        assert out.shape[0] == 2
        assert out.shape[2] == 512

    def test_different_input_sizes(self):
        embed = SonarSweepEmbedding(
            max_depth=200, bearing_bins=128, patch_size=14, embed_dim=256
        )
        x = torch.randn(1, 128, 200)
        out = embed(x)
        assert out.shape[0] == 1
        assert out.shape[2] == 256

    def test_4channel_input(self):
        embed = SonarSweepEmbedding(
            max_depth=200, bearing_bins=64, patch_size=14, embed_dim=256, in_channels=4
        )
        x = torch.randn(1, 4, 64, 200)
        out = embed(x)
        assert out.shape[2] == 256


class TestSonarEncoder:
    def test_output_shape(self):
        enc = SonarEncoder(
            max_depth=200, bearing_bins=128, patch_size=14, embed_dim=512, num_layers=2
        )
        x = torch.randn(2, 128, 200)
        tokens, info = enc(x)
        assert tokens.shape[0] == 2
        assert tokens.shape[2] == 512

    def test_with_accumulated(self):
        enc = SonarEncoder(
            max_depth=200, bearing_bins=64, patch_size=14, embed_dim=256, num_layers=1
        )
        x = torch.randn(1, 64, 200)
        acc = torch.randn(1, 64, 200)
        tokens, info = enc(x, accumulated=acc)
        assert tokens.shape[0] == 1

    def test_with_depth_axis(self):
        enc = SonarEncoder(
            max_depth=100, bearing_bins=32, patch_size=14, embed_dim=128, num_layers=1
        )
        x = torch.randn(1, 32, 100)
        depth = torch.linspace(0, 100, 100).unsqueeze(0)
        tokens, info = enc(x, depth_axis=depth)
        assert tokens.shape[2] == 128
