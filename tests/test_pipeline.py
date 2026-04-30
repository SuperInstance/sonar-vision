"""Tests for SonarVision pipeline."""

import torch
import pytest

from sonar_vision.pipeline import SonarVision


class TestSonarVisionForward:
    def test_generate_shape(self):
        model = SonarVision(
            max_depth=200, bearing_bins=128, embed_dim=256
        )
        # Use small model params to stay in memory
        model.encoder = torch.nn.Identity()
        model.aggregator = torch.nn.Identity()
        model.feature_adapter = torch.nn.Identity()

        sonar = torch.randn(1, 128, 200)
        output = model.generate(sonar)
        assert "frame" in output
        assert "depth_map" in output

    def test_training_loss(self):
        model = SonarVision(
            max_depth=50, bearing_bins=32, embed_dim=128
        )
        sonar = torch.randn(2, 32, 50)
        cam_frames = torch.randn(2, 2, 3, 384, 512)
        cam_depths = torch.tensor([[5.0, 15.0], [10.0, 20.0]])
        det = torch.tensor([[15.0, 0.0, -30.0], [10.0, 45.0, -25.0]])

        output = model(
            sonar_intensity=sonar,
            camera_frames=cam_frames,
            camera_depths=cam_depths,
            sonar_detections=det,
        )
        assert "loss" in output
        assert "loss_dict" in output
        assert output["loss"].requires_grad


class TestDepthWeights:
    def test_weight_computation(self):
        from sonar_vision.data.sonar_dataset import SonarVideoDataset

        cam_depths = [5.0, 10.0, 15.0, 20.0]
        detections = [{"depth": 15.2, "bearing": 0.0, "intensity": -30.0}]
        weights = SonarVideoDataset._compute_depth_weights(None, cam_depths, detections)

        # Camera at 15m should have highest weight (closest to 15.2m)
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]
        assert weights[2] > weights[3]
        assert weights[2] > 0.9  # Should be very high

    def test_no_detections(self):
        from sonar_vision.data.sonar_dataset import SonarVideoDataset

        cam_depths = [5.0, 15.0]
        weights = SonarVideoDataset._compute_depth_weights(None, cam_depths, [])
        assert all(w == 0.01 for w in weights)


class TestStreamingInference:
    def test_streaming_with_cache(self):
        model = SonarVision(
            max_depth=50, bearing_bins=32, embed_dim=128
        )
        model.eval()

        cache = model.aggregator.init_cache(batch_size=1, device=torch.device("cpu"))
        assert cache is not None

        sonar = torch.randn(1, 32, 50)
        frame, depth, cache = model.generate_stream(sonar, cache=cache)
        assert frame is not None
        assert depth is not None

        # Second call with same cache
        sonar2 = torch.randn(1, 32, 50)
        frame2, depth2, cache = model.generate_stream(sonar2, cache=cache)
        assert frame2.shape == frame.shape
