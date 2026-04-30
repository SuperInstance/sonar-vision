"""Tests for marine-gpu-edge integration bridge."""

import pytest
import numpy as np
from sonar_vision.integrations.marine_gpu import (
    MarineGPUBridge, CUDASonarPipeline, MEPSonarPacket
)


class TestMEPSonarPacket:
    def test_default_metadata(self):
        data = np.random.randn(10, 32).astype(np.float32)
        packet = MEPSonarPacket(
            seq=1, mep_type=6,
            depth_data=data,
            timestamps=[float(i) for i in range(10)],
        )
        assert packet.metadata == {}
        assert packet.water_type == "coastal"
        assert packet.depth_data.shape == (10, 32)

    def test_custom_metadata(self):
        packet = MEPSonarPacket(
            seq=1, mep_type=7,
            depth_data=np.array([[1.0, 2.0]]),
            timestamps=[0.0],
            metadata={"depth": 15.2, "temp": 8.3},
        )
        assert packet.metadata["depth"] == 15.2


class TestCUDASonarPipeline:
    def test_to_sonar_vision_tensor(self):
        data = np.random.randn(10, 32).astype(np.float32)
        pipeline = CUDASonarPipeline(data)
        tensor = pipeline.to_sonar_vision_tensor()
        assert tensor.shape == (1, 10, 32)
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_constant_data_handling(self):
        data = np.ones((5, 5), dtype=np.float32)
        pipeline = CUDASonarPipeline(data)
        tensor = pipeline.to_sonar_vision_tensor()
        assert np.allclose(tensor, 0.0), "Constant data should normalize to 0"

    def test_infer_returns_array(self):
        data = np.random.randn(8, 16).astype(np.float32)
        pipeline = CUDASonarPipeline(data)
        result = pipeline.infer()
        assert result is not None
        assert isinstance(result, np.ndarray)


class TestMarineGPUBridge:
    def test_packet_decoding_waterfall(self):
        """Test MEP waterfall packet decoding."""
        data = np.random.randn(5, 8).astype(np.float32)
        payload = np.array([5, 8], dtype=np.float32).tobytes() + data.tobytes()

        bridge = MarineGPUBridge()
        packet = bridge._decode_payload(6, 1, payload)
        assert isinstance(packet, MEPSonarPacket)
        assert packet.depth_data.shape == (5, 8)
        assert packet.seq == 1

    def test_packet_decoding_sensor_fused(self):
        payload = b"coastal\x00\x00" + np.array([2, 10.0, 8.3, 5.0, 15.0, 7.1, 6.0], dtype=np.float32).tobytes()
        bridge = MarineGPUBridge()
        packet = bridge._decode_payload(8, 2, payload)
        assert packet.water_type == "coastal"
        assert packet.metadata.get("sensor_fused") is True
        assert packet.depth_data.shape[1] >= 1

    def test_packet_decoding_empty_payload(self):
        bridge = MarineGPUBridge()
        packet = bridge._decode_payload(10, 0, b"")
        assert packet.seq == 0
        assert packet.mep_type == 10
