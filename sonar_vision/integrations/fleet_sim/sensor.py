"""Synthetic sonar sensor for fleet-simulator agents."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from sonar_vision.integrations.marine_gpu import MEPSonarPacket


@dataclass
class UnderwaterEnvironment:
    """Simulated underwater environment properties."""
    depth: float = 10.0
    water_type: str = "coastal"
    turbidity: float = 0.3
    thermocline_depth: float = 15.0
    fish_density: float = 0.1
    seabed_type: str = "sandy"
    ambient_light: float = 0.8


class SimulatedSonarSensor:
    """Simulates a depth sounder for fleet simulator agents.

    Generates synthetic sonar returns based on environment state,
    feeds them through SonarVision inference.

    Usage in fleet-simulator:
        sensor = SimulatedSonarSensor()
        env = UnderwaterEnvironment(depth=22.0, water_type="oceanic")
        packet = sensor.ping(env)
        video_frame = sensor.to_video(packet)
    """

    def __init__(self, bearing_bins: int = 32, max_depth: float = 50.0):
        self.bearing_bins = bearing_bins
        self.max_depth = max_depth
        self.rng = np.random.default_rng(42)

    def ping(self, env: UnderwaterEnvironment) -> MEPSonarPacket:
        """Generate a synthetic sonar return from environment state."""
        t = np.linspace(0, 1, self.bearing_bins)
        # Depth profile
        seabed_return = np.exp(-((t - 0.7) ** 2) / 0.01) * env.depth / self.max_depth
        # Fish scatter (random clusters)
        n_fish = int(env.fish_density * 100)
        fish_returns = np.zeros(self.bearing_bins)
        if n_fish > 0:
            positions = self.rng.integers(0, self.bearing_bins, n_fish)
            fish_returns[positions] += self.rng.uniform(0.1, 0.5, n_fish)
        # Turbidity attenuation
        attenuation = np.exp(-env.turbidity * np.arange(self.bearing_bins) / self.bearing_bins)
        # Noise
        noise = self.rng.normal(0, 0.05, self.bearing_bins)

        depth_data = ((seabed_return + fish_returns) * attenuation + noise).astype(np.float32)

        return MEPSonarPacket(
            seq=self.rng.integers(0, 100000),
            mep_type=6,
            depth_data=depth_data.reshape(1, -1),
            timestamps=[float(self.rng.random())],
            water_type=env.water_type,
            metadata={
                "depth": env.depth,
                "fish_count": n_fish,
                "turbidity": env.turbidity,
            },
        )

    def to_video(self, packet: MEPSonarPacket) -> np.ndarray:
        """Convert sonar packet to video frame (placeholder for actual inference)."""
        return packet.depth_data
