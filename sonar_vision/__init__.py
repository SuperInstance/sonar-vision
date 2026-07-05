"""Sonar Vision — signal processing and spatial awareness for agents."""

from sonar_vision.signal import Signal
from sonar_vision.sonar import Sonar, PingResult, SOUND_SPEED_WATER
from sonar_vision.map import SpatialMap, Obstacle, CellState
from sonar_vision.tracker import ObjectTracker, Track, Detection
from sonar_vision.display import SonarDisplay

__all__ = [
    "Signal",
    "Sonar",
    "PingResult",
    "SOUND_SPEED_WATER",
    "SpatialMap",
    "Obstacle",
    "CellState",
    "ObjectTracker",
    "Track",
    "Detection",
    "SonarDisplay",
]
__version__ = "0.1.0"
