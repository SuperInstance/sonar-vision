"""SonarVision Physics — deterministic underwater sonar physics engine."""

from .physics import compute_physics, dive_profile, FluxPhysics, WATER_TYPES, SEDIMENT_NAMES

__version__ = "1.0.1"
__all__ = ["compute_physics", "dive_profile", "FluxPhysics", "WATER_TYPES", "SEDIMENT_NAMES"]
