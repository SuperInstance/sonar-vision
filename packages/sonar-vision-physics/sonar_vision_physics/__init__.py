"""SonarVision Physics — deterministic underwater sonar physics engine."""

from .physics import compute_physics, dive_profile, FluxPhysics, WATER_TYPES, SEDIMENT_NAMES
from .ray_tracer import SonarRayTracer

__version__ = "1.2.0"
__all__ = ["compute_physics", "dive_profile", "FluxPhysics", "SonarRayTracer", "WATER_TYPES", "SEDIMENT_NAMES"]
