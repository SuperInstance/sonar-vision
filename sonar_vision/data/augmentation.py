"""
Data augmentation for SonarVision training.

Underwater-specific augmentations that model real-world variations:
- Sonar noise (electronic noise, dropout, speckle)
- Camera depth jitter (mounting tolerance)
- Turbidity simulation (particulate scattering)
- Underwater color shift (wavelength-dependent absorption)
"""

import numpy as np
from typing import Optional


class SonarNoiseAugmentation:
    """Add realistic sonar noise to intensity sweeps.
    
    Models:
    - Electronic noise (Gaussian, constant across depths)
    - Speckle noise (multiplicative, worse at range)
    - Ping dropout (random missing pings)
    """

    def __init__(
        self,
        gaussian_std: float = 0.02,
        speckle_std: float = 0.05,
        dropout_prob: float = 0.01,
    ):
        self.gaussian_std = gaussian_std
        self.speckle_std = speckle_std
        self.dropout_prob = dropout_prob

    def __call__(self, sonar: np.ndarray) -> np.ndarray:
        """Apply noise to sonar sweep (bearing_bins, max_depth)."""
        result = sonar.copy()

        # Gaussian noise
        noise = np.random.normal(0, self.gaussian_std, sonar.shape)
        result = result + noise

        # Speckle noise (multiplicative, increases with range)
        depth_grad = np.linspace(0, 1, sonar.shape[1]).reshape(1, -1)
        speckle = np.random.normal(1.0, self.speckle_std, sonar.shape)
        speckle = 1.0 + (speckle - 1.0) * (1 + depth_grad)
        result = result * speckle

        # Random ping dropout (zero out entire bearing columns)
        if self.dropout_prob > 0:
            mask = np.random.random(sonar.shape[0]) > self.dropout_prob
            result[~mask, :] = 0.0

        return np.clip(result, 0, 1)


class DepthJitter:
    """Jitter camera depth to simulate mounting imprecision.
    
    Real cameras on a downrigger sway ±0.5m in waves.
    """

    def __init__(self, max_shift: float = 0.5):
        self.max_shift = max_shift

    def __call__(self, depth: float) -> float:
        """Return jittered depth."""
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        return depth + shift


class TurbidityAugmentation:
    """Simulate varying water turbidity.
    
    Randomly applies a haze-like effect to camera frames,
    simulating different particulate concentrations.
    """

    def __init__(
        self,
        min_turbidity: float = 0.0,
        max_turbidity: float = 0.8,
        haze_intensity: float = 0.3,
    ):
        self.min_turbidity = min_turbidity
        self.max_turbidity = max_turbidity
        self.haze_intensity = haze_intensity

    def __call__(self, image: np.ndarray, turbidity: Optional[float] = None) -> np.ndarray:
        """Apply turbidity haze to image (H, W, 3) in [0, 1]."""
        if turbidity is None:
            turbidity = np.random.uniform(self.min_turbidity, self.max_turbidity)
        else:
            # Jitter the provided turbidity
            turbidity = turbidity + np.random.uniform(-0.1, 0.1)
            turbidity = np.clip(turbidity, 0, 1)

        if turbidity < 0.05:
            return image

        result = image.copy()
        h, w = result.shape[:2]

        # Create depth gradient (objects further away are more hazed)
        depth_grad = np.linspace(0, 1, h).reshape(-1, 1, 1)
        haze = turbidity * self.haze_intensity * depth_grad

        # Blue-green haze (underwater scattering color)
        haze_color = np.array([0.1, 0.4, 0.5])  # Blue-green
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - haze[:, :, 0]) + haze_color[c] * haze[:, :, 0]

        return np.clip(result, 0, 1)


class ColorShiftAugmentation:
    """Random underwater color shift.
    
    Models variation in:
    - Red absorption (varies with dissolved organic matter)
    - Blue-green cast (varies with water depth/type)
    - Overall brightness (surface light conditions)
    """

    def __init__(
        self,
        red_shift_range: float = 0.1,
        blue_shift_range: float = 0.05,
        brightness_range: float = 0.1,
    ):
        self.red_shift = red_shift_range
        self.blue_shift = blue_shift_range
        self.brightness = brightness_range

    def __call__(self, image: np.ndarray, turbidity: float = 0.5) -> np.ndarray:
        """Apply random color shift to image (H, W, 3) in [0, 1]."""
        result = image.copy()

        # Random brightness
        brightness = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        result = result * brightness

        # Red channel attenuation (increases with turbidity)
        red_factor = 1.0 - self.red_shift * turbidity * np.random.uniform(0.5, 1.5)
        result[:, :, 0] = result[:, :, 0] * max(red_factor, 0.1)

        # Blue channel boost
        blue_factor = 1.0 + self.blue_shift * np.random.uniform(-1, 2)
        result[:, :, 2] = np.clip(result[:, :, 2] * blue_factor, 0, 1)

        return np.clip(result, 0, 1)


class RandomFlipAugmentation:
    """Randomly flip sonar bearing (left/right symmetry)."""

    def __call__(self, sonar: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            return np.flip(sonar, axis=0).copy()
        return sonar


class CompositeAugmentation:
    """Combine multiple augmentations."""

    def __init__(self, augmentations: list):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data
