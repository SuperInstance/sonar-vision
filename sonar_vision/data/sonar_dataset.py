"""
SonarVideoDataset — PyTorch dataset for SonarVision training.

Loads timestamped sonar sweeps + multi-depth camera frames for self-supervised learning.

Directory structure expected:
    root/
      sonar/
        2024-06-15T10-30-00.npy     # Sonar sweep (bearing_bins, max_depth)
        2024-06-15T10-30-01.npy
      cameras/
        2024-06-15T10-30-00/
          5m.jpg                      # Camera at 5m depth
          10m.jpg
          15m.jpg
          20m.jpg
      detections/
        2024-06-15T10-30-00.json      # {"detections": [{"depth": 15.2, "bearing": 45.0, "intensity": -30.5}]}
      water/
        2024-06-15T10-30-00.json      # {"temperature": 12.0, "salinity": 34.5, "turbidity": 0.3}
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from sonar_vision.data.augmentation import (
    ColorShiftAugmentation,
    DepthJitter,
    SonarNoiseAugmentation,
    TurbidityAugmentation,
)


class SonarVideoDataset(Dataset):
    """Self-supervised SonarVision training dataset.
    
    Each sample contains:
    - Sonar intensity sweep
    - Camera frames at multiple depths
    - Sonar detections (for depth-weighted loss)
    - Water column parameters
    - Pre-computed depth weights
    """

    def __init__(
        self,
        root_dir: str,
        bearing_bins: int = 128,
        max_depth: int = 200,
        depth_sigma: float = 3.0,
        min_cameras: int = 1,
        transform=None,
        augment: bool = False,
        train: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.bearing_bins = bearing_bins
        self.max_depth = max_depth
        self.depth_sigma = depth_sigma
        self.min_cameras = min_cameras
        self.transform = transform
        self.augment = augment
        self.train = train

        # Augmentation
        self.sonar_noise = SonarNoiseAugmentation() if augment else None
        self.depth_jitter = DepthJitter(max_shift=0.5) if augment else None
        self.turbidity_aug = TurbidityAugmentation() if augment else None
        self.color_shift = ColorShiftAugmentation() if augment else None

        # Discover samples
        self.samples = self._discover_samples()
        if not self.samples:
            raise ValueError(f"No valid samples found in {root_dir}")

    def _discover_samples(self) -> List[Dict]:
        """Find all timestamped samples with at least min_cameras."""
        samples = []
        sonar_dir = self.root_dir / "sonar"
        cam_dir = self.root_dir / "cameras"
        det_dir = self.root_dir / "detections"

        if not sonar_dir.exists():
            return samples

        # Find all sonar files
        for sonar_file in sorted(sonar_dir.glob("*.npy")):
            timestamp = sonar_file.stem  # e.g., "2024-06-15T10-30-00"

            # Check for cameras
            cam_folder = cam_dir / timestamp
            if not cam_folder.exists():
                continue

            cam_files = list(cam_folder.glob("*.jpg")) + list(cam_folder.glob("*.png"))
            if len(cam_files) < self.min_cameras:
                continue

            # Parse camera depths from filenames
            camera_data = []
            for cf in cam_files:
                depth_m = self._parse_depth_from_filename(cf.stem)
                if depth_m is not None:
                    camera_data.append({"path": cf, "depth_m": depth_m})

            if len(camera_data) < self.min_cameras:
                continue

            # Load detections if available
            det_file = det_dir / f"{timestamp}.json"
            detections = []
            if det_file.exists():
                try:
                    with open(det_file) as f:
                        det_data = json.load(f)
                    detections = det_data.get("detections", [])
                except (json.JSONDecodeError, KeyError):
                    pass

            # Load water params if available
            water_file = self.root_dir / "water" / f"{timestamp}.json"
            water_params = {"temperature": 12.0, "salinity": 35.0, "turbidity": 0.5}
            if water_file.exists():
                try:
                    with open(water_file) as f:
                        water_params.update(json.load(f))
                except (json.JSONDecodeError, KeyError):
                    pass

            samples.append({
                "timestamp": timestamp,
                "sonar_path": sonar_file,
                "cameras": camera_data,
                "detections": detections,
                "water_params": water_params,
            })

        return samples

    @staticmethod
    def _parse_depth_from_filename(stem: str) -> Optional[float]:
        """Extract depth from filename like '5m', '10m', '15_2m'."""
        stem = stem.lower().replace("_", "")
        if "m" in stem:
            try:
                return float(stem.split("m")[0])
            except ValueError:
                return None
        # Try bare number
        try:
            return float(stem)
        except ValueError:
            return None

    def _compute_depth_weights(
        self,
        camera_depths: List[float],
        sonar_detections: List[Dict],
    ) -> np.ndarray:
        """Compute per-camera supervision weight based on sonar proximity."""
        weights = np.full(len(camera_depths), 0.01)  # minimum weight

        if not sonar_detections:
            return weights

        det_depths = np.array([d["depth"] for d in sonar_detections])

        for i, cam_d in enumerate(camera_depths):
            # Find closest detection
            depth_diffs = (det_depths - cam_d) ** 2
            min_diff = depth_diffs.min()
            # Exponential weight
            w = math.exp(-min_diff / (2 * self.depth_sigma ** 2))
            weights[i] = max(w, 0.01)

        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load sonar sweep
        sonar_intensity = np.load(sample["sonar_path"])  # (bearing_bins, max_depth)
        if sonar_intensity.shape != (self.bearing_bins, self.max_depth):
            # Resize to expected shape
            sonar_intensity = self._resize_sonar(sonar_intensity)

        # Normalize sonar to [0, 1]
        sonar_min, sonar_max = sonar_intensity.min(), sonar_intensity.max()
        if sonar_max > sonar_min:
            sonar_intensity = (sonar_intensity - sonar_min) / (sonar_max - sonar_min)
        else:
            sonar_intensity = np.zeros_like(sonar_intensity)

        # Apply sonar augmentation
        if self.sonar_noise is not None:
            sonar_intensity = self.sonar_noise(sonar_intensity)

        # Load camera frames
        camera_frames = []
        camera_depths = []
        for cam in sample["cameras"]:
            try:
                from PIL import Image
                img = Image.open(cam["path"]).convert("RGB")
                img = np.array(img).astype(np.float32) / 255.0  # [0, 1]

                # Apply color augmentation
                if self.color_shift is not None:
                    turb = sample["water_params"].get("turbidity", 0.5)
                    img = self.color_shift(img, turbidity=turb)

                camera_frames.append(img)
                depth = cam["depth_m"]
                if self.depth_jitter is not None:
                    depth = self.depth_jitter(depth)
                camera_depths.append(depth)
            except Exception:
                continue  # Skip corrupted images

        if not camera_frames:
            # Return dummy sample if all cameras failed
            camera_frames = [np.zeros((224, 224, 3), dtype=np.float32)]
            camera_depths = [10.0]

        # Compute depth weights
        depth_weights = self._compute_depth_weights(
            camera_depths, sample["detections"]
        )

        # Prepare detections tensor
        if sample["detections"]:
            det_array = np.array([
                [d["depth"], d.get("bearing", 0), d.get("intensity", -60)]
                for d in sample["detections"]
            ], dtype=np.float32)
        else:
            det_array = np.zeros((1, 3), dtype=np.float32)

        # Resize camera frames to uniform size
        from PIL import Image
        target_h, target_w = 384, 512
        resized_frames = []
        for frame in camera_frames:
            img_pil = Image.fromarray((frame * 255).astype(np.uint8))
            img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
            resized_frames.append(np.array(img_pil).astype(np.float32) / 255.0)

        return {
            "sonar_intensity": torch.from_numpy(sonar_intensity).float(),
            "camera_frames": torch.from_numpy(np.stack(resized_frames)).float(),
            "camera_depths": torch.tensor(camera_depths).float(),
            "sonar_detections": torch.from_numpy(det_array).float(),
            "depth_weights": torch.from_numpy(depth_weights).float(),
            "turbidity": torch.tensor(sample["water_params"].get("turbidity", 0.5)).float(),
            "timestamp": sample["timestamp"],
        }

    def _resize_sonar(self, sweep: np.ndarray) -> np.ndarray:
        """Resize sonar sweep to expected (bearing_bins, max_depth)."""
        from PIL import Image
        img = Image.fromarray(sweep)
        img = img.resize((self.max_depth, self.bearing_bins), Image.BILINEAR)
        return np.array(img)


def create_training_split(
    root_dir: str,
    train_ratio: float = 0.8,
    **kwargs,
) -> Tuple["SonarVideoDataset", "SonarVideoDataset"]:
    """Create train/val datasets with temporal split.
    
    Split by timestamp to avoid data leakage (consecutive frames are correlated).
    """
    all_samples = sorted(Path(root_dir).glob("sonar/*.npy"))
    n = len(all_samples)
    split_idx = int(n * train_ratio)

    # Create train/val by splitting the directory (temporal split)
    # We do this by creating symlink directories
    import tempfile
    train_root = tempfile.mkdtemp(prefix="sonar_train_")
    val_root = tempfile.mkdtemp(prefix="sonar_val_")

    for subdir in ["sonar", "cameras", "detections", "water"]:
        os.makedirs(os.path.join(train_root, subdir), exist_ok=True)
        os.makedirs(os.path.join(val_root, subdir), exist_ok=True)

    train_files = all_samples[:split_idx]
    val_files = all_samples[split_idx:]

    for f in train_files:
        _link_sample(f, all_samples, train_root, root_dir)
    for f in val_files:
        _link_sample(f, all_samples, val_root, root_dir)

    train_ds = SonarVideoDataset(train_root, augment=True, train=True, **kwargs)
    val_ds = SonarVideoDataset(val_root, augment=False, train=False, **kwargs)

    return train_ds, val_ds


def _link_sample(sonar_file: Path, all_files: List[Path], target_root: str, source_root: str) -> None:
    """Create symlinks for a sample in the target directory."""
    timestamp = sonar_file.stem
    src = Path(source_root)

    # Sonar file
    dst_sonar = Path(target_root) / "sonar" / sonar_file.name
    if not dst_sonar.exists():
        os.symlink(str(sonar_file), str(dst_sonar))

    # Camera folder
    src_cam = src / "cameras" / timestamp
    dst_cam = Path(target_root) / "cameras" / timestamp
    if src_cam.exists() and not dst_cam.exists():
        os.makedirs(dst_cam, exist_ok=True)
        for cf in src_cam.glob("*"):
            os.symlink(str(cf), str(dst_cam / cf.name))

    # Detections
    src_det = src / "detections" / f"{timestamp}.json"
    dst_det = Path(target_root) / "detections" / f"{timestamp}.json"
    if src_det.exists() and not dst_det.exists():
        os.symlink(str(src_det), str(dst_det))

    # Water params
    src_water = src / "water" / f"{timestamp}.json"
    dst_water = Path(target_root) / "water" / f"{timestamp}.json"
    if src_water.exists() and not dst_water.exists():
        os.symlink(str(src_water), str(dst_water))
