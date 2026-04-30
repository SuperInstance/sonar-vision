"""
Preprocessing utilities for SonarVision data pipeline.

Handles NMEA sonar parsing, detection extraction, camera calibration,
and training data preparation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_nmea_sonar(nmea_file: str) -> List[Dict]:
    """Parse NMEA sonar sentences into timestamped sonar sweeps.
    
    Supports:
    - $SDDBT — Depth Below Transducer
    - $SDMTW — Water Temperature
    - Proprietary sonar returns: $PSDVS,depth,bearing,intensity,beam_width*checksum
    
    Returns list of dicts with 'timestamp', 'pings', 'water_temp'.
    Each ping: {'depth': float, 'bearing': float, 'intensity': float}
    """
    results = []
    current_sweep = None
    
    with open(nmea_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("$"):
                continue
            
            try:
                sentence_id = line.split(",")[0]
                
                if sentence_id == "$SDDBT":
                    # Depth Below Transducer
                    parts = line.split(",")
                    depth = float(parts[1]) if parts[1] else 0.0
                    if current_sweep is not None:
                        current_sweep["pings"].append({
                            "depth": depth,
                            "bearing": 0.0,  # DBT has no bearing
                            "intensity": -60.0,
                        })
                
                elif sentence_id == "$SDMTW":
                    # Water Temperature
                    parts = line.split(",")
                    temp = float(parts[1]) if parts[1] else 12.0
                    if current_sweep is not None:
                        current_sweep["water_temp"] = temp
                
                elif sentence_id.startswith("$PSDVS"):
                    # Proprietary sonar return
                    parts = line.split("*")[0].split(",")
                    if len(parts) >= 5:
                        depth = float(parts[1])
                        bearing = float(parts[2])
                        intensity = float(parts[3])
                        beam_width = float(parts[4])
                        
                        # Start new sweep
                        if current_sweep is not None and current_sweep["pings"]:
                            results.append(current_sweep)
                        
                        current_sweep = {
                            "timestamp": len(results),
                            "pings": [],
                            "water_temp": 12.0,
                        }
                        current_sweep["pings"].append({
                            "depth": depth,
                            "bearing": bearing,
                            "intensity": intensity,
                        })
            
            except (ValueError, IndexError, KeyError):
                continue
    
    if current_sweep is not None and current_sweep["pings"]:
        results.append(current_sweep)
    
    return results


def extract_detections(
    sonar_sweep: np.ndarray,
    threshold_db: float = -40.0,
    min_blob_size: int = 3,
) -> List[Dict]:
    """Extract target detections from a sonar sweep.
    
    Simple threshold-based detection: finds contiguous regions above
    threshold intensity and returns bounding boxes.
    
    Args:
        sonar_sweep: (bearing_bins, max_depth) intensity array
        threshold_db: Detection threshold (normalized 0-1, default maps to -40 dB)
        min_blob_size: Minimum contiguous pixels for a detection
    
    Returns:
        List of {'depth': float, 'bearing': float, 'intensity': float, 'size': int}
    """
    from PIL import Image
    
    # Normalize threshold from dB to [0, 1]
    threshold = (threshold_db + 80) / 80.0  # Map [-80, 0] dB to [0, 1]
    
    # Threshold
    binary = (sonar_sweep > threshold).astype(np.uint8)
    
    # Find connected components
    img = Image.fromarray(binary)
    detections = []
    
    # Simple blob detection: find peaks
    from scipy import ndimage
    labeled, num_features = ndimage.label(binary)
    
    for i in range(1, num_features + 1):
        mask = labeled == i
        size = mask.sum()
        if size < min_blob_size:
            continue
        
        # Get centroid
        coords = np.argwhere(mask)
        bearing_idx = int(coords[:, 0].mean())
        depth_idx = int(coords[:, 1].mean())
        
        # Map indices to physical coordinates
        bearing = (bearing_idx / sonar_sweep.shape[0]) * 180 - 90  # [-90, 90] deg
        depth = (depth_idx / sonar_sweep.shape[1]) * 200  # [0, 200] m
        intensity = sonar_sweep[mask].mean()
        
        detections.append({
            "depth": round(depth, 1),
            "bearing": round(bearing, 1),
            "intensity": round(intensity * 80 - 80, 1),  # Back to dB
            "size": int(size),
        })
    
    return detections


def calibrate_cameras(config_file: str) -> Dict[str, Dict]:
    """Load camera calibration from config file.
    
    Expected JSON format:
    {
        "cameras": [
            {
                "depth_m": 5.0,
                "bearing_deg": 0,
                "intrinsics": {
                    "fx": 800, "fy": 800,
                    "cx": 640, "cy": 480,
                    "width": 1280, "height": 960
                },
                "extrinsics": {
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0
                },
                "lens_distortion": [0.0, 0.0, 0.0, 0.0]
            },
            ...
        ],
        "sonar": {
            "max_depth_m": 200,
            "bearing_range_deg": [-90, 90],
            "bearing_bins": 128,
            "frequency_khz": 200
        },
        "sync": {
            "method": "ntp",
            "max_sync_error_ms": 100
        }
    }
    
    Returns parsed calibration dict.
    """
    with open(config_file) as f:
        config = json.load(f)
    return config


def sonar_to_image(
    pings: List[Dict],
    bearing_bins: int = 128,
    max_depth: int = 200,
    bearing_range: Tuple[float, float] = (-90, 90),
) -> np.ndarray:
    """Convert sonar pings to a 2D intensity image.
    
    Args:
        pings: List of {depth, bearing, intensity}
        bearing_bins: Number of bearing bins
        max_depth: Maximum depth in meters
        bearing_range: (min_deg, max_deg)
    
    Returns: (bearing_bins, max_depth) float array in [0, 1]
    """
    image = np.full((bearing_bins, max_depth), -80.0)  # Default: silence
    
    for ping in pings:
        depth_idx = int(ping["depth"] / max_depth * max_depth)
        bearing_norm = (ping["bearing"] - bearing_range[0]) / (bearing_range[1] - bearing_range[0])
        bearing_idx = int(bearing_norm * bearing_bins)
        
        if 0 <= depth_idx < max_depth and 0 <= bearing_idx < bearing_bins:
            image[bearing_idx, depth_idx] = ping["intensity"]
    
    # Normalize to [0, 1]
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)
    
    return image.astype(np.float32)


def save_sample(
    output_dir: str,
    timestamp: str,
    sonar_image: np.ndarray,
    camera_frames: Dict[float, np.ndarray],
    detections: List[Dict],
    water_params: Dict,
):
    """Save a training sample to disk in the expected format.
    
    Creates:
      output_dir/sonar/{timestamp}.npy
      output_dir/cameras/{timestamp}/{depth}m.jpg
      output_dir/detections/{timestamp}.json
      output_dir/water/{timestamp}.json
    """
    import cv2
    
    # Sonar
    sonar_path = Path(output_dir) / "sonar"
    sonar_path.mkdir(parents=True, exist_ok=True)
    np.save(sonar_path / f"{timestamp}.npy", sonar_image)
    
    # Cameras
    cam_path = Path(output_dir) / "cameras" / timestamp
    cam_path.mkdir(parents=True, exist_ok=True)
    for depth, frame in camera_frames.items():
        cv2.imwrite(str(cam_path / f"{int(depth)}m.jpg"), frame)
    
    # Detections
    det_path = Path(output_dir) / "detections"
    det_path.mkdir(parents=True, exist_ok=True)
    with open(det_path / f"{timestamp}.json", "w") as f:
        json.dump({"detections": detections}, f, indent=2)
    
    # Water params
    water_path = Path(output_dir) / "water"
    water_path.mkdir(parents=True, exist_ok=True)
    with open(water_path / f"{timestamp}.json", "w") as f:
        json.dump(water_params, f, indent=2)
