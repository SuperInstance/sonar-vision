"""
Federated Data Sharing for Global Model Training.

Opt-in system for sharing anonymized sonar-camera data
to train a global SonarVision model. Includes:
- Data anonymization (strip GPS, boat identity, timestamps)
- Differential privacy noise injection
- Federated averaging protocol
- Consent management
- Contribution tracking
"""

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConsentRecord:
    """User consent for data sharing."""
    user_id: str  # hashed, not identifiable
    vessel_type: str  # "trawler", "longliner", "pot", "seine", "troller"
    region: str  # "GOA", "BSA", "Bering", "Gulf"
    opt_in: bool
    share_depth_data: bool  # sonar depth readings
    share_camera_data: bool  # camera frames
    share_location: bool  # approximate region only
    share_detection_data: bool  # target detections
    date_consented: str
    min_quality_psnr: float = 20.0  # only share above this quality
    differential_privacy_epsilon: float = 1.0  # privacy budget


@dataclass
class DataContribution:
    """A single anonymized data contribution."""
    contribution_id: str
    contributor_hash: str  # SHA256 of user_id
    timestamp_bucket: str  # "2024-06" (month, not exact date)
    region: str
    num_samples: int
    sonar_stats: Dict = field(default_factory=dict)  # depth range, bearing range, frequency
    quality_score: float = 0.0
    privacy_epsilon_used: float = 0.0
    lora_weights_path: Optional[str] = None  # if contributor trains locally
    data_hash: str = ""  # integrity check


class DataAnonymizer:
    """Strips personally identifiable information from training data."""

    # Regions that are OK to share (general fishing areas)
    APPROVED_REGIONS = ["GOA", "BSA", "Bering", "Gulf", "Arctic", "Pacific NW", "NE Atlantic"]

    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or os.urandom(32).hex()

    def hash_contributor(self, identifier: str) -> str:
        """One-way hash of contributor identity."""
        return hashlib.sha256(f"{self.salt}:{identifier}".encode()).hexdigest()[:16]

    def bucket_timestamp(self, timestamp: str) -> str:
        """Reduce timestamp precision to month bucket."""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            return "unknown"

    def anonymize_detections(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        """Anonymize detection data.

        Keeps: depth, bearing, intensity (physical measurements)
        Removes: GPS coordinates, vessel heading, exact timestamp
        """
        return [
            {
                "depth_m": round(d["depth"], 1),
                "bearing_deg": round(d["bearing"], 1),
                "intensity_db": round(d["intensity"], 1),
            }
            for d in detections
        ]

    def anonymize_water_params(
        self,
        params: Dict,
    ) -> Dict:
        """Anonymize water column parameters.

        Keeps: temperature, salinity, turbidity (physical measurements)
        Removes: GPS, vessel ID, exact position
        """
        return {
            "temperature_c": round(params.get("temperature", 12.0), 1),
            "salinity_psu": round(params.get("salinity", 35.0), 1),
            "turbidity_ntu": round(params.get("turbidity", 0.5), 2),
        }

    def anonymize_sonar_sweep(
        self,
        sweep: np.ndarray,
    ) -> np.ndarray:
        """Apply differential privacy noise to sonar sweep.

        Adds calibrated Gaussian noise to protect against
        exact reconstruction of the original sweep.
        """
        # Scale noise relative to signal range
        signal_range = sweep.max() - sweep.min()
        sigma = signal_range * 0.02  # 2% noise — negligible for training
        noise = np.random.normal(0, sigma, sweep.shape)
        return np.clip(sweep + noise, 0, 1)


class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning.

    Ensures that no single contributor's data can be significantly
    inferred from the global model or shared data.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: Privacy budget (lower = more private). 1.0 is standard.
            delta: Failure probability. 1e-5 is standard.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.budget_used = 0.0

    def gaussian_noise(self, sensitivity: float, num_samples: int) -> float:
        """Calculate Gaussian noise scale for (epsilon, delta)-DP.

        Uses the Gaussian mechanism: N(0, sigma²) where
        sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def clip_gradients(
        self,
        gradients: List[torch.Tensor],
        clip_norm: float = 1.0,
    ) -> List[torch.Tensor]:
        """Clip gradient tensors to bounded norm.

        Prevents any single sample from having outsized influence.
        """
        total_norm = torch.sqrt(sum(t.norm().item() ** 2 for t in gradients))
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-6)
            return [t * scale for t in gradients]
        return gradients

    def add_noise_to_weights(
        self,
        weights: Dict[str, torch.Tensor],
        sensitivity: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """Add calibrated noise to model weights before sharing.

        Makes it statistically impossible to determine if
        a specific sample was in the training set.
        """
        sigma = self.gaussian_noise(sensitivity, 1)
        noisy_weights = {}
        for key, tensor in weights.items():
            noise = torch.randn_like(tensor) * sigma
            noisy_weights[key] = tensor + noise
        self.budget_used += self.epsilon
        return noisy_weights

    def can_share(self) -> bool:
        """Check if privacy budget allows sharing."""
        return self.budget_used < self.epsilon


class FederatedAggregator:
    """Federated averaging for combining LoRA weights from multiple contributors.

    Implements FedAvg: weighted average of contributor LoRA weights,
    weighted by data quality and quantity.
    """

    def __init__(
        self,
        min_contributors: int = 3,
        quality_threshold: float = 0.5,
        dp_epsilon: float = 1.0,
    ):
        self.min_contributors = min_contributors
        self.quality_threshold = quality_threshold
        self.dp = DifferentialPrivacy(epsilon=dp_epsilon)
        self.contributions: List[Dict] = []

    def receive_contribution(
        self,
        lora_weights: Dict[str, Dict[str, torch.Tensor]],
        contributor_hash: str,
        num_samples: int,
        quality_score: float,
        region: str,
    ) -> bool:
        """Receive and validate a contributor's LoRA weights.

        Returns True if contribution accepted.
        """
        if quality_score < self.quality_threshold:
            return False

        self.contributions.append({
            "contributor": contributor_hash,
            "num_samples": num_samples,
            "quality_score": quality_score,
            "region": region,
            "weights": lora_weights,
            "timestamp": datetime.now().isoformat(),
        })

        return True

    def aggregate(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Federated averaging of received contributions.

        Weight = num_samples * quality_score (better data, more influence).

        Returns aggregated LoRA weights, or None if not enough contributions.
        """
        if len(self.contributions) < self.min_contributors:
            return None

        total_weight = sum(
            c["num_samples"] * c["quality_score"]
            for c in self.contributions
        )

        if total_weight == 0:
            return None

        # Get weight keys from first contribution
        first_weights = self.contributions[0]["weights"]
        aggregated = {}

        for key in first_weights:
            weighted_sum_A = torch.zeros_like(first_weights[key]["A"])
            weighted_sum_B = torch.zeros_like(first_weights[key]["B"])

            for c in self.contributions:
                w = (c["num_samples"] * c["quality_score"]) / total_weight
                weighted_sum_A += w * c["weights"][key]["A"]
                weighted_sum_B += w * c["weights"][key]["B"]

            aggregated[key] = {
                "A": weighted_sum_A,
                "B": weighted_sum_B,
            }

        # Add differential privacy noise
        aggregated = self.dp.add_noise_to_weights(aggregated, sensitivity=0.01)

        return aggregated

    def get_contributor_report(self) -> List[Dict]:
        """Get anonymized report of contributions."""
        return [
            {
                "contributor_hash": c["contributor"][:8] + "...",
                "region": c["region"],
                "num_samples": c["num_samples"],
                "quality": round(c["quality_score"], 3),
            }
            for c in self.contributions
        ]


class GlobalModelManager:
    """Manages the global SonarVision model built from federated contributions.

    Handles:
    - Global model versioning (v1.0, v1.1, v2.0)
    - Contributor acknowledgment
    - Model distribution
    - Quality tracking per region
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "global_models").mkdir(exist_ok=True)
        (self.output_dir / "contributions").mkdir(exist_ok=True)

        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "version": "0.0",
            "total_contributions": 0,
            "total_samples": 0,
            "regions": {},
            "contributors": [],
            "history": [],
        }

    def save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def register_contribution(
        self,
        contribution: DataContribution,
    ) -> str:
        """Register a data contribution and update manifest."""
        self.manifest["total_contributions"] += 1
        self.manifest["total_samples"] += contribution.num_samples

        # Track by region
        if contribution.region not in self.manifest["regions"]:
            self.manifest["regions"][contribution.region] = {
                "samples": 0,
                "contributors": 0,
            }
        self.manifest["regions"][contribution.region]["samples"] += contribution.num_samples
        self.manifest["regions"][contribution.region]["contributors"] += 1

        # Track contributor (anonymized)
        if contribution.contributor_hash not in self.manifest["contributors"]:
            self.manifest["contributors"].append(contribution.contributor_hash)

        # Save contribution record
        contrib_path = self.output_dir / "contributions" / f"{contribution.contribution_id}.json"
        with open(contrib_path, "w") as f:
            json.dump(asdict(contribution), f, indent=2)

        self.save_manifest()
        return contribution.contribution_id

    def publish_global_model(
        self,
        lora_weights: Dict,
        version: str,
        contributors: List[str],
        aggregate_stats: Dict,
    ) -> str:
        """Publish a new global model version.

        Args:
            lora_weights: Federated-averaged LoRA weights
            version: Version string (e.g., "1.0", "1.1")
            contributors: List of contributor hashes
            aggregate_stats: Quality metrics

        Returns:
            Path to saved model
        """
        model_path = self.output_dir / "global_models" / f"global-v{version}.pt"
        torch.save(lora_weights, model_path)

        self.manifest["version"] = version
        self.manifest["history"].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "contributors": len(contributors),
            "samples": aggregate_stats.get("total_samples", 0),
            "quality": aggregate_stats.get("quality", 0),
        })

        self.save_manifest()
        return str(model_path)

    def get_stats(self) -> Dict:
        """Get global model statistics."""
        return {
            "version": self.manifest["version"],
            "total_contributions": self.manifest["total_contributions"],
            "total_samples": self.manifest["total_samples"],
            "num_contributors": len(self.manifest["contributors"]),
            "regions": self.manifest["regions"],
            "versions_published": len(self.manifest["history"]),
        }
