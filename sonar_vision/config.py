"""
SonarVision module.

SonarVision is an end-to-end deep learning framework for underwater sonar perception.
It ingests raw side-scan sonar ping streams and produces high-resolution bathymetric maps,
object detections, and semantic segmentations. Designed for edge deployment on Jetson
platforms as well as high-throughput GPU servers, SonarVision supports training,
inference, and deployment workflows.

This module implements the SonarVision configuration system.
Manages experiment configs for training, inference, and deployment.
Uses YAML for human-readable configuration with schema validation.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class EncoderConfig:
    max_depth: int = 200
    bearing_bins: int = 128
    patch_size: int = 14
    embed_dim: int = 1024
    pretrained: str = ""
    use_dino: bool = False


@dataclass
class GCTConfig:
    num_layers: int = 6
    num_heads: int = 16
    gqa_ratio: int = 4
    window_size: int = 32
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_cache_seq: int = 512


@dataclass
class DecoderConfig:
    output_resolution: Tuple[int, int] = (384, 512)
    num_upsample_stages: int = 3
    temporal_refine_layers: int = 2


@dataclass
class SupervisionConfig:
    depth_sigma: float = 3.0
    temporal_weight: float = 0.1
    turbidity_weight: float = 0.05


@dataclass
class DataConfig:
    root_dir: str = "./data"
    train_split: float = 0.8
    num_workers: int = 4
    sonar_channels: int = 4  # intensity, gradient, depth-norm, accumulated


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation: int = 4
    checkpoint_every: int = 1000
    log_every: int = 50
    sample_every: int = 500
    warmup_steps: int = 500
    resume: str = ""


@dataclass
class DeployConfig:
    target: str = "jetson-orin-nx"  # jetson-orin-nx, jetson-agx, x86-gpu
    precision: str = "fp16"  # fp32, fp16, int8
    max_latency_ms: float = 100.0
    max_memory_mb: float = 8192.0
    use_tensorrt: bool = True
    trt_workspace_mb: int = 2048


@dataclass
class WaterConfig:
    sonar_frequency_khz: float = 200.0
    default_salinity: float = 35.0
    default_temperature: float = 12.0


@dataclass
class SonarVisionConfig:
    """Complete SonarVision experiment configuration."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    gct: GCTConfig = field(default_factory=GCTConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    water: WaterConfig = field(default_factory=WaterConfig)

    # Experiment metadata
    name: str = "sonarvision-default"
    output_dir: str = "./output"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "SonarVisionConfig":
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "SonarVisionConfig":
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "SonarVisionConfig":
        """Recursively build config from nested dict."""
        cfg = cls()
        for key, val in d.items():
            if hasattr(cfg, key):
                current = getattr(cfg, key)
                if isinstance(current, (int, float, str, bool, tuple)) or current is None:
                    setattr(cfg, key, val)
                elif isinstance(val, dict):
                    for k, v in val.items():
                        if hasattr(current, k):
                            setattr(current, k, v)
        return cfg

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)

    def effective_batch_size(self) -> int:
        return self.train.batch_size * self.train.gradient_accumulation


# Preset configurations

def jetson_nx_config() -> SonarVisionConfig:
    """Optimized for Jetson Orin NX (16GB, 1024 CUDA cores)."""
    cfg = SonarVisionConfig()
    cfg.name = "sonarvision-jetson-nx"
    cfg.encoder.embed_dim = 768
    cfg.encoder.patch_size = 16
    cfg.gct.num_layers = 4
    cfg.gct.num_heads = 12
    cfg.gct.window_size = 24
    cfg.decoder.output_resolution = (288, 384)
    cfg.train.batch_size = 2
    cfg.deploy.target = "jetson-orin-nx"
    cfg.deploy.precision = "fp16"
    cfg.deploy.max_latency_ms = 100.0
    cfg.deploy.max_memory_mb = 8192.0
    cfg.deploy.use_tensorrt = True
    return cfg


def jetson_agx_config() -> SonarVisionConfig:
    """Full model for Jetson AGX Orin (64GB, 2048 CUDA cores)."""
    cfg = SonarVisionConfig()
    cfg.name = "sonarvision-jetson-agx"
    cfg.deploy.target = "jetson-agx"
    cfg.deploy.precision = "fp16"
    cfg.deploy.max_latency_ms = 50.0
    cfg.deploy.max_memory_mb = 16384.0
    return cfg


def debug_config() -> SonarVisionConfig:
    """Tiny model for fast iteration and testing."""
    cfg = SonarVisionConfig()
    cfg.name = "sonarvision-debug"
    cfg.encoder.embed_dim = 256
    cfg.encoder.patch_size = 16
    cfg.gct.num_layers = 2
    cfg.gct.num_heads = 4
    cfg.gct.window_size = 8
    cfg.decoder.output_resolution = (128, 172)
    cfg.train.batch_size = 1
    cfg.train.gradient_accumulation = 1
    cfg.train.epochs = 2
    cfg.data.num_workers = 0
    cfg.deploy.target = "x86-gpu"
    cfg.deploy.precision = "fp32"
    cfg.deploy.use_tensorrt = False
    return cfg


def create_default_config(output_dir: str) -> str:
    """Write default config YAML and return path."""
    cfg = SonarVisionConfig()
    path = os.path.join(output_dir, "config.yaml")
    cfg.to_yaml(path)
    return path
