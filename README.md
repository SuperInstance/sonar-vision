# SonarVision ⚓

> **Depth sounder pings → self-supervised generative underwater video**

Adapts [LingBot-Map](https://github.com/SuperInstance/lingbot-map)'s Geometric Context Transformer (GCT) to convert marine sonar returns into predicted underwater video frames, with automatic self-supervision from multi-depth camera arrays.

## The Idea

When a sonar detects a fish at 15m depth, the camera positioned at 15m provides the ground truth image. No manual labeling required — the physics of the water column does the work.

```
Sonar: "Fish at 15.2m"
Camera@5m:  dark water    → w = 0.01 (wrong depth)
Camera@15m: ★ FISH ★     → w = 0.95 (ground truth!)
Camera@20m: dark water    → w = 0.01

Loss = Σ exp(-|d_cam - d_sonar|²/σ²) × L2(predicted, camera)
```

## Architecture

```
Sonar Hardware → NMEAInterpreter → SonarSweepEmbedding → SonarEncoder
                                                          ↓
                                                    Streaming GCT
                                                  (causal attention,
                                                   KV cache, 3D RoPE)
                                                          ↓
                                                   VideoDecoder
                                              (DPT head + color head)
                                                          ↓
                                                     RGB Frame (384×512)
                                                          ↓
                                              DepthWeightedLoss ← cameras
```

## Key Components

| Module | What it does |
|---|---|
| `SonarEncoder` | ViT-adapted encoder for acoustic sweeps (4 channels: intensity, gradient, depth-norm, accumulated) |
| `StreamingGCTAggregator` | LingBot-Map's GCT with causal temporal attention, KV cache, 3D RoPE (depth × bearing × time) |
| `VideoDecoder` | DPT multi-scale head → underwater video frames with depth-dependent color |
| `DepthWeightedLoss` | Self-supervision: cameras closest to sonar detections get highest weight |
| `WaterColumnModel` | Mackenzie sound speed, Beer-Lambert light attenuation, RGB color by depth |
| `UnderwaterColorHead` | Models the blue-green shift and red absorption with depth |

## Quick Start

```python
from sonar_vision.pipeline import SonarVision

# Create model
model = SonarVision(max_depth=200, bearing_bins=128, embed_dim=1024)

# Generate video from sonar
import torch
sonar = torch.randn(1, 128, 200)  # (batch, bearing_bins, max_depth)
output = model.generate(sonar)
frame = output["frame"]           # (1, 3, 384, 512)
depth = output["depth_map"]       # (1, 1, 384, 512)
```

### Streaming Inference

```python
cache = None
for sweep in sonar_stream:
    frame, depth, cache = model.generate_stream(sweep, cache=cache)
    # cache carries state across calls — real-time processing
```

### Training

```bash
python -m sonar_vision.train \
    --data_dir /data/sonar \
    --output_dir ./checkpoints \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --gradient_accumulation_steps 4
```

### Config-Based Training

```python
from sonar_vision.config import SonarVisionConfig, jetson_nx_config

# Load from YAML
cfg = SonarVisionConfig.from_yaml("configs/jetson-nx.yaml")

# Or use a preset
cfg = jetson_nx_config()
cfg.to_yaml("my_experiment.yaml")
```

## Preset Configs

| Config | Embed Dim | GCT Layers | Target | Use Case |
|---|---|---|---|---|
| `configs/default.yaml` | 1024 | 6 | x86 GPU | Full model, maximum quality |
| `configs/jetson-nx.yaml` | 768 | 4 | Jetson Orin NX | Optimized for 16GB edge |
| `configs/debug.yaml` | 256 | 2 | CPU | Fast iteration, testing |

## Data Format

```
data/
  sonar/
    2024-06-15T10-30-00.npy     # (128, 200) intensity array
  cameras/
    2024-06-15T10-30-00/
      5m.jpg                      # Camera at 5m depth
      10m.jpg
      15m.jpg
      20m.jpg
  detections/
    2024-06-15T10-30-00.json      # {"detections": [{"depth": 15.2, "bearing": 45, "intensity": -30.5}]}
  water/
    2024-06-15T10-30-00.json      # {"temperature": 12.0, "salinity": 34.5, "turbidity": 0.3}
```

## Deployment (Jetson)

```python
from sonar_vision.deploy import JetsonInference

engine = JetsonInference(
    model_path="checkpoints/best.pt",
    precision="fp16",
)

frame, depth, latency = engine.process_sweep(sonar_sweep)
stats = engine.get_stats()  # {"avg_fps": 12.5, "mean_latency_ms": 80.2}
```

## Project Structure

```
sonar_vision/
  pipeline.py          # Full SonarVision model (encoder → GCT → decoder)
  config.py            # YAML-based experiment configuration
  deploy.py            # TorchScript export, quantization, Jetson inference
  train.py             # Training loop (AMP, EMA, gradient accumulation)
  encoder/
    sonar_encoder.py   # SonarSweepEmbedding + SonarEncoder
  aggregator/
    gct.py             # Streaming GCT with KV cache, 3D RoPE, GQA
  decoder/
    video_decoder.py   # DPT head + UnderwaterColorHead
  supervision/
    depth_weighted_loss.py  # DepthWeightedLoss, TemporalConsistency
  water/
    physics.py         # Sound speed, light attenuation, NMEA parsing
  data/
    sonar_dataset.py   # SonarVideoDataset + train/val split
    augmentation.py    # Sonar noise, turbidity, color shift
    preprocessing.py   # NMEA parsing, detection extraction
  utils/
    visualization.py   # Sonar heatmap, detection overlay, comparison
configs/
  default.yaml         # Full model config
  debug.yaml           # Tiny model for testing
  jetson-nx.yaml       # Jetson Orin NX optimized
```

## Hardware Targets

| Target | GPU | VRAM | Expected FPS |
|---|---|---|---|
| Jetson Orin NX | 1024 CUDA cores | 16 GB | 10-15 fps |
| Jetson AGX Orin | 2048 CUDA cores | 64 GB | 20-30 fps |
| RTX 4090 | 16384 CUDA cores | 24 GB | 60+ fps |

## Adapted from LingBot-Map

| LingBot-Map | SonarVision |
|---|---|
| RGB video frames → ViT patch embed | Sonar pings → 4-channel acoustic embed |
| 3D RoPE (x, y, time) | 3D RoPE (depth, bearing, time) |
| Camera intrinsics for pose | Fixed sonar geometry |
| DPT head → 3D point cloud | DPT head → underwater video frames |
| Pretrained depth supervision | Self-supervised via camera array |

## License

MIT

## Acknowledgments

- [LingBot-Map](https://github.com/SuperInstance/lingbot-map) — Base GCT architecture
- [DINOv2](https://github.com/facebookresearch/dinov2) — Transfer learning backbone
- [DPT](https://github.com/isl-org/DPT) — Dense prediction head design
