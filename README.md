# SonarVision

> Depth sounder → generative underwater video with self-supervised multi-camera learning.

Adapts [LingBot-Map](https://github.com/SuperInstance/lingbot-map)'s Geometric Context Transformer to convert depth sounder returns into predicted underwater video. Multiple cameras at different depths provide automatic ground truth supervision — when the sonar shows a fish at 15m, the camera at 15m captures what should be generated. This self-supervision loop eliminates manual labeling.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           SonarVision Pipeline            │
                    │                                          │
  Depth Sounder ──► │  ┌─────────────┐   ┌──────────────────┐ │
  (ping stream)     │  │ Sonar       │   │ Depth-to-Feature │ │
                    │  │ Encoder     │──►│ Adapter          │ │
                    │  │ (ViT-B/14)  │   │ (patch embed +   │ │
                    │               │   │  positional enc) │ │
                    └─────────────┘   └────────┬─────────┘ │
                                               │             │
                                               ▼             │
                    ┌──────────────────────────────────────┐ │
                    │    Streaming Aggregator (GCT)        │ │
                    │    - Temporal causal attention        │ │
                    │    - KV cache for online inference    │ │
                    │    - 3D RoPE for depth consistency    │ │
                    │    - Trajectory memory                │ │
                    └──────────────┬───────────────────────┘ │
                                   │                         │
                    ┌──────────────┼───────────────────────┐ │
                    │              ▼                        │ │
                    │    ┌─────────────────┐               │ │
                    │    │ Video Decoder   │               │ │
                    │    │ (DPT Head →     │──► Predicted   │ │
                    │    │  Video Diffusion│    Underwater  │ │
                    │    │  Decoder)       │    Video Frame  │ │
                    │    └─────────────────┘               │ │
                    │              │                        │ │
                    │              ▼                        │ │
                    │    ┌─────────────────┐               │ │
                    │    │ Self-Supervision│               │ │
                    │    │ Loss Module     │◄── Camera at   │ │
                    │    │ (depth-weighted │    matched depth│ │
                    │    │  L1 + perceptual│               │ │
                    │    │  + depth L2)    │               │ │
                    │    └─────────────────┘               │ │
                    └──────────────────────────────────────┘
```

## Key Innovation: Automatic Self-Supervision

```
Time T:
  Sonar: "Fish arch at depth=15.2m, bearing=045°"
  Camera@5m:  dark water, no fish     → low loss weight (wrong depth)
  Camera@10m: dark water, no fish     → low loss weight
  Camera@15m: ★ FISH CAPTURED ★       → HIGH loss weight (ground truth!)
  Camera@20m: dark water, no fish     → low loss weight

  Loss = Σ_w(d) * L2(predicted(d), camera(d))
       where w(d) = exp(-|d_camera - d_sonar|² / σ²)
```

The model automatically learns what a sonar return "looks like" underwater by
matching predicted video frames against camera footage at the corresponding depth.

## How It Differs from LingBot-Map

| Aspect | LingBot-Map | SonarVision |
|--------|-------------|-------------|
| Input | RGB video frames | Sonar pings (depth/bearing/intensity) |
| Output | 3D point clouds | Underwater video frames |
| Supervision | Pretrained + sparse depth | Self-supervised (cameras) |
| Modality | Vision → 3D geometry | Acoustic → visual generation |
| Real-time | ~20 FPS streaming | Target: 10+ FPS on Jetson |
| Environment | Indoor/outdoor | Underwater (low light, turbidity) |

## Installation

```bash
git clone https://github.com/SuperInstance/sonar-vision.git
cd sonar-vision
pip install -e ".[dev]"
```

## Quick Start

```python
from sonar_vision import SonarVision

model = SonarVision.from_pretrained("superinstance/sonar-vision-base")

# Process a sonar ping stream
video_frames = model.generate(
    sonar_pings=sonar_data,     # (N, 4) - depth, bearing, intensity, beam_width
    camera_calibration=cams,    # Camera positions + intrinsics
    water_column_params=env,    # Temperature, salinity, turbidity
)
```

## Data Format

### Sonar Input
```python
# Each ping: (depth_m, bearing_deg, intensity_db, beam_width_deg)
sonar_pings = np.array([
    [15.2, 45.0, -30.5, 12.0],  # Fish arch at 15.2m
    [5.0,  45.0, -60.0, 12.0],  # Bottom at 5m
    [20.0, 45.0, -45.0, 12.0],  # Another contact at 20m
])
```

### Camera Input (for training)
```python
cameras = [
    {"depth_m": 5,  "bearing_deg": 45, "image": cam5_frame},
    {"depth_m": 10, "bearing_deg": 45, "image": cam10_frame},
    {"depth_m": 15, "bearing_deg": 45, "image": cam15_frame},
    {"depth_m": 20, "bearing_deg": 45, "image": cam20_frame},
]
```

## Hardware

- **Development**: Any CUDA GPU (8GB+ VRAM)
- **Edge deployment**: Jetson Orin NX (16GB) — target for onboard processing
- **Cameras**: 3-5 underwater cameras at staggered depths (5m intervals)
- **Sounder**: Any standard fish-finding sonar with NMEA output

## Project Structure

```
sonar_vision/
├── encoder/          # Sonar ping encoder (ViT adapted for acoustic data)
├── adapter/          # Depth-to-feature adapter (maps sonar space to ViT space)
├── aggregator/       # Streaming GCT aggregator (from lingbot-map)
├── decoder/          # Video decoder (DPT head + temporal diffusion)
├── supervision/      # Self-supervision loss module
├── water/            # Underwater physics (sound speed, attenuation, turbidity)
└── utils/            # NMEA parsing, camera calibration, visualization
```

## License

Apache 2.0 (inherits from LingBot-Map)

## Acknowledgments

- [LingBot-Map](https://github.com/SuperInstance/lingbot-map) — Geometric Context Transformer base architecture
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — DPT head inspiration
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Efficient KV cache attention
