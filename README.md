# SonarVision ⚓

> **Depth sounder pings → self-supervised generative underwater video**
> Sonar hears. Cameras see. The physics of the water column does the rest.

[![Landing Page](https://img.shields.io/badge/Live-Demo-blue)](https://superinstance.github.io/sonar-vision-landing/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What Is This?

SonarVision converts marine sonar depth sounder returns into predicted underwater video frames. When sonar detects a fish at 15 meters, the camera positioned at 15 meters provides the ground truth. **No manual labeling required.**

```
Sonar: "Fish at 15.2m"
Camera@5m:  dark water    → w = 0.01
Camera@15m: ★ FISH ★     → w = 0.95 (ground truth!)
Camera@20m: dark water    → w = 0.01

Loss = Σ exp(-|d_cam - d_sonar|²/σ²) × L2(predicted, camera)
```

## Quick Start

```bash
git clone https://github.com/SuperInstance/sonar-vision.git
cd sonar-vision
pip install torch numpy pillow pyyaml einops

python -c "
import torch
from sonar_vision.pipeline import SonarVision
model = SonarVision(max_depth=50, bearing_bins=32, embed_dim=256)
model.eval()
sonar = torch.randn(1, 32, 50)
output = model.generate(sonar)
print(f'Frame: {output[\"frame\"].shape}')  # (1, 3, 384, 512)
"
```

📖 **[Full Getting Started Guide](docs/GETTING_STARTED.md)**

## Documentation

| Doc | What it covers |
|-----|---------------|
| [**Getting Started**](docs/GETTING_STARTED.md) | Install, first inference, troubleshooting |
| [**Tutorials**](docs/TUTORIALS.md) | 6 step-by-step tutorials (prediction, data pipeline, cameras, LoRA, Jetson, federated) |
| [**Onboarding**](docs/ONBOARDING.md) | Contributing, code structure, PR process |
| [**Use Cases**](docs/USE_CASES.md) | Commercial fishing, aquaculture, AUVs, sustainability, market analysis |
| [**Hardware Guide**](docs/HARDWARE_GUIDE.md) | Sonar, cameras, Jetson, wiring diagrams, cost breakdown |
| [**Architecture**](ARCHITECTURE.md) | System design, GCT adapter, training pipeline, deployment plan |

## Architecture

```
Sonar → SonarEncoder (4-ch ViT) → GCT Stream (causal attn, KV cache) → VideoDecoder → RGB Frame
                                                                         ↓
                                                          DepthWeightedLoss ← cameras
```

| Module | What it does |
|--------|-------------|
| `SonarEncoder` | 4-channel ViT (intensity, gradient, depth-norm, accumulated) |
| `StreamingGCTAggregator` | Causal attention, KV cache, 3D RoPE, grouped-query attention |
| `VideoDecoder` | DPT head + UnderwaterColorHead (blue-green cast by depth) |
| `DepthWeightedLoss` | Automatic ground truth from camera array |
| `WaterColumnModel` | Unified model: Mackenzie sound speed, Francois-Garrison absorption, Jerlov light |
| `NightlyTrainer` | Automated daily LoRA training with quality scoring |
| `FederatedAggregator` | Opt-in data sharing with differential privacy |

## Real-World Applications

- 🐟 **Commercial fishing** — Species ID from sonar, bycatch reduction, catch forecasting
- 🏭 **Aquaculture** — Fish counting, health monitoring, feeding optimization ($910M market by 2026)
- 🤖 **AUVs** — Obstacle avoidance, SLAM, 3D reconstruction in zero-visibility water
- 🔬 **Marine research** — Habitat mapping, species surveys, environmental monitoring
- 🌊 **Sustainability** — IUU detection, catch monitoring, traceability

📖 **[Full Use Cases + Market Analysis](docs/USE_CASES.md)**

## Hardware Setup

Complete system from $500 (entry) to $5,000 (professional):

| Config | Hardware | FPS |
|--------|----------|-----|
| Entry | Used sonar + 2 cameras + Jetson Nano | 5-8 |
| Standard | Garmin sonar + 4 GoPros + Jetson Orin NX | 10-15 |
| Pro | Premium sonar + 4 GoPros + Jetson Orin NX | 10-15 |
| Research | Full system + Jetson AGX Orin | 20-30 |

📖 **[Full Hardware Guide + Wiring Diagrams](docs/HARDWARE_GUIDE.md)**

## Nightly Training

The system gets better every day. Mount cameras on your downrigger, let it run:

```bash
# Set up cron — trains automatically at 3am
0 3 * * * cd /opt/sonar-vision && python -m sonar_vision.nightly.cron \
    --data_dir /data/boat_data --output_dir /data/nightly
```

- Progressive epochs (3 → 10 as dataset grows)
- Automatic promotion/rollback based on quality score
- LoRA weights are tiny (~2MB) — fast training, easy sharing

📖 **[Tutorial: Training Your First LoRA](docs/TUTORIALS.md#tutorial-4-training-your-first-lora)**

## Federated Learning

Opt-in to share anonymized data for a global underwater vision model:

- **Privacy first:** Only noise-injected LoRA weights leave your boat (no raw images, no GPS)
- **Differential privacy:** Calibrated Gaussian noise (ε=1.0 budget)
- **Federated averaging:** Weighted by data quality × quantity
- **Regional tracking:** GOA, Bering, Gulf, Pacific NW

📖 **[Tutorial: Joining the Federated Network](docs/TUTORIALS.md#tutorial-6-joining-the-federated-network)**

## Project Structure

```
sonar_vision/
├── pipeline.py           # Main model (encoder → GCT → decoder)
├── config.py             # YAML experiment configs
├── deploy.py             # TorchScript, quantization, Jetson
├── train.py              # Training loop (AMP, EMA)
├── encoder/              # SonarSweepEmbedding + SonarEncoder
├── aggregator/           # GCT streaming (KV cache, 3D RoPE)
├── decoder/              # DPT head + UnderwaterColorHead
├── supervision/          # Self-supervision losses
├── water/                # Underwater physics (Francois-Garrison, Jerlov, Thermocline, Seabed)
├── data/                 # Dataset, augmentation, preprocessing
├── nightly/              # LoRA training, cron job
├── federated/            # Data anonymization, DP, FedAvg
├── utils/                # Visualization
└── tests/                # Unit tests
```

## Stats

| Metric | Value |
|--------|-------|
| Source files | 28 |
| Lines of Python | 5,000+ |
| Test files | 5 |
| Preset configs | 3 |
| PLATO knowledge rooms | 4 |
| Tutorials | 6 |
| MIT License | ✅ |

## Adapted from LingBot-Map

| LingBot-Map | SonarVision |
|---|---|
| RGB video → ViT patches | Sonar pings → 4-channel acoustic embed |
| 3D RoPE (x, y, time) | 3D RoPE (depth, bearing, time) |
| Camera intrinsics | Fixed sonar geometry |
| DPT → 3D point cloud | DPT → underwater video frames |
| Pretrained depth labels | Self-supervised via camera array |

## Fleet Integration

SonarVision is part of the [Cocapn](https://github.com/cocapn) ecosystem:
- 🏛️ **PLATO** — 4 knowledge rooms (architecture, physics, GCT, self-supervision)
- 🐚 **I2I** — Fleet communication bottles
- ⚡ **Flux** — NMEA preprocessing bytecode
- 🐙 **Git-Native** — Push protocol and recovery

## Contributing

Contributions welcome! See [docs/ONBOARDING.md](docs/ONBOARDING.md) for setup and PR process.

Good first issues:
- Add more NMEA sentence parsers ($SDDBT, $SDVHW, $SDMTW)
- Real data benchmarks on actual sonar recordings
- Jetson TensorRT INT8 quantization
- Documentation improvements and translations

## License

MIT — use it however you want.

---

*Sonar hears. Cameras see. The physics of the water column does the rest.*

*Part of the [SuperInstance](https://github.com/SuperInstance) fleet. Built by [Forgemaster ⚒️](https://github.com/SuperInstance/forgemaster).*
