# Quick Start — SonarVision

Get a predicted underwater video frame from sonar data in 5 minutes.

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ (CUDA optional, CPU works for testing)
- NumPy, Pillow

```bash
# Clone
git clone https://github.com/SuperInstance/sonar-vision.git
cd sonar-vision

# Install dependencies
pip install torch numpy pillow pyyaml einops

# Verify
python -c "import sonar_vision; print('SonarVision ready ⚓')"
```

## 5-Minute First Inference

```python
import torch
from sonar_vision.pipeline import SonarVision

# 1. Create model (debug config for CPU)
model = SonarVision(
    max_depth=50,       # 50m range
    bearing_bins=32,    # 32 bearing bins
    embed_dim=256,      # Small for testing
)
model.eval()

# 2. Generate synthetic sonar data
# Shape: (batch, bearing_bins, max_depth)
sonar = torch.randn(1, 32, 50)

# 3. Predict underwater video frame
with torch.no_grad():
    output = model.generate(sonar)

frame = output["frame"]       # (1, 3, 384, 512) — RGB video frame
depth = output["depth_map"]   # (1, 1, 384, 512) — predicted depth

print(f"Frame shape: {frame.shape}")
print(f"Depth shape: {depth.shape}")
print(f"Frame range: [{frame.min():.2f}, {frame.max():.2f}]")
```

**Expected output:**
```
Frame shape: torch.Size([1, 3, 384, 512])
Depth shape: torch.Size([1, 1, 384, 512])
Frame range: [-0.95, 0.87]
```

## Streaming Inference

Process sonar sweeps one at a time (for real-time use):

```python
cache = None
for i in range(10):
    sweep = torch.randn(1, 32, 50)  # One sonar sweep
    frame, depth, cache = model.generate_stream(sweep, cache=cache)
    print(f"Sweep {i}: frame {frame.shape}, cache entries: {cache.seq_len if cache else 0}")
```

## Training Mode (Self-Supervised)

```python
model.train()

sonar = torch.randn(2, 32, 50)              # 2 sweeps
cameras = torch.randn(2, 3, 3, 384, 512)    # 2 batches × 3 cameras × RGB
cam_depths = torch.tensor([[5.0, 15.0, 25.0],   # Camera depths
                            [5.0, 15.0, 25.0]])
detections = torch.tensor([[15.2, 0.0, -30.0],  # Sonar detections
                            [12.0, 45.0, -28.0]])

output = model(
    sonar_intensity=sonar,
    camera_frames=cameras,
    camera_depths=cam_depths,
    sonar_detections=detections,
)

print(f"Loss: {output['loss'].item():.4f}")
print(f"Details: {output['loss_dict']}")
```

## Preset Configs

```python
from sonar_vision.config import (
    SonarVisionConfig,
    jetson_nx_config,
    jetson_agx_config,
    debug_config,
)

# Small model for testing
cfg = debug_config()
print(f"Debug: embed_dim={cfg.encoder.embed_dim}, layers={cfg.gct.num_layers}")

# Jetson-optimized
cfg = jetson_nx_config()
print(f"Jetson NX: embed_dim={cfg.encoder.embed_dim}, layers={cfg.gct.num_layers}")

# Full model
cfg = SonarVisionConfig()
cfg.to_yaml("my_experiment.yaml")
```

## Using Config Files

```bash
# Train with a config
python -m sonar_vision.train \
    --data_dir ./data \
    --output_dir ./checkpoints \
    --config configs/jetson-nx.yaml

# Or use preset
python -m sonar_vision.train \
    --data_dir ./data \
    --output_dir ./checkpoints \
    --embed_dim 768 \
    --max_depth 200 \
    --bearing_bins 128
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: sonar_vision` | Run `pip install -e .` in the repo root |
| CUDA out of memory | Use `debug_config()` or reduce `batch_size` |
| `einops` not found | `pip install einops` |
| Slow on CPU | Model is designed for GPU. Use `debug_config()` for testing |
| Frame is all zeros | Check that sonar input has non-zero values |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Ensure `bearing_bins` and `max_depth` match config |

## What's Next?

- **[Tutorial: Your First Prediction](TUTORIALS.md#tutorial-1-your-first-sonar-prediction)** — Full walkthrough with real data
- **[Tutorial: Boat Data Pipeline](TUTORIALS.md#tutorial-2-setting-up-your-boats-data-pipeline)** — NMEA sonar integration
- **[Tutorial: Nightly LoRA Training](TUTORIALS.md#tutorial-4-training-your-first-lora)** — Automated daily improvement
- **[Hardware Guide](HARDWARE_GUIDE.md)** — Sonar, cameras, and Jetson setup
- **[Use Cases](USE_CASES.md)** — Real-world applications
