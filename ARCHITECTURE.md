# SonarVision — Architecture

> Depth sounder pings → self-supervised generative underwater video
> Adapts LingBot-Map's Geometric Context Transformer (GCT) to the acoustic modality.

---

## System Diagram

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                  SonarVision Pipeline                    │
                     │                                                          │
  Sonar Hardware     │   NMEA Stream                                            │
  ($PSDVS / $SDDBT) ─►  NMEAInterpreter ──► sonar_to_image()                  │
                     │                         (bearing_bins × max_depth)      │
                     │                                ▼                         │
                     │   ┌──────────────────────────────────────────────────┐  │
                     │   │  SonarEncoder (SonarSweepEmbedding)               │  │
                     │   │  · 4-channel preprocess:                          │  │
                     │   │      ch0 raw intensity  ch1 depth gradient        │  │
                     │   │      ch2 spreading comp ch3 accumulated returns   │  │
                     │   │  · Conv2d patch embed (14×14)                     │  │
                     │   │  · Separable depth + bearing positional encoding  │  │
                     │   │  · Learnable scale token (prepended)              │  │
                     │   └──────────────────────┬───────────────────────────┘  │
                     │                          │ tokens (B, N+1, 1024)         │
                     │                          ▼                               │
                     │   ┌──────────────────────────────────────────────────┐  │
                     │   │  GCT Streaming Aggregator  ← LingBot-Map adapted  │  │
                     │   │  · Causal temporal attention                      │  │
                     │   │  · KV cache (sliding window, online inference)    │  │
                     │   │  · 3D RoPE (depth × bearing × time)               │  │
                     │   │  · Trajectory memory (ring buffer)                │  │
                     │   └──────────────────────┬───────────────────────────┘  │
                     │                          │ aggregated tokens              │
                     │                          ▼                               │
                     │   ┌──────────────────────────────────────────────────┐  │
                     │   │  VideoDecoder (DPT-style)                         │  │
                     │   │  · Token reassemble → (B, 256, 14, 14)           │  │
                     │   │  · Multi-scale feature fusion (4 levels)         │  │
                     │   │  · Progressive upsampling 14→224→384             │  │──► RGB frame
                     │   │  · UnderwaterColorHead (depth → R↓ B↑)          │  │    384×512
                     │   │  · DepthHead → depth map (self-supervision)      │  │
                     │   │  · TemporalRefinementModule (3D Conv)             │  │
                     │   └──────────────────────┬───────────────────────────┘  │
                     │                          │                               │
                     │   ┌──────────────────────▼───────────────────────────┐  │
                     │   │  DepthWeightedLoss  (training only)               │  │◄── cameras
                     │   │  w(cam) = exp(−|d_cam−d_sonar|² / 2σ²)           │  │    at depth
                     │   │  L = Σ_c w(c)·L1(pred, cam_c) + λ·MSE(depth)    │  │
                     │   └──────────────────────────────────────────────────┘  │
                     │                                                          │
                     └─────────────────────────────────────────────────────────┘

  WaterColumnModel (physics conditioning — always active)
  · Mackenzie sound-speed equation  · Francois-Garrison absorption
  · Beer-Lambert light attenuation  · Per-channel RGB transmission
```

---

## Data Flow

### Inference

```
NMEA sentences
  └─► NMEAInterpreter.parse_sonar_return()
        └─► sonar_to_image()  →  (128, 200) float32
              └─► SonarEncoder.forward()
                    └─► preprocess_sonar()  →  (B, 4, 128, 200)
                          └─► Conv2d patch embed  →  (B, 130, 1024)
                                └─► GCT Aggregator (KV cache update)
                                      └─► VideoDecoder.forward()
                                            └─► UnderwaterColorHead (depth-conditioned)
                                                  └─► (B, 3, 384, 512)  RGB frame
```

### Training (self-supervised)

```
SonarVideoDataset.__getitem__()
  ├─ sonar_intensity        (128, 200)        ← .npy file
  ├─ camera_frames          (C, 3, 384, 512)  ← cameras/{ts}/{depth}m.jpg
  ├─ camera_depths          (C,)              ← parsed from filename ("15m.jpg" → 15.0)
  ├─ sonar_detections       (D, 3)            ← detections/{ts}.json
  ├─ depth_weights          (C,)              ← Gaussian weights, precomputed
  └─ turbidity              scalar            ← water/{ts}.json

  → SonarVision.forward()
      ├─ encode + adapt tokens
      ├─ decode → predicted frame + depth map
      └─ DepthWeightedLoss.forward()
            loss = Σ_c w(c)·L1(pred, cam_c) + λ_d·MSE(depth_map, sonar_d)
                 + λ_t·TemporalL1(frames)
```

### Depth-weight formula

```
w(d_cam, d_sonar) = exp( −(d_cam − d_sonar)² / (2·σ²) )   σ = 3.0 m default

Example — fish detected at 15 m:
  camera@5m  → w ≈ 0.003   (ignored)
  camera@10m → w ≈ 0.072   (low)
  camera@15m → w = 1.000   ← ground truth
  camera@20m → w ≈ 0.072   (low)
```

---

## Training Pipeline

### Dataset layout

```
data/
  sonar/         YYYY-MM-DDTHH-MM-SS.npy   (128, 200) float32
  cameras/       YYYY-MM-DDTHH-MM-SS/
                   5m.jpg  10m.jpg  15m.jpg  20m.jpg
  detections/    YYYY-MM-DDTHH-MM-SS.json  {detections: [{depth, bearing, intensity}]}
  water/         YYYY-MM-DDTHH-MM-SS.json  {temperature, salinity, turbidity}
```

Train/val split: temporal (80/20 by timestamp, symlink-based) — prevents leakage.

### Augmentations (training only)

| Class | Models |
|---|---|
| `SonarNoiseAugmentation` | Gaussian + range-dependent speckle + ping dropout |
| `DepthJitter` | Camera mount sway ±0.5 m |
| `TurbidityAugmentation` | Depth-gradient blue-green haze |
| `ColorShiftAugmentation` | Depth-dependent R↓/B↑ + brightness jitter |
| `RandomFlipAugmentation` | Bearing axis flip (left/right symmetry) |

### Loss weights

| Term | Default λ | Purpose |
|---|---|---|
| Weighted L1 | 1.0 | Pixel reconstruction at matched depth |
| Depth consistency | 0.5 | MSE(depth_map, sonar reading) |
| Temporal L1 | 0.1 | Anti-flicker between frames |
| Turbidity-aware | adaptive | Forgives fine detail in murky water |

### Transfer learning from DINOv2

`SonarEncoder._load_pretrained()` maps DINOv2's `patch_embed.proj` (3-ch RGB) to the
4-channel acoustic projection by averaging across RGB and replicating for channel 4.
Freeze patch embed for early training; unfreeze for fine-tuning on domain data.

---

## GCT Adapter Design (from LingBot-Map)

LingBot-Map's GCT was built for RGB video → 3D geometry. SonarVision repurposes it
for streaming acoustic → visual generation.

### Structural differences

| Component | LingBot-Map | SonarVision |
|---|---|---|
| Input modality | RGB ViT patches (3 ch) | Sonar sweeps (4 ch: intensity, gradient, depth-norm, accum.) |
| Positional encoding | 2D sinusoidal (H×W) | Separable 1D depth + 1D bearing (different physical units) |
| Scale token | Scene scale estimator | Acoustic scale token (prepended to sequence) |
| Attention causality | Bidirectional | Causal (supports online streaming with KV cache) |
| RoPE dimensions | 2D spatial | 3D: depth × bearing × time |
| Memory | Camera pose history | Depth-bin detection history (ring buffer) |

### Integration point

```python
# pipeline.py:74 — current placeholder:
self.feature_adapter = nn.Sequential(
    nn.LayerNorm(embed_dim), nn.Linear(...), nn.GELU(), nn.Linear(...)
)

# Month 1 target — swap in:
from sonar_vision.aggregator import GCTStreamingAggregator
self.feature_adapter = GCTStreamingAggregator(
    embed_dim=embed_dim,
    num_heads=16,
    kv_cache_size=32,   # keep last 32 sweeps
    rope_dims=(depth_bins, bearing_bins, time),
)
```

### KV cache memory budget (Jetson)

```
32 sweeps × 130 tokens × 1024 dim × FP16 = 8.5 MB  (well within budget)
Eviction: oldest sweep → compressed summary token (256-dim projection)
```

---

## Jetson Deployment Plan

**Hardware**: NVIDIA Jetson Orin NX 16 GB
**Target**: ≥10 FPS at 384×512, <100 ms end-to-end, <15 W

### Optimization stages

```
Stage 1 — Model export
  · torch.compile(mode="reduce-overhead") for training host
  · TorchScript trace: encoder + decoder (stateless, traceable)
  · Aggregator stays eager/FP16 (stateful KV cache)

Stage 2 — Quantization
  · INT8 PTQ (torch.ao.quantization) on encoder + decoder
  · FP16 for aggregator (attention precision-sensitive)
  · Validate: SSIM delta < 5% vs FP32 baseline

Stage 3 — TensorRT
  · ONNX export → trtexec --fp16 --workspace=4096
  · Fuse UnderwaterColorHead into decoder TRT plan
  · Decoder TRT plan; encoder TRT plan; aggregator PyTorch FP16
```

### Runtime threads

```
CPU thread  → NMEA reader → ring buffer (lock-free)
GPU stream  → encoder TRT → aggregator FP16 → decoder TRT → RTSP/ROS2
```

### Memory budget (quantized)

| Module | Memory |
|---|---|
| Encoder (INT8) | ~180 MB |
| Aggregator FP16 + KV cache | ~120 MB |
| Decoder (INT8) | ~210 MB |
| Total | ~510 MB of 16 GB |

### Latency targets

| Stage | Budget |
|---|---|
| Encoder | <30 ms |
| Aggregator | <20 ms |
| Decoder | <40 ms |
| End-to-end | <100 ms |

---

## 3-Month Roadmap

```
Month 1 — Foundation & Data  (2026-05)
──────────────────────────────────────
  [M1.1] Collect first 50-sample dataset
         · Mount 3-4 cameras at 5, 10, 15, 20 m
         · NTP-sync sonar NMEA to camera timestamps
         · Validate NMEAInterpreter + sonar_to_image() on real hardware
  [M1.2] Verify data pipeline end-to-end
         · SonarVideoDataset loads, augments, weights correctly
         · Training loop smoke test — loss decreases on single batch
  [M1.3] Implement GCT aggregator (sonar_vision/aggregator/)
         · Port causal temporal attention from LingBot-Map
         · 3D RoPE for depth/bearing/time
         · Wire into pipeline.py feature_adapter slot
  [M1.4] Baseline training run (50 samples, 100 epochs)
         · Establish PSNR / SSIM baseline vs nearest camera

  ★ Milestone: model generates recognizable underwater frames from sonar

Month 2 — Quality & Quantization  (2026-06)
────────────────────────────────────────────
  [M2.1] Scale dataset to 500+ samples
         · Vary turbidity, depth range, species, time of day
  [M2.2] Tune loss hyperparameters (σ, λ_depth, λ_temporal)
         · Grid search on val set; target SSIM > 0.65
  [M2.3] Jetson optimization — Stage 1 + 2
         · torch.compile + TorchScript encoder/decoder
         · INT8 PTQ; validate quality delta < 5% SSIM
  [M2.4] Benchmark on Jetson hardware
         · Latency per stage; identify bottleneck component

  ★ Milestone: INT8 model runs ≥5 FPS on Jetson Orin NX

Month 3 — Deployment & Field Validation  (2026-07)
───────────────────────────────────────────────────
  [M3.1] TensorRT conversion (encoder + decoder FP16 plans)
  [M3.2] Full pipeline at ≥10 FPS on Jetson
         · CPU/GPU thread split; RTSP output stream
  [M3.3] Field validation
         · Deploy on vessel; compare live predictions vs camera footage
         · Collect 1 000+ sample dataset for v2 training
  [M3.4] Checkpoint release
         · Publish sonar-vision-base weights
         · Integration guide for fish-finder NMEA sources

  ★ Milestone: real-time onboard deployment + public checkpoint
```

---

## Module Reference

| Module | File | Role |
|---|---|---|
| `SonarVision` | `pipeline.py` | Top-level nn.Module; wires all components |
| `SonarEncoder` | `encoder/sonar_encoder.py` | Sonar sweep → patch tokens (ViT-style) |
| `SonarSweepEmbedding` | `encoder/sonar_encoder.py` | 4-ch preprocess + patch embed + pos. encoding |
| `VideoDecoder` | `decoder/video_decoder.py` | Tokens → RGB frame (DPT multi-scale) |
| `VideoDecoderSequence` | `decoder/video_decoder.py` | Streaming sequence with temporal refinement |
| `UnderwaterColorHead` | `decoder/video_decoder.py` | Depth-driven R↓ B↑ color correction |
| `TemporalRefinementModule` | `decoder/video_decoder.py` | 3D Conv temporal smoothing |
| `DepthWeightedLoss` | `supervision/depth_weighted_loss.py` | Self-supervised Gaussian depth-weight loss |
| `TemporalConsistencyLoss` | `supervision/depth_weighted_loss.py` | Frame-to-frame L1 anti-flicker |
| `TurbidityAwareLoss` | `supervision/depth_weighted_loss.py` | Turbidity-adaptive loss sigma |
| `WaterColumnModel` | `water/physics.py` | Mackenzie sound speed; Beer-Lambert attenuation |
| `SonarBeamModel` | `water/physics.py` | Beam geometry; sonar equation; range resolution |
| `NMEAInterpreter` | `water/physics.py` | NMEA parser; sonar image builder |
| `SonarVideoDataset` | `data/sonar_dataset.py` | Self-supervised dataset with depth weights |
| Augmentations | `data/augmentation.py` | Sonar noise, depth jitter, turbidity, color shift |
| Preprocessing | `data/preprocessing.py` | NMEA parsing, blob detection, calibration, save |
| GCT Aggregator | `aggregator/` | *(Month 1 target)* Streaming causal attention |
