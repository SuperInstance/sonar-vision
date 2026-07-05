# PLUG_AND_PLAY — Sonar Vision

> **Signal processing and spatial awareness for agents — sonar ping simulation, object tracking, and spatial mapping.**

## What Is This?

A Python library for sonar-based perception. Simulates active sonar pings, processes echo returns, tracks moving objects across detections, and builds spatial maps from sonar data. Designed for marine robotics and agent-based systems.

## Why Should You Care?

- **Realistic sonar simulation** — Ping generation, propagation loss, echo detection
- **Multi-object tracking** — Constant-velocity prediction with gating
- **Spatial mapping** — Occupancy grid from sonar returns
- **Zero hardware needed** — Pure Python, no special drivers

## Quick Start

```bash
pip install sonar-vision
```

```python
from sonar_vision import Sonar, Signal, ObjectTracker

sonar = Sonar(frequency=50000)
ping = sonar.generate_ping()
result = sonar.ping(distance=100)
print(f"Distance: {result.distance}m, SNR: {result.snr_db}dB")
```

## ✨ Key Features

- Sonar ping/echo simulation with spreading loss and noise
- Signal generation (sine, chirp, noise) and filtering (lowpass, highpass)
- Constant-velocity multi-object tracking with gating and timeout
- Spatial occupancy grid mapping

## Next Steps

| Guide | What It Covers |
|-------|----------------|
| [`GETTING_STARTED.md`](./GETTING_STARTED.md) | Install, ping, track an object |
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Module design and data pipeline |
| [`API_REFERENCE.md`](./API_REFERENCE.md) | All public classes and methods |
| [`LOW_LEVEL.md`](./LOW_LEVEL.md) | Internals and testing |

## Status

**v0.1.0 — Active development.** Core sonar, signal, and tracker modules are stable.
