# Sonar Vision — Signal Processing and Spatial Awareness for Agents

> **📚 Documentation:** [`PLUG_AND_PLAY.md`](./PLUG_AND_PLAY.md) · [`GETTING_STARTED.md`](./GETTING_STARTED.md) · [`ARCHITECTURE.md`](./ARCHITECTURE.md) · [`API_REFERENCE.md`](./API_REFERENCE.md) · [`LOW_LEVEL.md`](./LOW_LEVEL.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue?style=flat-square)](https://www.python.org/)

Sonar Vision is a pure-Python sonar simulation and perception toolkit. It
generates active sonar pings, simulates two-way acoustic propagation loss,
processes echo returns, tracks moving objects across detections, and builds
spatial occupancy maps from sonar data. It is intended for simulation,
prototyping, and educational use in marine robotics.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [What It Is Not](#what-it-is-not)
- [License](#license)

---

## Overview

- **Sonar ping/echo simulation** — Spreading loss, configurable absorption loss, noise
- **Signal generation and filtering** — Sine waves, chirps, noise, lowpass/highpass filters
- **Multi-object tracking** — Constant-velocity prediction with gating-based association
- **Spatial mapping** — Occupancy grids from sonar returns
- **Zero hardware required** — Pure Python, no special drivers or external dependencies

### Real-World Use Cases

1. **Marine robotics simulation** — Model sensor behavior before deploying hardware
2. **Training data generation** — Generate realistic sonar returns for ML model training
3. **Algorithm prototyping** — Test tracking and mapping pipelines in simulation

---

## Installation

```bash
pip install sonar-vision
```

**Requirements:** Python 3.10+

---

## Quick Start

```python
from sonar_vision import Sonar, ObjectTracker, Detection

# Create a sonar model
sonar = Sonar(frequency=50000, max_range=500)

# Ping a target at 100 meters
result = sonar.ping(distance=100)
print(f"Distance: {result.distance:.1f}m, SNR: {result.snr_db:.1f} dB")

# Track objects across multiple detections
tracker = ObjectTracker()
tracker.update([Detection(x=10, y=20)])
tracker.update([Detection(x=12, y=22)])
tracker.update([Detection(x=14, y=19)])
print(f"Tracks: {len(tracker.tracks)}")
```

---

## API Overview

### `Sonar`
Active sonar model. Configurable frequency, sound speed, beam width, source level, noise level.

| Method | Description |
|--------|-------------|
| `ping(distance)` | Simulate full ping → echo cycle |
| `ping_return_signal(distance)` | Generate synthetic echo waveform |
| `generate_ping(sample_rate)` | Generate transmit tone burst |
| `round_trip_time(distance)` | Compute 2-way travel time |
| `spreading_loss(distance)` | Geometric spreading loss (dB) |
| `absorption_loss(distance)` | Configurable absorption loss (dB) |

### `Signal`
Discrete-time signal processing. Constructors: `sine()`, `noise()`, `chirp()`. Filters: `lowpass()`, `highpass()`. Utilities: `dft_magnitude()`, `energy()`, `envelope()`.

### `ObjectTracker`
Multi-object tracker with constant-velocity motion model. Gating-based association, timeout for lost tracks.

### `SpatialMap`
Occupancy grid from sonar returns. Resolution-configurable. `add_obstacle()`, `set_cell()`, `get_cell()`, `ray_cast()`.

See [`API_REFERENCE.md`](./API_REFERENCE.md) for the full API.

---

## What It Is Not

- **Not a real-time hydrophone processing stack.** It simulates sonar physics
  in Python; it does not interface with audio hardware or NMEA devices.
- **Not a video or camera system.** There is no video prediction, multi-camera
  learning, or image processing in this codebase.
- **Not a high-fidelity ocean-acoustics solver.** Propagation models are
  intentionally simple (geometric spreading + configurable dB/km absorption)
  for fast simulation and prototyping.
- **Not a beamforming library.** Single-beam sonar is modelled; phased arrays
  and multi-beam reconstruction are out of scope.

---

## How It Fits

Sonar Vision is part of the [SuperInstance fleet](https://github.com/SuperInstance) ecosystem.

---

## Testing

```bash
pip install pytest
pytest tests/
```

---

## License

MIT © SuperInstance Labs — see [`LICENSE`](./LICENSE).
