# Architecture — Sonar Vision

> *Sonar signal processing pipeline: ping generation → propagation → echo detection → tracking → mapping.*

## Design Goals

1. **Simulate to understand** — Model the full acoustic path from ping to detection
2. **Real-time capable** — Pure Python with no external dependencies
3. **Composable** — Each stage is independent; swap components freely

## High-Level Overview

```
Ping (emit) ──▶ Propagation (spreading + absorption loss)
                     │
                     ▼
              Echo (return) ──▶ Detection (SNR threshold)
                     │
                     ▼
              ObjectTracker (multi-target gating + velocity estimation)
                     │
                     ▼
              SpatialMap (occupancy grid)
```

## Core Components

### Signal (`signal.py`)
Discrete-time signal with uniform sampling. Construction methods: `sine`, `chirp`, `noise`. Filters: lowpass, highpass. Utility: FFT, energy, envelope.

### Sonar (`sonar.py`)
Active sonar model. Configurable: sound speed, frequency, pulse duration, max range, beam width, source level, noise level. Methods: `ping(distance)`, `ping_return_signal()`, `round_trip_time()`, `spreading_loss()`, `absorption_loss()`, `total_loss()`, `in_beam()`, `beam_coverage()`.

### ObjectTracker (`tracker.py`)
Multi-object tracker with constant-velocity motion model. Gating-based association, timeout for lost tracks. Tracks maintain ID, position, velocity, detection count.

### SpatialMap (`map.py`)
Occupancy grid from sonar returns. Resolution-configurable. Methods: `add_obstacle(obstacle)`, `set_cell(x, y, state)`, `get_cell(x, y)`, `mark_free_ray(...)`, `ray_cast(...)`, `occupancy_count()`, `coverage()`.

## Data Flow

```
generate_ping() → sound travels → echo returns
  → snr_db computed from spreading + absorption + noise
  → Detection(x, y, confidence)
  → ObjectTracker.update(detections)
  → Tracks maintained with velocity prediction
  → SpatialMap updated with occupancy data
```

## Key Design Decisions

### Constant-Velocity Tracker
Simple but effective. Each track has (x, y, vx, vy). Prediction: x += vx * dt. Gating: new detections within Mahalanobis-like distance are associated.

### Pure Python
No numpy, no scipy. Keeps it zero-dependency. Signal processing is educational-grade.

## Dependencies

None. Pure Python 3.10+ standard library only.

## Extension Points

- **New signal filters** — Add methods to `Signal` class
- **New propagation models** — Replace spreading/absorption in `Sonar`
- **New tracking algorithms** — Implement tracker interface with Kalman filter, etc.

## See Also

- [GETTING_STARTED.md](./GETTING_STARTED.md) — Quick start
- [API_REFERENCE.md](./API_REFERENCE.md) — Full API
- [LOW_LEVEL.md](./LOW_LEVEL.md) — Internal details
