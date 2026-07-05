# LOW LEVEL — Sonar Vision

> *For contributors extending the sonar processing pipeline.*

## Internal Architecture

```
sonar-vision/
├── sonar_vision/
│   ├── __init__.py    # Package re-exports
│   ├── signal.py      # Signal generation and filtering
│   ├── sonar.py       # Active sonar model (ping, propagation)
│   ├── map.py         # Spatial occupancy grid
│   ├── tracker.py     # Multi-object tracking (gating, prediction)
│   └── display.py     # ASCII visualization utilities
├── tests/
│   ├── test_all.py              # Unit/integration tests
│   └── test_simulation_scenarios.py  # End-to-end simulation tests
├── pyproject.toml
└── README.md
```

## Key Internal Patterns

### Spreading Loss Model

```python
def spreading_loss(self, distance: float) -> float:
    if distance <= 0:
        return 0.0
    return 20.0 * math.log10(max(distance, 1e-12))
```

Uses a simplified 20 log₁₀(r) geometric spreading model.

### Absorption Loss Model

```python
def absorption_loss(self, distance: float, absorption_db_km: float = 10.0) -> float:
    return absorption_db_km * distance / 1000.0
```

The dB/km coefficient is configurable but constant; callers should choose a
value appropriate for their frequency and water conditions.

### Two-Way Propagation in `ping`

Active sonar uses a two-way path (transmit to target, echo back), so
`ping()` and `ping_return_signal()` apply `2.0 * total_loss(distance)`:

```python
two_way_loss = 2.0 * self.total_loss(distance)
received_level = self.source_level - two_way_loss + target_strength
```

### Object Tracker Gating

Detections are associated to tracks using a nearest-neighbour gate. A
detection within `association_gate` of a predicted track position is
associated. Unassociated detections become new tracks. Tracks not updated
within `lost_timeout` seconds are reported as lost by `lost_tracks()` and
can be removed with `prune_lost()`.

### Constant-Velocity Prediction

```python
predicted_x = track.x + track.vx * dt
predicted_y = track.y + track.vy * dt
new_vx = 0.5 * (dx / dt) + 0.5 * track.vx
new_vy = 0.5 * (dy / dt) + 0.5 * track.vy
```

Velocity is estimated from the latest displacement and exponentially
smoothed; speeds are clamped to `max_velocity`.

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover: ping simulation, two-way loss, signal generation and filtering,
object tracking association, spatial map updates, and end-to-end simulation
scenarios.

## Debugging

- `Sonar.ping()` returns `signal_strength` and `snr_db` for each ping.
- `ObjectTracker` exposes `tracks`, `active_tracks()`, and `lost_tracks()`.
- `SpatialMap` exposes `occupancy_count()` and `coverage()` for inspection.

## What Is Not Implemented

- Kalman or particle-filter tracking (constant-velocity only).
- Real-time NMEA depth-sounder integration.
- Multi-beam or hydrophone-array beamforming.
- Video prediction or camera-based features.
