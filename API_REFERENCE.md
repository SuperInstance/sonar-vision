# API Reference — Sonar Vision

> *Public Python API. Requires Python 3.10+.*

---

## `Sonar`

```python
class Sonar(
    sound_speed=1500.0,
    frequency=50000.0,
    pulse_duration=0.001,
    max_range=1000.0,
    beam_width=30.0,
    source_level=200.0,
    noise_level=60.0,
)
```

Active sonar model. Simulates transmit, two-way propagation loss, and echo detection.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate_ping` | `(sample_rate=44100) -> Signal` | Generate transmit tone burst |
| `ping` | `(distance, target_strength=-20.0) -> PingResult` | Simulate full ping → echo cycle |
| `ping_return_signal` | `(distance, target_strength=-20.0, sample_rate=44100) -> Signal` | Generate synthetic echo return waveform |
| `round_trip_time` | `(distance) -> float` | Compute 2-way travel time |
| `distance_from_time` | `(travel_time) -> float` | Estimate distance from RTT |
| `spreading_loss` | `(distance) -> float` | Geometric spreading loss (dB) |
| `absorption_loss` | `(distance, absorption_db_km=10.0) -> float` | Absorption loss with configurable dB/km |
| `total_loss` | `(distance, absorption_db_km=10.0) -> float` | One-way spreading + absorption loss (dB) |
| `in_beam` | `(bearing_offset_deg) -> bool` | Whether a bearing is inside the beam |
| `beam_coverage` | `() -> float` | Beam angular coverage (degrees) |

**PingResult:**

| Field | Type | Description |
|-------|------|-------------|
| `distance` | `float` | Estimated distance (m) |
| `travel_time` | `float` | Round-trip time (s) |
| `signal_strength` | `float` | Received amplitude (0–1) |
| `snr_db` | `float` | Signal-to-noise ratio (dB) |
| `frequency` | `float` | Ping frequency (Hz) |

---

## `Signal`

```python
class Signal(samples=[], sample_rate=44100.0)
```

Discrete-time signal with uniform sampling.

**Constructors:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `sine` | `(frequency, duration, sample_rate=44100, amplitude=1.0)` | Sine wave |
| `noise` | `(duration, sample_rate=44100, amplitude=1.0, seed=None)` | Uniform random noise |
| `chirp` | `(f0, f1, duration, sample_rate=44100, amplitude=1.0)` | Linear chirp |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `resample` | `(target_rate) -> Signal` | Linear-interpolation resample |
| `lowpass` | `(cutoff, order=5) -> Signal` | Moving-average low-pass filter |
| `highpass` | `(cutoff, order=5) -> Signal` | High-pass by subtracting low-pass |
| `bandpass` | `(low_cutoff, high_cutoff, order=5) -> Signal` | Low-pass then high-pass |
| `threshold` | `(level) -> Signal` | Zero samples below absolute threshold |
| `envelope` | `() -> Signal` | Sliding-maximum magnitude envelope |
| `rms` | `() -> float` | Root-mean-square amplitude |
| `peak` | `() -> float` | Peak absolute amplitude |
| `snr_db` | `(noise) -> float` | SNR in dB relative to another signal |
| `energy` | `() -> float` | Total signal energy |
| `dominant_frequency` | `() -> float` | Zero-crossing frequency estimate |
| `dft_magnitude` | `(n=None) -> list[float]` | Naïve DFT magnitude spectrum |

**Operators:** `+` (sample-rate-matched addition), `*` (scalar or signal multiplication).

---

## `ObjectTracker`

```python
class ObjectTracker(
    association_gate=5.0,
    max_velocity=10.0,
    lost_timeout=5.0,
    next_id=1,
)
```

Multi-object tracker with constant-velocity prediction and nearest-neighbour gating.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `update` | `(detections, now=None) -> list[int]` | Update with new detections; returns affected track IDs |
| `tracks` | property | All tracks |
| `get_track` | `(track_id) -> Track \| None` | Fetch a track by ID |
| `active_tracks` | `(now=None) -> list[Track]` | Tracks seen within `lost_timeout` |
| `lost_tracks` | `(now=None) -> list[Track]` | Tracks that have timed out |
| `prune_lost` | `(now=None) -> int` | Remove timed-out tracks |
| `predict` | `(track_id, dt) -> tuple[float, float] \| None` | Predict a track's future position |
| `predict_all` | `(dt) -> dict[int, tuple[float, float]]` | Predict all tracks |
| `nearest_track` | `(x, y, now=None) -> Track \| None` | Nearest active track |
| `tracks_in_radius` | `(x, y, radius, now=None) -> list[Track]` | Active tracks within radius |
| `speed` | `(track_id) -> float` | Track speed (m/s) |
| `heading` | `(track_id) -> float` | Track heading in degrees (0 = east, 90 = north) |

**Track fields:** `track_id`, `x`, `y`, `vx`, `vy`, `last_seen`, `detections`, `label`

**Detection fields:** `x`, `y`, `timestamp`, `confidence`, `label`

---

## `SpatialMap`

```python
class SpatialMap(width=100.0, height=100.0, resolution=1.0)
```

2-D occupancy grid centred at the origin.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_cell` | `(x, y) -> CellState` | State of cell at world coordinates |
| `set_cell` | `(x, y, state) -> None` | Set cell state |
| `add_obstacle` | `(obstacle) -> None` | Add obstacle and mark cells occupied |
| `remove_obstacle` | `(index) -> None` | Remove obstacle by list index |
| `clear` | `() -> None` | Clear all obstacles and reset grid |
| `is_occupied` | `(x, y) -> bool` | Whether a position is occupied |
| `is_free` | `(x, y) -> bool` | Whether a position is known free |
| `nearest_obstacle` | `(x, y) -> Obstacle \| None` | Nearest obstacle |
| `obstacles_in_radius` | `(x, y, radius) -> list[Obstacle]` | Obstacles within radius |
| `distance_to_nearest` | `(x, y) -> float` | Distance to nearest obstacle |
| `ray_cast` | `(ox, oy, angle_deg, max_dist) -> tuple[float, float] \| None` | Cast a ray until occupied |
| `mark_free_ray` | `(ox, oy, angle_deg, distance) -> None` | Mark cells free along a ray |
| `occupancy_count` | `() -> dict[CellState, int]` | Count cells by state |
| `coverage` | `() -> float` | Fraction of grid that is explored |

**Obstacle fields:** `x`, `y`, `radius`, `confidence`, `label`

**CellState:** `UNKNOWN`, `FREE`, `OCCUPIED`

---

## `SonarDisplay`

```python
class SonarDisplay(width=41, height=21, range_m=50.0)
```

ASCII rendering helpers for sweeps, occupancy maps, and tracks.

**Methods:** `render_sweep`, `render_map`, `render_tracks`

---

## Minimum Supported Python Version

Python 3.10+
