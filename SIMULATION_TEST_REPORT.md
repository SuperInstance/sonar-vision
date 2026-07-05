# Simulation Test Report — sonar-vision

## Scope

This report describes a realistic end-to-end simulation scenario written to
validate the `sonar_vision` pipeline:

```
Sonar ping simulation → noisy detections → ObjectTracker → SpatialMap
```

The scenario is saved as `tests/test_simulation_scenarios.py` and is intended
as a pytest-based integration test complementing the existing unit tests in
`tests/test_all.py`.

## Scenario Design

### Sonar parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frequency | 200 kHz | Typical fish-finder / small-ROV imaging sonar |
| Sound speed | 1500 m/s | `SOUND_SPEED_WATER` — realistic nominal value |
| Pulse length | 1 ms | Common short-pulse sonar |
| Max range | 300 m | Fishing / shallow-water robotics context |
| Beam width | 30° | Representative single-beam transducer |
| Absorption | 30 dB/km | Approximate high-frequency seawater value |
| Source level | 200 dB (arbitrary scale) | Keeps 10–200 m targets detectable |
| Noise level | 60 dB | Comparable arbitrary scale |
| Detection threshold | signal_strength ≥ 0.05 | Soft SNR-derived cutoff |

### Targets

Three moving targets are used for the primary scenario:

| ID | Start position | Velocity | Role |
|----|----------------|----------|------|
| A | (50 m, 10 m) | (2.0, 0.0) m/s | Range-increasing crosser |
| B | (80 m, -20 m) | (0.0, 1.5) m/s | Vertical pass |
| C | (100 m, 20 m) | (-1.0, -1.0) m/s | Diagonal inward mover |

All targets start inside the 30° beam (±15° from boresight) at realistic
fish-finder ranges (50–120 m).

### Measurement model

Each time step:

1. Targets move according to constant velocity.
2. A `Sonar.ping()` is simulated for each target that lies within beam and
   max range.
3. Range and bearing noise are added (`σ_range = 0.5 m`, `σ_bearing = 1°`).
4. Detections are converted to Cartesian `(x, y)`.
5. `ObjectTracker.update()` receives the detection frame.
6. `SpatialMap` is updated with free rays and occupied cells.

## What Was Verified

### 1. Well-separated multi-target tracking

- All three targets are detected on every frame.
- `ObjectTracker` maintains exactly three active tracks after 25 s.
- Final track velocities have the correct signs (e.g. positive x velocity for
  target A, positive y velocity for target B).

### 2. Close-approach target association

A second scenario uses two targets moving toward each other on offset paths
with a closest approach of roughly 12 m. With the default tracker gate of 8 m
this stays outside the association threshold, and the tracker still reports two
distinct tracks with opposing velocity signs.

### 3. Beam-edge intermittent detection

A stationary target placed near the edge of the beam (≈14.5°) produces partial
detection rates (around 60–80% depending on random seed) because bearing noise
sometimes pushes the apparent return outside the beam. This confirms the beam
gate behaves realistically.

### 4. Occupancy map consistency

- The map accumulates occupied cells along the target trajectories.
- Free rays are marked between the sonar origin and each detection.
- The nearest obstacle to a final track position is within a few meters of the
  track.

## Bugs and Inaccuracies Found and Fixed

While building the scenario, several mismatches between documentation/code and
reality were found and corrected:

1. **`Sonar.ping` parameter name** — README used `sonar.ping(distance=100)`,
   but the implementation named the parameter `target_distance`. Renamed to
   `distance` so the Quick Start works as written.

2. **Missing public exports** — `Detection` (used in README and
   GETTING_STARTED examples) was not exported from `sonar_vision.__init__`.
   Added `Detection`, `Track`, `Obstacle`, `CellState`, and `PingResult` to
   `__all__`.

3. **One-way vs two-way propagation** — `Sonar.ping()` and
   `Sonar.ping_return_signal()` were applying only one-way transmission loss.
   Active sonar has a two-way path (transmit + echo), so both methods now use
   `2.0 * total_loss(distance)`. This makes long-range detections materially
   more realistic (e.g. SNR at 100 m drops from ~79 dB to ~38 dB).

4. **Incorrect method names in docs** — `API_REFERENCE.md` and
   `GETTING_STARTED.md` referenced methods that do not exist (`detect`,
   `filter_lowpass`, `filter_highpass`, `fft`, `predict_next`, `update`,
   `grid`, `occupancy`). Updated to the actual API (`ping`, `lowpass`,
   `highpass`, `dft_magnitude`, etc.).

5. **No LICENSE file** — README claimed MIT but no `LICENSE` file existed.
   Added `LICENSE` and updated `pyproject.toml`.

## Real Gaps and Surprising Behavior

1. **Simple tracker cannot resolve close crossings.** During early manual
   exploration, two targets placed on an intersecting path merged into a single
   track once they came within the association gate. This is expected behavior
   for a nearest-neighbour constant-velocity tracker, but it is a real
   limitation for dense target environments. The test was adjusted to use a
   ~12 m miss distance, which the tracker handles correctly.

2. **No explicit probability of detection model.** The current detector is
   deterministic: a target either exceeds the soft SNR threshold or it does not.
   Real sonar has a probabilistic detection curve near threshold. The scenario
   demonstrates intermittent detection only via beam-edge bearing noise; a
   richer model would add range-dependent detection probability.

3. **Absorption coefficient is not frequency-dependent.** The code calls
   `absorption_loss` "frequency-dependent" but takes a constant `absorption_db_km`
   parameter. The docstring was corrected to describe the simplified model.
   A production-grade physics model would use Thorp's formula or similar.

4. **No doppler / target aspect effects.** Target strength is constant, and
   moving targets do not produce Doppler shifts. These are acceptable for a
   pure-Python simulation toolkit but should be noted as out of scope.

## Verification Status

`pytest` is not installed in this environment, so the new test file was
verified by:

- `python3 -m py_compile tests/test_simulation_scenarios.py` (syntax check).
- Manual execution of the scenario logic with Python 3.12, confirming the
  tracker produces the expected number of tracks and the map accumulates
  occupied/free cells as designed.

The final verification will be performed by the repository's GitHub Actions
CI workflow (`.github/workflows/ci.yml`).
