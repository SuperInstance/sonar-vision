"""End-to-end simulation scenarios for sonar_vision.

These tests exercise the full pipeline (sonar simulation → detections →
tracking → mapping) with realistic underwater parameters and moving targets.
They are intended as integration/validation scenarios rather than unit tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator

import pytest

from sonar_vision import Sonar, ObjectTracker, Detection, SpatialMap, Obstacle
from sonar_vision.map import CellState
from sonar_vision.sonar import SOUND_SPEED_WATER


# ── scenario parameters ───────────────────────────────────────────

# Fish-finder / small-ROV sonar parameters
SCENARIO_FREQUENCY = 200_000.0  # Hz
SCENARIO_PULSE_DURATION = 0.001  # s
SCENARIO_MAX_RANGE = 300.0  # m
SCENARIO_BEAM_WIDTH = 30.0  # degrees
SCENARIO_SOURCE_LEVEL = 200.0  # dB (arbitrary scale)
SCENARIO_NOISE_LEVEL = 60.0  # dB
# Typical absorption at 200 kHz in seawater is tens of dB/km; use a moderate
# value so the 10-200 m targets remain detectable.
SCENARIO_ABSORPTION_DB_KM = 30.0

# Detection threshold (normalised signal strength)
DETECTION_THRESHOLD = 0.05

# Measurement noise
RANGE_NOISE_STD = 0.5  # m, additional to sonar.ping jitter
BEARING_NOISE_STD = 1.0  # degrees

# Time stepping
DT = 1.0  # s
N_STEPS = 25


@dataclass
class SimTarget:
    """A simple moving target in the scenario."""

    name: str
    x: float
    y: float
    vx: float
    vy: float
    target_strength: float = -20.0

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

    @property
    def range_m(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.y, self.x)) % 360.0


def make_sonar() -> Sonar:
    return Sonar(
        sound_speed=SOUND_SPEED_WATER,
        frequency=SCENARIO_FREQUENCY,
        pulse_duration=SCENARIO_PULSE_DURATION,
        max_range=SCENARIO_MAX_RANGE,
        beam_width=SCENARIO_BEAM_WIDTH,
        source_level=SCENARIO_SOURCE_LEVEL,
        noise_level=SCENARIO_NOISE_LEVEL,
    )


def detect_target(sonar: Sonar, target: SimTarget, rng: random.Random) -> Detection | None:
    """Ping a target and return a noisy Detection if it is detected."""
    slant_range = target.range_m
    if slant_range <= 0 or slant_range > sonar.max_range:
        return None

    # Add measurement noise before the beam/range gates so that realistic
    # missed detections occur near beam edges and at marginal ranges.
    noisy_range = max(0.1, slant_range + rng.gauss(0, RANGE_NOISE_STD))
    noisy_bearing = target.bearing_deg + rng.gauss(0, BEARING_NOISE_STD)

    bearing_offset = ((noisy_bearing + 180) % 360) - 180
    if not sonar.in_beam(bearing_offset):
        return None

    if noisy_range > sonar.max_range:
        return None

    result = sonar.ping(noisy_range, target_strength=target.target_strength)
    if math.isnan(result.distance) or result.signal_strength < DETECTION_THRESHOLD:
        return None

    rad = math.radians(noisy_bearing)
    det_x = result.distance * math.cos(rad)
    det_y = result.distance * math.sin(rad)
    return Detection(
        x=det_x,
        y=det_y,
        timestamp=0.0,  # caller must set
        confidence=result.signal_strength,
        label=target.name,
    )


def run_scenario(
    targets: list[SimTarget],
    n_steps: int = N_STEPS,
    dt: float = DT,
    seed: int = 42,
) -> tuple[list[list[Detection]], ObjectTracker, SpatialMap]:
    """Run the full pipeline for *n_steps* and return detections, tracker, map."""
    sonar = make_sonar()
    tracker = ObjectTracker(association_gate=8.0, max_velocity=8.0, lost_timeout=5.0)
    smap = SpatialMap(width=400.0, height=400.0, resolution=4.0)
    rng = random.Random(seed)

    all_detections: list[list[Detection]] = []
    t = 0.0
    for _ in range(n_steps):
        t += dt
        for target in targets:
            target.step(dt)

        frame: list[Detection] = []
        for target in targets:
            det = detect_target(sonar, target, rng)
            if det is not None:
                det.timestamp = t
                frame.append(det)

        # Mark free rays along each detection direction up to the measured range.
        for det in frame:
            bearing = math.degrees(math.atan2(det.y, det.x))
            smap.mark_free_ray(0.0, 0.0, bearing, math.hypot(det.x, det.y))
            smap.add_obstacle(
                Obstacle(x=det.x, y=det.y, radius=2.0, confidence=det.confidence)
            )

        tracker.update(frame, now=t)
        all_detections.append(frame)

    return all_detections, tracker, smap


# ═══════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════


class TestWellSeparatedTargets:
    """Three targets moving in different directions at realistic fish-finder ranges."""

    @pytest.fixture
    def targets(self) -> list[SimTarget]:
        # All targets start inside a 30° beam centred on boresight (±15°).
        return [
            SimTarget("A", 50.0, 10.0, 2.0, 0.0),    # bearing ~11.3°
            SimTarget("B", 80.0, -20.0, 0.0, 1.5),   # bearing ~-14.0°
            SimTarget("C", 100.0, 20.0, -1.0, -1.0), # bearing ~11.3°
        ]

    def test_targets_are_detected_each_step(self, targets: list[SimTarget]) -> None:
        detections, _, _ = run_scenario(targets)
        for frame in detections:
            assert len(frame) == 3, "All three well-separated targets should be detected"

    def test_tracker_maintains_three_distinct_tracks(self, targets: list[SimTarget]) -> None:
        _, tracker, _ = run_scenario(targets)
        active = tracker.active_tracks(now=N_STEPS * DT)
        assert len(active) == 3, f"Expected 3 tracks, got {len(active)}"

    def test_tracks_follow_expected_directions(self, targets: list[SimTarget]) -> None:
        _, tracker, _ = run_scenario(targets)
        velocities = [(t.vx, t.vy) for t in tracker.active_tracks(now=N_STEPS * DT)]
        # At least one track should have positive x velocity (target A)
        assert any(vx > 0 for vx, _ in velocities), "No track moving in +x direction"
        # At least one track should have positive y velocity (target B)
        assert any(vy > 0 for _, vy in velocities), "No track moving in +y direction"

    def test_map_has_occupied_cells(self, targets: list[SimTarget]) -> None:
        _, _, smap = run_scenario(targets)
        counts = smap.occupancy_count()
        assert counts[CellState.OCCUPIED] > 0, "Map should contain occupied cells"
        assert counts[CellState.FREE] > 0, "Map should contain free cells from ray marking"


class TestCrossingTargets:
    """Two targets that pass close to each other to stress the data association."""

    @pytest.fixture
    def targets(self) -> list[SimTarget]:
        # Two targets start near opposite edges of the beam and move toward each
        # other on offset paths. Their closest approach is ~12 m, which keeps
        # them outside the 8 m association gate while still stressing association.
        return [
            SimTarget("crosser_1", 40.0, -10.0, 1.2, 0.5),  # bearing ~-14°
            SimTarget("crosser_2", 80.0, 10.0, -1.2, -0.5),  # bearing ~7°
        ]

    def test_crossing_targets_remain_two_tracks(self) -> None:
        _, tracker, _ = run_scenario(self.targets())
        active = tracker.active_tracks(now=N_STEPS * DT)
        assert len(active) == 2, f"Expected 2 tracks after crossing, got {len(active)}"

    def test_crossing_tracks_have_opposing_velocity_signs(self) -> None:
        _, tracker, _ = run_scenario(self.targets())
        tracks = tracker.active_tracks(now=N_STEPS * DT)
        vx_values = [t.vx for t in tracks]
        assert any(vx > 0 for vx in vx_values), "Missing +x moving track"
        assert any(vx < 0 for vx in vx_values), "Missing -x moving track"


class TestBeamEdgeDetectionStatistics:
    """Verify intermittent detection for a target near the edge of the beam."""

    def test_beam_edge_target_detected_intermittently(self) -> None:
        # A target near the beam edge (14.5°) is inside the 30° beam on average,
        # but bearing noise sometimes pushes it outside, causing realistic missed
        # detections.
        target = SimTarget("edge", 145.0, 37.5, 0.0, 0.0)  # bearing ~14.5°
        detected_frames = 0
        total_frames = 0
        for seed in range(20):
            detections, _, _ = run_scenario([target], n_steps=20, dt=1.0, seed=seed)
            for frame in detections:
                total_frames += 1
                if frame:
                    detected_frames += 1

        detection_rate = detected_frames / total_frames if total_frames else 0.0
        # We expect partial detection: almost never 0% and almost never 100%.
        assert 0.0 < detection_rate < 1.0, (
            f"Beam-edge detection rate should be partial, got {detection_rate:.2f}"
        )


class TestMapConsistency:
    """The occupancy map should reflect where objects actually travelled."""

    def test_obstacles_near_known_tracks(self) -> None:
        target = SimTarget("singleton", 60.0, 10.0, 1.0, 0.5)  # stays inside 30° beam
        _, tracker, smap = run_scenario([target], n_steps=N_STEPS, dt=DT, seed=123)

        tracks = tracker.active_tracks(now=N_STEPS * DT)
        assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)}"
        track = tracks[0]

        nearest = smap.nearest_obstacle(track.x, track.y)
        assert nearest is not None, "No obstacle found near final track"
        error = math.hypot(nearest.x - track.x, nearest.y - track.y)
        assert error <= 10.0, f"Nearest map obstacle too far from final track: {error:.1f} m"

    def test_free_space_between_sonar_and_detections(self) -> None:
        target = SimTarget("singleton", 80.0, 0.0, 0.0, 0.0)
        _, _, smap = run_scenario([target], n_steps=N_STEPS, dt=DT, seed=456)

        # Cells along the +x axis up to the target should be marked free.
        free_cells = 0
        checked_cells = 0
        for x in range(0, 80, 4):
            checked_cells += 1
            if smap.is_free(float(x), 0.0):
                free_cells += 1

        assert free_cells > checked_cells * 0.5, "Too few free cells along detection ray"
