"""Sonar — ping/echo simulation and distance estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sonar_vision.signal import Signal


# Default speed of sound in water (m/s)
SOUND_SPEED_WATER: float = 1500.0
SOUND_SPEED_AIR: float = 343.0


@dataclass
class PingResult:
    """Result of a sonar ping.

    Attributes:
        distance: Estimated distance in meters.
        travel_time: Round-trip time in seconds.
        signal_strength: Received signal amplitude (0–1 normalised).
        snr_db: Signal-to-noise ratio in decibels.
        frequency: Ping frequency in Hz.
    """

    distance: float
    travel_time: float
    signal_strength: float
    snr_db: float
    frequency: float


@dataclass
class Sonar:
    """Active sonar model with configurable medium and transducer.

    Attributes:
        sound_speed: Speed of sound in the medium (m/s).
        frequency: Transducer centre frequency (Hz).
        pulse_duration: Transmit pulse length (seconds).
        max_range: Maximum detectable range (meters).
        beam_width: Transducer beam width (degrees).
        source_level: Transmit source level (arbitrary dB scale).
        noise_level: Ambient noise floor (arbitrary dB scale).
    """

    sound_speed: float = SOUND_SPEED_WATER
    frequency: float = 50000.0
    pulse_duration: float = 0.001
    max_range: float = 1000.0
    beam_width: float = 30.0
    source_level: float = 200.0
    noise_level: float = 60.0

    # ── transmit ──────────────────────────────────────────────────

    def generate_ping(self, sample_rate: float = 44100.0) -> Signal:
        """Generate a transmit pulse (tone burst)."""
        return Signal.sine(
            frequency=self.frequency,
            duration=self.pulse_duration,
            sample_rate=sample_rate,
        )

    # ── propagation ───────────────────────────────────────────────

    def round_trip_time(self, distance: float) -> float:
        """Compute round-trip travel time for a given *distance*."""
        return 2.0 * distance / self.sound_speed

    def distance_from_time(self, travel_time: float) -> float:
        """Estimate distance from measured round-trip *travel_time*."""
        return travel_time * self.sound_speed / 2.0

    def spreading_loss(self, distance: float) -> float:
        """Cylindrical + spherical spreading loss in dB (simplified)."""
        if distance <= 0:
            return 0.0
        return 20.0 * math.log10(max(distance, 1e-12))

    def absorption_loss(self, distance: float, absorption_db_km: float = 10.0) -> float:
        """Absorption loss in dB using a constant dB/km coefficient.

        The coefficient is configurable but does not vary with frequency in
        this simplified model; pass an appropriate value for the transducer
        frequency and water conditions.
        """
        return absorption_db_km * distance / 1000.0

    def total_loss(self, distance: float, absorption_db_km: float = 10.0) -> float:
        """Total one-way transmission loss (spreading + absorption) in dB."""
        return self.spreading_loss(distance) + self.absorption_loss(distance, absorption_db_km)

    # ── detect ────────────────────────────────────────────────────

    def ping(self, distance: float, target_strength: float = -20.0) -> PingResult:
        """Simulate a ping at *distance* and return detection result.

        Models two-way transmission loss (outgoing plus return path) plus
        target strength. Returns ``PingResult`` with a distance estimate
        that includes small stochastic jitter proportional to range.
        """
        if distance <= 0 or distance > self.max_range:
            return PingResult(
                distance=float("nan"),
                travel_time=float("nan"),
                signal_strength=0.0,
                snr_db=-float("inf"),
                frequency=self.frequency,
            )

        rtt = self.round_trip_time(distance)
        # Active sonar: sound travels to the target and back, so use two-way loss.
        two_way_loss = 2.0 * self.total_loss(distance)
        received_level = self.source_level - two_way_loss + target_strength
        snr = received_level - self.noise_level
        strength = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-(snr - 6.0) / 3.0))))

        # Small measurement jitter proportional to range
        import random
        rng = random.Random()
        jitter = rng.gauss(0, 0.005 * distance)
        estimated_distance = distance + jitter
        estimated_rtt = self.round_trip_time(estimated_distance)

        return PingResult(
            distance=estimated_distance,
            travel_time=estimated_rtt,
            signal_strength=strength,
            snr_db=snr,
            frequency=self.frequency,
        )

    def ping_return_signal(
        self,
        distance: float,
        target_strength: float = -20.0,
        sample_rate: float = 44100.0,
    ) -> Signal:
        """Generate a synthetic echo return signal for a target.

        Returns a ``Signal`` containing transmit pulse, silence, and attenuated echo.
        """
        if distance <= 0 or distance > self.max_range:
            return Signal(sample_rate=sample_rate)

        tx = self.generate_ping(sample_rate)
        rtt_samples = int(self.round_trip_time(distance) * sample_rate)

        # Echo is attenuated by the two-way propagation path; target strength
        # increases received level, so it reduces the effective loss.
        loss_db = 2.0 * self.total_loss(distance) - target_strength
        gain = 10.0 ** (-loss_db / 20.0)
        gain = max(0.0, min(1.0, gain))
        echo_samples = [s * gain for s in tx.samples]

        # Build return: tx pulse → silence → echo
        silence_len = max(0, rtt_samples - len(tx.samples))
        combined = tx.samples + [0.0] * silence_len + echo_samples
        return Signal(samples=combined, sample_rate=sample_rate)

    # ── beam geometry ─────────────────────────────────────────────

    def in_beam(self, bearing_offset_deg: float) -> bool:
        """Check whether a target at *bearing_offset_deg* from boresight is within the beam."""
        return abs(bearing_offset_deg) <= self.beam_width / 2.0

    def beam_coverage(self) -> float:
        """Return the angular coverage in degrees of the beam."""
        return self.beam_width
