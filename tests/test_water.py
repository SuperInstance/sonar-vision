"""Tests for water physics models."""

import math
import torch
import pytest

from sonar_vision.water.physics import WaterColumnModel, SonarBeamModel


class TestWaterColumnModel:
    def test_sound_speed(self):
        water = WaterColumnModel()
        speed = water.sound_speed(depth=10, temperature=12.0, salinity=35.0)
        # Mackenzie equation gives ~1489 m/s at these conditions
        assert 1480 < speed < 1500

    def test_sound_speed_increases_with_depth(self):
        water = WaterColumnModel()
        s_shallow = water.sound_speed(depth=5, temperature=12.0, salinity=35.0)
        s_deep = water.sound_speed(depth=100, temperature=12.0, salinity=35.0)
        assert s_deep > s_shallow

    def test_sound_speed_increases_with_salinity(self):
        water = WaterColumnModel()
        s_fresh = water.sound_speed(depth=10, temperature=12.0, salinity=0.0)
        s_salt = water.sound_speed(depth=10, temperature=12.0, salinity=35.0)
        assert s_salt > s_fresh

    def test_light_attenuation(self):
        water = WaterColumnModel()
        # At 10m, some light should still pass
        r, g, b = water.light_transmission(10)
        assert 0 < r < 1
        assert 0 < g < 1
        assert 0 < b < 1

    def test_blue_dominates_at_depth(self):
        water = WaterColumnModel()
        r, g, b = water.light_transmission(50)
        # Blue should penetrate deepest
        assert b > r

    def test_color_attenuation_returns_rgb(self):
        water = WaterColumnModel()
        rgb = water.color_at_depth(20, turbidity=0.5)
        assert len(rgb) == 3
        assert all(0 <= c <= 1 for c in rgb)


class TestSonarBeamModel:
    def test_beam_width(self):
        beam = SonarBeamModel(frequency_khz=200.0)
        # 200 kHz should have narrow beam
        width = beam.beam_width_degrees
        assert 0 < width < 30

    def test_range_resolution(self):
        beam = SonarBeamModel(frequency_khz=200.0)
        resolution = beam.range_resolution_m
        assert resolution > 0
