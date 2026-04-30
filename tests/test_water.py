"""Tests for water physics models."""

import math
import torch
import pytest

from sonar_vision.water.physics import WaterColumnModel, SonarBeamModel, NMEAInterpreter


class TestWaterColumnModel:
    def test_sound_speed(self):
        water = WaterColumnModel()
        depth = torch.tensor([10.0])
        speed = water.sound_speed(depth)
        # Mackenzie equation gives ~1489 m/s at these conditions
        assert 1480 < speed.item() < 1500

    def test_sound_speed_increases_with_depth(self):
        water = WaterColumnModel()
        s_shallow = water.sound_speed(torch.tensor([5.0]))
        s_deep = water.sound_speed(torch.tensor([100.0]))
        assert s_deep.item() > s_shallow.item()

    def test_sound_speed_with_temp_override(self):
        water = WaterColumnModel()
        s_default = water.sound_speed(torch.tensor([10.0]))
        s_warm = water.sound_speed(torch.tensor([10.0]), temperature=torch.tensor([20.0]))
        # Warmer water → faster sound
        assert s_warm.item() > s_default.item()

    def test_light_attenuation(self):
        water = WaterColumnModel()
        depth = torch.tensor([10.0])
        # Light should decrease with depth
        light = water.light_attenuation(depth)
        assert light.shape == (1,)

    def test_color_attenuation_vector(self):
        water = WaterColumnModel()
        depth = torch.tensor([20.0])
        rgb = water.color_attenuation_vector(depth)
        assert rgb.shape == (1, 3)
        # All values should be in [0, 1]
        assert (rgb >= 0).all()
        assert (rgb <= 1).all()

    def test_absorption_coefficient(self):
        water = WaterColumnModel()
        depth = torch.tensor([10.0])
        freq = torch.tensor([200.0])
        alpha = water.absorption_coefficient(depth, freq)
        assert alpha.shape == (1,)
        assert alpha.item() > 0


class TestSonarBeamModel:
    def test_range_resolution(self):
        beam = SonarBeamModel(frequency_khz=200.0)
        resolution = beam.range_resolution()
        assert resolution > 0

    def test_beam_footprint(self):
        beam = SonarBeamModel(frequency_khz=200.0)
        footprint = beam.beam_footprint(torch.tensor([50.0]))
        assert footprint.item() > 0

    def test_target_strength_to_intensity(self):
        beam = SonarBeamModel(frequency_khz=200.0)
        intensity = beam.target_strength_to_intensity(
            torch.tensor([-30.0]),
            torch.tensor([50.0]),
        )
        assert intensity.shape == (1,)


class TestNMEAInterpreter:
    def test_parse_sonar_return(self):
        result = NMEAInterpreter.parse_sonar_return("$PSDVS,15.2,45.0,-30.5,3.0*4A")
        assert result["depth"] == 15.2
        assert result["bearing"] == 45.0
        assert result["intensity"] == -30.5
