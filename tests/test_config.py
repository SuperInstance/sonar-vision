"""Tests for config system."""

import os
import tempfile
import pytest

from sonar_vision.config import (
    SonarVisionConfig,
    EncoderConfig,
    GCTConfig,
    jetson_nx_config,
    debug_config,
    create_default_config,
)


class TestConfigYAML:
    def test_roundtrip(self):
        cfg = SonarVisionConfig()
        cfg.name = "test-experiment"
        cfg.encoder.embed_dim = 512

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            cfg.to_yaml(f.name)
            path = f.name

        try:
            loaded = SonarVisionConfig.from_yaml(path)
            assert loaded.name == "test-experiment"
            assert loaded.encoder.embed_dim == 512
        finally:
            os.unlink(path)

    def test_from_dict(self):
        d = {
            "name": "my-test",
            "encoder": {"embed_dim": 768, "num_layers": 4},
            "train": {"lr": 0.001, "batch_size": 8},
        }
        cfg = SonarVisionConfig.from_dict(d)
        assert cfg.name == "my-test"
        assert cfg.encoder.embed_dim == 768
        assert cfg.train.lr == 0.001


class TestPresets:
    def test_jetson_nx(self):
        cfg = jetson_nx_config()
        assert cfg.encoder.embed_dim == 768
        assert cfg.gct.num_layers == 4
        assert cfg.deploy.target == "jetson-orin-nx"
        assert cfg.deploy.precision == "fp16"

    def test_debug(self):
        cfg = debug_config()
        assert cfg.encoder.embed_dim == 256
        assert cfg.gct.num_layers == 2
        assert cfg.train.epochs == 2

    def test_debug_effective_batch(self):
        cfg = debug_config()
        assert cfg.effective_batch_size() == 1 * 1  # batch=1, accum=1

    def test_default_effective_batch(self):
        cfg = SonarVisionConfig()
        assert cfg.effective_batch_size() == 4 * 4  # batch=4, accum=4


class TestCreateDefault:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_default_config(tmpdir)
            assert os.path.exists(path)
            loaded = SonarVisionConfig.from_yaml(path)
            assert loaded.name == "sonarvision-default"
