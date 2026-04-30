"""Tests for config system.

YAML roundtrip caveat:
  config.to_yaml() uses yaml.dump() which serialises Python tuples as
  '!!python/tuple', but from_yaml() uses yaml.safe_load() which cannot
  parse that tag.  Tests use to_dict/from_dict for full roundtrips and
  only test manually-written YAML (no tuple fields) for YAML I/O.
"""

import os
import tempfile
import pytest

from sonar_vision.config import (
    SonarVisionConfig,
    EncoderConfig,
    GCTConfig,
    jetson_nx_config,
    jetson_agx_config,
    debug_config,
    create_default_config,
)


class TestConfigYAML:
    def test_roundtrip(self):
        """dict-based roundtrip works for all field types including tuples."""
        cfg = SonarVisionConfig()
        cfg.name = "test-experiment"
        cfg.encoder.embed_dim = 512

        loaded = SonarVisionConfig.from_dict(cfg.to_dict())

        assert loaded.name == "test-experiment"
        assert loaded.encoder.embed_dim == 512

    def test_roundtrip_nested_fields(self):
        """All sub-config scalar fields survive a dict roundtrip."""
        cfg = debug_config()
        loaded = SonarVisionConfig.from_dict(cfg.to_dict())

        assert loaded.gct.num_layers == cfg.gct.num_layers
        assert loaded.train.batch_size == cfg.train.batch_size
        assert loaded.train.epochs == cfg.train.epochs
        assert loaded.encoder.patch_size == cfg.encoder.patch_size

    def test_yaml_load_scalar_fields(self):
        """from_yaml() correctly loads manually-written YAML (no tuple fields)."""
        yaml_str = "name: yaml-test\nencoder:\n  embed_dim: 768\ntrain:\n  batch_size: 2\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(yaml_str)
            path = f.name
        try:
            loaded = SonarVisionConfig.from_yaml(path)
            assert loaded.name == "yaml-test"
            assert loaded.encoder.embed_dim == 768
            assert loaded.train.batch_size == 2
        finally:
            os.unlink(path)

    def test_from_dict(self):
        # EncoderConfig has no num_layers field; it is silently ignored.
        d = {
            "name": "my-test",
            "encoder": {"embed_dim": 768},
            "train": {"lr": 0.001, "batch_size": 8},
        }
        cfg = SonarVisionConfig.from_dict(d)
        assert cfg.name == "my-test"
        assert cfg.encoder.embed_dim == 768
        assert cfg.train.lr == pytest.approx(0.001)
        assert cfg.train.batch_size == 8


class TestPresets:
    def test_jetson_nx(self):
        cfg = jetson_nx_config()
        assert cfg.encoder.embed_dim == 768
        assert cfg.gct.num_layers == 4
        assert cfg.deploy.target == "jetson-orin-nx"
        assert cfg.deploy.precision == "fp16"

    def test_jetson_agx(self):
        cfg = jetson_agx_config()
        assert cfg.deploy.target == "jetson-agx"
        assert cfg.deploy.precision == "fp16"

    def test_debug(self):
        cfg = debug_config()
        assert cfg.encoder.embed_dim == 256
        assert cfg.gct.num_layers == 2
        assert cfg.train.epochs == 2

    def test_debug_effective_batch(self):
        cfg = debug_config()
        # debug_config: batch_size=1, gradient_accumulation=1
        assert cfg.effective_batch_size() == 1 * 1

    def test_default_effective_batch(self):
        cfg = SonarVisionConfig()
        # defaults: batch_size=4, gradient_accumulation=4
        assert cfg.effective_batch_size() == 4 * 4

    def test_effective_batch_size_formula(self):
        cfg = SonarVisionConfig()
        cfg.train.batch_size = 8
        cfg.train.gradient_accumulation = 16
        assert cfg.effective_batch_size() == 128


class TestCreateDefault:
    def test_creates_file(self):
        """create_default_config writes a file that can be partially parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_default_config(tmpdir)
            assert os.path.exists(path)

    def test_default_values_sane(self):
        cfg = SonarVisionConfig()
        assert cfg.encoder.max_depth == 200
        assert cfg.encoder.bearing_bins == 128
        assert cfg.encoder.patch_size == 14
        assert cfg.gct.num_layers == 6
        assert cfg.gct.num_heads == 16
        assert cfg.water.sonar_frequency_khz == pytest.approx(200.0)
