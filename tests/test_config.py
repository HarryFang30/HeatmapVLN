"""Tests for config loading and Pydantic validation."""

import glob

import pytest
import yaml
from pydantic import ValidationError

from src.config_schema import TrainConfig, validate_config


class TestLoadConfig:
    def test_load_config_parses_yaml(self, tmp_path):
        """load_config reads a YAML file and returns a dict."""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump({"seed": 123, "data": {"root": "/x", "image_size": [64, 64], "init_hm_size": [32, 32]}, "training": {"stages": [{"name": "s", "epochs": 1}]}, "log": {"out_dir": "/tmp"}}))

        from scripts.training.utils import load_config
        cfg = load_config(str(cfg_path), validate=False)
        assert cfg["seed"] == 123

    def test_load_config_validates_by_default(self, tmp_path):
        """load_config with validate=True rejects invalid configs."""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.dump({"data": {"root": "/x"}}))

        from scripts.training.utils import load_config
        with pytest.raises((ValidationError, KeyError)):
            load_config(str(cfg_path), validate=True)

    def test_load_config_materializes_schema_defaults(self, tmp_path):
        """Validated load_config should return defaults materialized into the dict."""
        cfg_path = tmp_path / "defaults.yaml"
        cfg_path.write_text(yaml.dump({
            "seed": 7,
            "data": {"root": "/x", "image_size": [64, 64], "init_hm_size": [32, 32]},
            "training": {"stages": [{"name": "s", "epochs": 1}]},
            "log": {"out_dir": "/tmp"},
        }))

        from scripts.training.utils import load_config
        cfg = load_config(str(cfg_path), validate=True)
        assert cfg["loss"]["trajectory_weight"] == 0.0
        assert cfg["optim"]["amp"] == "bf16"
        assert cfg["model"]["device"] == "cuda"


class TestPydanticValidation:
    def test_validate_minimal_config(self, minimal_cfg):
        """Minimal config fixture passes validation."""
        result = validate_config(minimal_cfg)
        assert isinstance(result, TrainConfig)
        assert result.seed == 42
        assert result.data.root == "/tmp/fake_data"

    def test_validate_all_real_configs(self):
        """Every YAML in configs/ passes validation."""
        config_files = sorted(glob.glob("configs/train_*.yaml"))
        assert len(config_files) > 0, "No config files found"
        for path in config_files:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            result = validate_config(cfg)
            assert isinstance(result, TrainConfig), f"Failed: {path}"

    def test_rejects_invalid_dataset_type(self, minimal_cfg):
        """Invalid dataset_type raises ValidationError."""
        minimal_cfg["data"]["dataset_type"] = "invalid_type"
        with pytest.raises(Exception, match="dataset_type"):
            validate_config(minimal_cfg)

    def test_rejects_wrong_type(self, minimal_cfg):
        """Wrong value type (str instead of int) raises ValidationError."""
        minimal_cfg["optim"]["batch_size"] = "eight"
        with pytest.raises(ValidationError):
            validate_config(minimal_cfg)

    def test_rejects_negative_batch_size(self, minimal_cfg):
        """Negative batch_size raises ValidationError."""
        minimal_cfg["optim"]["batch_size"] = -1
        with pytest.raises(Exception, match="batch_size"):
            validate_config(minimal_cfg)

    def test_rejects_bad_image_size(self, minimal_cfg):
        """image_size with wrong length raises ValidationError."""
        minimal_cfg["data"]["image_size"] = [256]
        with pytest.raises(Exception, match="Expected"):
            validate_config(minimal_cfg)

    def test_missing_required_field(self):
        """Missing required top-level 'data' key raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_config({"training": {"stages": [{"name": "s", "epochs": 1}]}, "log": {"out_dir": "/tmp"}})

    def test_extra_keys_in_strict_section(self, minimal_cfg):
        """Misspelled key in strict section (data) raises ValidationError."""
        minimal_cfg["data"]["roott"] = "/typo"
        with pytest.raises(ValidationError):
            validate_config(minimal_cfg)
