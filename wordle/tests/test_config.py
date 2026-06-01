"""Tests for Pydantic config models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FinetuneConfig, PretrainConfig


class TestPretrainConfig:
    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
model:
  n_layers: 6
  n_heads: 8
  embed_dim: 256
training:
  learning_rate: 3e-4
  max_steps: 100
data:
  dataset: "test/dataset"
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = PretrainConfig.from_yaml(config_path)

        assert config.model.n_layers == 6
        assert config.model.n_heads == 8
        assert config.model.embed_dim == 256
        assert config.training.learning_rate == pytest.approx(3e-4)
        assert config.training.max_steps == 100
        assert config.data.dataset == "test/dataset"

    def test_float_coercion(self, tmp_path: Path) -> None:
        """YAML parses 3e-4 as string; Pydantic should coerce it to float."""
        yaml_content = """
training:
  learning_rate: 3e-4
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = PretrainConfig.from_yaml(config_path)

        assert isinstance(config.training.learning_rate, float)
        assert config.training.learning_rate == pytest.approx(3e-4)

    def test_defaults(self) -> None:
        config = PretrainConfig()
        assert config.model.context_len == 256
        assert config.model.dropout == pytest.approx(0.1)
        assert config.training.seed == 42
        assert config.data.val_fraction == pytest.approx(0.05)

    def test_model_dump_roundtrip(self) -> None:
        config = PretrainConfig()
        d = config.model_dump()
        assert d["model"]["n_layers"] == 6
        assert d["training"]["learning_rate"] == pytest.approx(3e-4)


class TestFinetuneConfig:
    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
rl:
  algorithm: reinforce
  learning_rate: 5e-5
  group_size: 4
reward:
  solved: 2.0
  invalid_word: -2.0
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = FinetuneConfig.from_yaml(config_path)

        assert config.rl.algorithm == "reinforce"
        assert config.rl.learning_rate == pytest.approx(5e-5)
        assert config.rl.group_size == 4
        assert config.reward.solved == pytest.approx(2.0)
        assert config.reward.invalid_word == pytest.approx(-2.0)

    def test_reward_has_no_new_info(self) -> None:
        config = FinetuneConfig()
        assert hasattr(config.reward, "no_new_info")
        assert config.reward.no_new_info == pytest.approx(0.0)

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        """Extra YAML fields like model.checkpoint should not crash."""
        yaml_content = """
model:
  checkpoint: some/path
  n_layers: 4
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(yaml_content)
        config = FinetuneConfig.from_yaml(config_path)
        assert config.model.n_layers == 4
