"""Config defaults + YAML round-trip for the bufo sample."""

from __future__ import annotations

from pathlib import Path

from bufo.config import BufoLoRAConfig

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lora-sd15.yaml"


def test_defaults_are_sane():
    cfg = BufoLoRAConfig()
    assert cfg.data.resolution == 512
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 16
    assert cfg.training.base_model.endswith("stable-diffusion-v1-5")
    assert cfg.training.grad_accum >= 1


def test_from_yaml_loads_shipped_config():
    cfg = BufoLoRAConfig.from_yaml(_CONFIG)
    assert cfg.training.max_steps == 1500
    assert cfg.training.checkpoint_interval == 500
    assert cfg.data.exclude_substrings == ["bigbufo_"]
    assert "to_q" in cfg.lora.target_modules
    # model_dump must round-trip for the run manifest.
    assert BufoLoRAConfig.model_validate(cfg.model_dump()) == cfg
