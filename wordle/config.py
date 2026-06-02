"""Pydantic config models for training scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path
from pydantic import BaseModel


class ModelConfig(BaseModel):
    n_layers: int = 6
    n_heads: int = 8
    embed_dim: int = 256
    context_len: int = 256
    dropout: float = 0.1


class PretrainTrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 5000
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 1000


class DataConfig(BaseModel):
    dataset: str = "roneneldan/TinyStories"
    word_list_repeats: int = 50
    val_fraction: float = 0.05


class PretrainConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: PretrainTrainingConfig = PretrainTrainingConfig()
    data: DataConfig = DataConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


class RLConfig(BaseModel):
    algorithm: str = "grpo"
    decoding: str = "constrained"
    action_space: str = "answers"  # "answers" (~2,300 words) or "all" (~13K)
    seed: int = 42
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 2000
    grad_clip: float = 1.0
    eval_interval: int = 100
    checkpoint_interval: int = 500
    group_size: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    baseline_momentum: float = 0.99
    batch_size: int = 8
    max_eval_games: int = 50


class RewardModelConfig(BaseModel):
    invalid_word: float = -1.0
    repeated_guess: float = -0.5
    contradicts_clues: float = -0.3
    no_new_info: float = 0.0
    green_letter: float = 0.2
    yellow_letter: float = 0.1
    solved: float = 1.0
    failed: float = -0.5


class FinetuneConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    rl: RLConfig = RLConfig()
    reward: RewardModelConfig = RewardModelConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
