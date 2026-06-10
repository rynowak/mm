"""Pydantic config models for V3 training scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class ModelConfig(BaseModel):
    n_layers: int = 8
    n_heads: int = 8
    embed_dim: int = 320
    context_len: int = 128
    dropout: float = 0.1


class PretrainTrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 20000
    grad_clip: float = 1.0
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    # Speed (CUDA-gated; fp32 eager fallback on MPS/CPU).
    amp: bool = True
    compile: bool = True
    # Per-step metric trio (valid-word rate / info gain / win rate), §5.9.
    # The cheap two (valid-word rate + opener info gain) log every step_metrics_interval;
    # win rate runs the game-rollout mini-eval every win_rate_interval (carried forward between).
    step_metrics_interval: int = 1
    win_rate_interval: int = 50
    step_eval_games: int = 16  # mini-eval games per split (train + hold-out); 0 disables win rate
    eval_games: int = 256  # larger eval at eval_interval
    # Constraint-conditioned retrieval objective (§12): teach (tight state -> answer)
    # over the full lexicon, alongside the marginal (empty prompt -> word) examples.
    retrieval_pretrain: bool = True
    games_per_word: int = 2  # decent rollouts per word to harvest tight states
    max_candidates: int = 3  # only keep states narrowed to <= this many candidates (unambiguous)


class PretrainConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    training: PretrainTrainingConfig = PretrainTrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


class SFTTrainingConfig(BaseModel):
    seed: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-4  # lower than pre-train (fine-tuning a warm start)
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 8000
    grad_clip: float = 1.0
    eval_interval: int = 1000
    checkpoint_interval: int = 4000
    amp: bool = True
    compile: bool = True
    step_metrics_interval: int = 1
    win_rate_interval: int = 50
    step_eval_games: int = 16
    eval_games: int = 256
    n_golden_games: int = 4000  # golden demonstration games to generate (train answers only)
    replay_frac: float = 0.05  # fraction of steps drawing word-only replay batches (N1)


class SFTConfig(BaseModel):
    # Architecture is taken from the pre-train checkpoint, so no model section here.
    training: SFTTrainingConfig = SFTTrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


class RLTrainingConfig(BaseModel):
    seed: int = 42
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 3000
    grad_clip: float = 1.0
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    group_size: int = 8  # guesses sampled per turn (GRPO group)
    clip_epsilon: float = 0.2
    kl_beta: float = 0.2
    ppo_epochs: int = 2
    batch_size: int = 8  # games per step
    curriculum_phase: int = 1  # 1 = openers (turns 1-2), 2 = mid/late (turns 3-6)
    max_turns: int = 2
    probe_top_k: int = 300  # bounds the best-available-word search (§5.7-A)
    step_metrics_interval: int = 10
    win_rate_interval: int = 50
    step_eval_games: int = 16
    eval_games: int = 256


class FinetuneConfig(BaseModel):
    # Architecture is taken from the input checkpoint, so no model section here.
    training: RLTrainingConfig = RLTrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
