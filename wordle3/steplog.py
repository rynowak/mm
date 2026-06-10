"""Shared per-step metric reporter for V3 training phases (§5.9).

Encapsulates the fixed (seeded) eval target sets and the carried win-rate state so
pre-train and SFT share one implementation of the trio + snapshot logic.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from wordle3.metrics import play_games, sample_openers
from wordle3.trainutil import write_live, write_snapshot

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from mm_model import GPT
    from mm_training import MetricsLogger
    from mm_wordle import ConstraintTokenizer, PatternMatrix

    from wordle3.config import PretrainTrainingConfig, RLTrainingConfig, SFTTrainingConfig
    from wordle3.splits import Split

# Eval targets are drawn with a FIXED seed DECOUPLED from the training seed.
# Sharing the training seed correlated the train-answer eval sample with the
# training-target stream, so the "train" eval landed on memorized answers and
# inflated train win rate ~2x. The hold-out set is the real generalization eval
# set (its words are never training answers, by construction); the train sample is
# only a secondary in-distribution monitor. Must differ from the training seed.
EVAL_SEED = 59297


def sample_eval_targets(
    split: Split, step_games: int, eval_games: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Fixed, training-decoupled eval target sets: (step_train, step_holdout, big_train, big_holdout).

    Hold-out sets are the explicit generalization eval (never trained); train sets
    are an in-distribution monitor. Deterministic across runs for comparability.
    """
    rng = random.Random(EVAL_SEED)
    step_train = rng.sample(split.train_answers, min(step_games, len(split.train_answers))) if step_games else []
    step_holdout = rng.sample(split.holdout, min(step_games, len(split.holdout))) if step_games else []
    big_train = rng.sample(split.train_answers, min(eval_games, len(split.train_answers)))
    big_holdout = rng.sample(split.holdout, min(eval_games, len(split.holdout)))
    return step_train, step_holdout, big_train, big_holdout


class MetricReporter:
    """Logs the per-step trio and writes eval snapshots for a training run."""

    def __init__(
        self,
        logger: MetricsLogger | None,
        run_dir: Path,
        model: GPT,
        tokenizer: ConstraintTokenizer,
        pattern_matrix: PatternMatrix,
        device: torch.device,
        tcfg: PretrainTrainingConfig | SFTTrainingConfig | RLTrainingConfig,
        split: Split,
    ) -> None:
        self.logger = logger
        self.run_dir = run_dir
        self.model = model
        self.tok = tokenizer
        self.pm = pattern_matrix
        self.device = device
        self.tcfg = tcfg
        # Decoupled from tcfg.seed (see EVAL_SEED) — hold-out is the generalization metric.
        self.step_train, self.step_holdout, self.big_train, self.big_holdout = sample_eval_targets(
            split, tcfg.step_eval_games, tcfg.eval_games
        )
        self.last_win = 0.0
        self.last_holdout_win = 0.0
        self.last_replays: list[dict] = []

    def maybe_step(self, step: int, loss_v: float) -> None:
        tcfg = self.tcfg
        if not tcfg.step_metrics_interval or step % tcfg.step_metrics_interval != 0:
            return
        opener = sample_openers(self.model, self.tok, self.pm, self.device)
        if self.step_train and tcfg.win_rate_interval and step % tcfg.win_rate_interval == 0:
            train_res = play_games(self.model, self.tok, self.pm, self.step_train, self.device)
            holdout_res = play_games(self.model, self.tok, self.pm, self.step_holdout, self.device)
            self.last_win, self.last_holdout_win = train_res.win_rate, holdout_res.win_rate
            self.last_replays = train_res.replays + holdout_res.replays
        row = {
            "step": step,
            "loss": loss_v,
            "valid_word_rate": opener.valid_word_rate,
            "info_gain": opener.info_gain,
            "win_rate": self.last_win,
            "holdout_win_rate": self.last_holdout_win,
        }
        if self.logger is not None:
            for name, val in row.items():
                if name != "step":
                    self.logger.log_scalar(f"step/{name}", val, step)
            self.logger.log_scalar("step/distinct_openers", opener.distinct_valid, step)
        write_live(self.run_dir, row, self.last_replays)

    def maybe_eval(self, step: int) -> None:
        tcfg = self.tcfg
        if step % tcfg.eval_interval != 0 and step != tcfg.max_steps:
            return
        opener_eval = sample_openers(self.model, self.tok, self.pm, self.device, n_samples=256)
        train_eval = play_games(self.model, self.tok, self.pm, self.big_train, self.device)
        holdout_eval = play_games(self.model, self.tok, self.pm, self.big_holdout, self.device)
        write_snapshot(self.run_dir, step, train_eval, holdout_eval, opener_eval)
        print(
            f"  [eval {step}] HOLDOUT win {holdout_eval.win_rate:.0%} (generalization) | "
            f"train win {train_eval.win_rate:.0%} (in-dist) | "
            f"opener-valid {opener_eval.valid_word_rate:.0%} IG {opener_eval.info_gain:.2f}"
        )
