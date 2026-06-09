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
        erng = random.Random(tcfg.seed)
        k = tcfg.step_eval_games
        self.step_train = erng.sample(split.train_answers, min(k, len(split.train_answers))) if k else []
        self.step_holdout = erng.sample(split.holdout, min(k, len(split.holdout))) if k else []
        self.big_train = erng.sample(split.train_answers, min(tcfg.eval_games, len(split.train_answers)))
        self.big_holdout = erng.sample(split.holdout, min(tcfg.eval_games, len(split.holdout)))
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
            f"  [eval {step}] opener-valid {opener_eval.valid_word_rate:.0%} | "
            f"opener-IG {opener_eval.info_gain:.2f} bits | "
            f"win train {train_eval.win_rate:.0%} / holdout {holdout_eval.win_rate:.0%} "
            f"(gap {train_eval.win_rate - holdout_eval.win_rate:+.0%})"
        )
