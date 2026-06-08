"""Tiny end-to-end pre-train smoke test (CPU, toy word set, a few steps).

Guards the training loop, the per-step metric trio, the UI live/snapshot writes,
and checkpoint round-tripping — per the repo guardrail (train via running tiny).
"""

import json

import torch
from mm_model import load_checkpoint
from mm_wordle import ConstraintTokenizer, PatternMatrix, load_full_word_set

from wordle3.config import ModelConfig, PretrainConfig, PretrainTrainingConfig
from wordle3.metrics import play_games, sample_openers
from wordle3.pretrain import train
from wordle3.splits import Split


def _toy_setup():
    words = load_full_word_set()[:48]
    split = Split(seed=0, holdout_frac=0.1, train_answers=words[:40], holdout=words[40:])
    pm = PatternMatrix.from_words(words)
    return words, split, pm


def _toy_config() -> PretrainConfig:
    return PretrainConfig(
        model=ModelConfig(n_layers=2, n_heads=2, embed_dim=32, context_len=128, dropout=0.0),
        training=PretrainTrainingConfig(
            batch_size=8,
            max_steps=4,
            warmup_steps=1,
            eval_interval=2,
            checkpoint_interval=4,
            amp=False,
            compile=False,
            step_metrics_interval=1,
            win_rate_interval=2,
            step_eval_games=3,
            eval_games=3,
        ),
    )


def test_pretrain_runs_and_writes_ui_files(tmp_path):
    words, split, pm = _toy_setup()
    run_dir = train(
        _toy_config(),
        words=words,
        pattern_matrix=pm,
        split=split,
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert run_dir == tmp_path

    # Live UI contract.
    latest = json.loads((tmp_path / "live" / "latest.json").read_text())
    for key in ("step", "loss", "valid_word_rate", "info_gain", "win_rate", "holdout_win_rate", "games"):
        assert key in latest
    assert 0.0 <= latest["valid_word_rate"] <= 1.0
    assert (tmp_path / "live" / "history.jsonl").exists()

    # A game replay must satisfy the dashboard schema (equal-length guesses/feedback, color strings).
    if latest["games"]:
        g = latest["games"][0]
        assert len(g["guesses"]) == len(g["feedback"])
        assert len(g["turn_rewards"]) == len(g["guesses"])  # dashboard shows these per guess
        for row in g["feedback"]:
            assert all(c in {"green", "yellow", "gray"} for c in row)

    # Eval snapshot with hold-out fields.
    snaps = list(tmp_path.glob("eval-*/snapshot.json"))
    assert snaps
    snap = json.loads(snaps[0].read_text())
    for key in ("step", "win_rate", "avg_guesses", "holdout_win_rate", "holdout_avg_guesses"):
        assert key in snap

    # Checkpoint round-trips and carries the V3 vocab.
    ckpts = list(tmp_path.glob("checkpoint-*/model.pt"))
    assert ckpts
    ckpt = load_checkpoint(ckpts[0], torch.device("cpu"))
    assert ckpt["config"]["vocab_size"] == ConstraintTokenizer().vocab_size == 265


def test_sample_openers_returns_rates():
    words, _, pm = _toy_setup()
    from wordle3.pretrain import build_model

    model = build_model(ModelConfig(n_layers=2, n_heads=2, embed_dim=32), ConstraintTokenizer().vocab_size)
    res = sample_openers(model, ConstraintTokenizer(), pm, torch.device("cpu"), n_samples=16)
    assert 0.0 <= res.valid_word_rate <= 1.0
    assert res.info_gain >= 0.0
    assert res.distinct_valid >= 0


def test_play_games_smoke():
    words, split, pm = _toy_setup()
    from wordle3.pretrain import build_model

    model = build_model(ModelConfig(n_layers=2, n_heads=2, embed_dim=32), ConstraintTokenizer().vocab_size)
    res = play_games(model, ConstraintTokenizer(), pm, split.holdout, torch.device("cpu"))
    assert 0.0 <= res.win_rate <= 1.0
    assert 0.0 <= res.valid_word_rate <= 1.0
    assert res.avg_guesses >= 0.0
