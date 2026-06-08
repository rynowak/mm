"""Tiny end-to-end SFT smoke test (CPU, toy word set, warm-started from a tiny ckpt)."""

import json

import torch
from mm_model import GPT, GPTConfig, save_checkpoint
from mm_wordle import ConstraintTokenizer, PatternMatrix, load_full_word_set

from wordle3.config import SFTConfig, SFTTrainingConfig
from wordle3.sft import generate_sft_dataset, train
from wordle3.splits import Split


def _toy():
    words = load_full_word_set()[:48]
    split = Split(seed=0, holdout_frac=0.1, train_answers=words[:40], holdout=words[40:])
    pm = PatternMatrix.from_words(words)
    return words, split, pm


def _tiny_checkpoint(tmp_path) -> str:
    tok = ConstraintTokenizer()
    model = GPT(GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=tok.vocab_size, context_len=128, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / "pre" / "model.pt"
    save_checkpoint(path, model, opt, 0, model.config)
    return str(path)


def _tiny_config() -> SFTConfig:
    return SFTConfig(
        training=SFTTrainingConfig(
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
            n_golden_games=40,
            replay_frac=0.5,
        )
    )


def test_generate_sft_dataset_produces_examples():
    words, split, pm = _toy()
    ds = generate_sft_dataset(pm, ConstraintTokenizer(), split.train_answers, n_games=30, seed=0)
    assert len(ds) > 0
    input_ids, target_ids, loss_mask = ds[0]
    assert target_ids.shape == input_ids.shape == loss_mask.shape
    assert loss_mask.sum().item() == 5.0  # one guess = 5 letters


def test_sft_runs_and_writes_ui(tmp_path):
    words, split, pm = _toy()
    ckpt = _tiny_checkpoint(tmp_path)
    run_dir = tmp_path / "sft"
    out = train(
        _tiny_config(),
        checkpoint=ckpt,
        words=words,
        pattern_matrix=pm,
        split=split,
        run_dir=run_dir,
        device=torch.device("cpu"),
    )
    assert out == run_dir
    latest = json.loads((run_dir / "live" / "latest.json").read_text())
    for key in ("step", "loss", "valid_word_rate", "info_gain", "win_rate", "holdout_win_rate", "games"):
        assert key in latest
    assert list(run_dir.glob("checkpoint-*/model.pt"))
    assert list(run_dir.glob("eval-*/snapshot.json"))
