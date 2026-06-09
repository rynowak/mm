"""Tiny end-to-end RL smoke test (CPU, toy words, both curriculum phases)."""

import json

import torch
from mm_model import GPT, GPTConfig, save_checkpoint
from mm_wordle import ConstraintTokenizer, PatternMatrix, load_full_word_set

from wordle3.config import FinetuneConfig, RLTrainingConfig
from wordle3.finetune import train
from wordle3.splits import Split


def _toy():
    words = load_full_word_set()[:48]
    split = Split(seed=0, holdout_frac=0.1, train_answers=words[:40], holdout=words[40:])
    return words, split, PatternMatrix.from_words(words)


def _tiny_checkpoint(tmp_path, name: str) -> str:
    tok = ConstraintTokenizer()
    model = GPT(GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=tok.vocab_size, context_len=128, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters())
    path = tmp_path / name / "model.pt"
    save_checkpoint(path, model, opt, 0, model.config)
    return str(path)


def _cfg(phase: int, max_turns: int) -> FinetuneConfig:
    return FinetuneConfig(
        training=RLTrainingConfig(
            max_steps=3,
            warmup_steps=1,
            eval_interval=2,
            checkpoint_interval=3,
            group_size=4,
            ppo_epochs=1,
            batch_size=2,
            curriculum_phase=phase,
            max_turns=max_turns,
            probe_top_k=20,
            step_metrics_interval=1,
            win_rate_interval=2,
            step_eval_games=2,
            eval_games=2,
        )
    )


def test_rl_phase1_runs(tmp_path):
    words, split, pm = _toy()
    ckpt = _tiny_checkpoint(tmp_path, "sft")
    run_dir = train(
        _cfg(phase=1, max_turns=2),
        checkpoint=ckpt,
        words=words,
        pattern_matrix=pm,
        split=split,
        run_dir=tmp_path / "rl1",
        device=torch.device("cpu"),
    )
    latest = json.loads((run_dir / "live" / "latest.json").read_text())
    assert "win_rate" in latest and "valid_word_rate" in latest
    assert list(run_dir.glob("checkpoint-*/model.pt"))
    assert list(run_dir.glob("eval-*/snapshot.json"))


def test_rl_phase2_with_opener_runs(tmp_path):
    words, split, pm = _toy()
    ckpt = _tiny_checkpoint(tmp_path, "sft")
    opener = _tiny_checkpoint(tmp_path, "opener")
    run_dir = train(
        _cfg(phase=2, max_turns=6),
        checkpoint=ckpt,
        opener_checkpoint=opener,
        words=words,
        pattern_matrix=pm,
        split=split,
        run_dir=tmp_path / "rl2",
        device=torch.device("cpu"),
    )
    assert list(run_dir.glob("checkpoint-*/model.pt"))
