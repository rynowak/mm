"""V3 Phase 1: word-only pre-training (empty constraint prompt).

Trains the model to generate every valid 5-letter word (all 14,855, hold-out
included) conditioned on the zero-constraint prompt. Logs the per-step trio
(valid-word rate / info gain / win rate) via a small mini-eval (§5.9) and writes
the dashboard's live + eval-snapshot files.

Usage:
    uv run python -m wordle3.pretrain --config wordle3/configs/pretrain-large.yaml
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mm_model import save_checkpoint
from mm_training import (
    MetricsLogger,
    RunManifest,
    clip_grad_norm,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from mm_wordle import ConstraintTokenizer, PatternMatrix, load_full_word_set
from torch.utils.data import DataLoader

from wordle3.config import PretrainConfig
from wordle3.data import ConstraintDataset, WordOnlyDataset, collate_padded, generate_retrieval_examples
from wordle3.splits import Split, load_split
from wordle3.steplog import MetricReporter
from wordle3.trainutil import autocast, build_model, infinite

if TYPE_CHECKING:
    from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 word-only pre-training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="runs/cache")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps (quick runs)")
    return parser.parse_args()


def train(
    config: PretrainConfig,
    *,
    words: list[str] | None = None,
    pattern_matrix: PatternMatrix | None = None,
    split: Split | None = None,
    run_dir: Path | None = None,
    device: torch.device | None = None,
    cache_dir: str = "runs/cache",
) -> Path:
    """Run word-only pre-training. Returns the run directory.

    Test/override hooks (``words``, ``pattern_matrix``, ``split``, ``run_dir``,
    ``device``) default to the real 14,855-word universe and canonical split.
    """
    tcfg = config.training
    seed_everything(tcfg.seed)
    device = device or get_device()
    tokenizer = ConstraintTokenizer()
    words = words if words is not None else load_full_word_set()
    split = split if split is not None else load_split()
    if pattern_matrix is None:
        print("Loading/building pattern matrix...")
        pattern_matrix = PatternMatrix.load_or_build(words, cache_dir)
    print(f"Device: {device}")

    if run_dir is None:
        logger = MetricsLogger(experiment="pretrain-v3")
        run_dir = logger.log_dir
    else:
        logger = None
    print(f"Run dir: {run_dir}")
    RunManifest.capture(
        experiment="pretrain-v3", config=config.model_dump(), seed=tcfg.seed, dataset_id="wordle-v3-words"
    ).save(run_dir / "manifest.json")

    model = build_model(config.model, tokenizer.vocab_size).to(device)
    # Compiled wrapper (CUDA only) used for the hot training forward; `model` stays the
    # canonical GPT for optimizer/eval/checkpoint (shares parameters with the wrapper).
    forward_model = torch.compile(model) if (tcfg.compile and device.type == "cuda") else model
    optimizer = create_optimizer(model, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=tcfg.warmup_steps, total_steps=tcfg.max_steps)

    if tcfg.retrieval_pretrain:
        # Marginal (empty prompt -> word) for lexicon coverage + constraint-conditioned
        # (tight state -> answer) for retrieval/commit, over the full lexicon (§12).
        pairs: list[tuple[list[int], list[int]]] = [
            (tokenizer.empty_prompt(), [tokenizer.encode_token(c) for c in w]) for w in words
        ]
        print("Generating retrieval (constraint -> answer) examples...")
        retrieval = generate_retrieval_examples(
            tokenizer, pattern_matrix, words, tcfg.games_per_word, tcfg.seed, tcfg.max_candidates
        )
        pairs += retrieval
        print(f"  pretrain examples: {len(words)} marginal + {len(retrieval)} retrieval = {len(pairs)}")
        dataset: WordOnlyDataset | ConstraintDataset = ConstraintDataset(
            [p for p, _ in pairs], [t for _, t in pairs], tokenizer.pad_id
        )
    else:
        dataset = WordOnlyDataset(words, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_padded(b, tokenizer.pad_id),
    )

    reporter = MetricReporter(logger, run_dir, model, tokenizer, pattern_matrix, device, tcfg, split)
    vocab_size = tokenizer.vocab_size
    model.train()
    t0 = time.time()
    data_iter = infinite(loader)

    for step in range(1, tcfg.max_steps + 1):
        input_ids, target_ids, loss_mask = next(data_iter)
        input_ids, target_ids, loss_mask = input_ids.to(device), target_ids.to(device), loss_mask.to(device)

        with autocast(device, tcfg.amp):
            logits, _, _ = forward_model(input_ids)
            loss_all = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), reduction="none")
            loss = (loss_all.view(target_ids.shape) * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(model, tcfg.grad_clip)
        optimizer.step()
        scheduler.step()
        loss_v = loss.item()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:>6d}/{tcfg.max_steps} | loss {loss_v:.4f} | grad {grad_norm:.2f} | {elapsed:.0f}s")

        reporter.maybe_step(step, loss_v)
        reporter.maybe_eval(step)

        if step % tcfg.checkpoint_interval == 0 or step == tcfg.max_steps:
            ckpt_dir = run_dir / f"checkpoint-{step}"
            save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)

    if logger is not None:
        logger.close()
    print(f"Done. Run dir: {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    config = PretrainConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    train(config, cache_dir=args.cache_dir)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
