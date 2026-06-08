"""V3 Phase 2: SFT on golden games (behavior cloning, §5.3).

Warm-starts from the pre-train checkpoint and fine-tunes on per-turn
(constraint-state, guess) examples from strong golden games whose answers are
drawn ONLY from the train split (hold-out is never an answer, R4). A small
fraction of word-only replay batches retains lexical coverage of the hold-out (N1).

Usage:
    uv run python -m wordle3.sft --config wordle3/configs/sft-large.yaml \
        --checkpoint runs/pretrain-v3/<ts>/checkpoint-15000/model.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from mm_training import (
    MetricsLogger,
    RunManifest,
    clip_grad_norm,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from mm_wordle import (
    ConstraintTokenizer,
    GoldenSolver,
    PatternMatrix,
    WordleEnv,
    load_full_word_set,
    play_golden_game,
)
from torch.utils.data import DataLoader

from wordle3.config import SFTConfig
from wordle3.data import (
    ConstraintDataset,
    WordOnlyDataset,
    collate_padded,
    golden_examples_from_game,
    oversample_late_turns,
)
from wordle3.splits import Split, load_split
from wordle3.steplog import MetricReporter
from wordle3.trainutil import autocast, infinite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 SFT on golden games")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Pre-train checkpoint (model.pt)")
    parser.add_argument("--cache-dir", type=str, default="runs/cache")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps (quick runs)")
    parser.add_argument("--n-golden-games", type=int, default=None, help="Override config n_golden_games")
    return parser.parse_args()


def generate_sft_dataset(
    pattern_matrix: PatternMatrix,
    tokenizer: ConstraintTokenizer,
    train_answers: list[str],
    n_games: int,
    seed: int,
) -> ConstraintDataset:
    """Play golden games on train answers and convert to per-turn SFT examples."""
    solver = GoldenSolver(pattern_matrix)
    env = WordleEnv()
    grng = random.Random(seed)
    train_set = set(train_answers)
    examples: list[tuple[list[int], list[int], int]] = []
    for i in range(n_games):
        target = grng.choice(train_answers)
        assert target in train_set, "hold-out word used as a golden answer (R4 violation)"
        state = play_golden_game(solver, env, target)
        examples.extend(golden_examples_from_game(state, tokenizer))
        if (i + 1) % 1000 == 0:
            print(f"  generated {i + 1}/{n_games} golden games")
    examples = oversample_late_turns(examples)
    prompts = [e[0] for e in examples]
    targets = [e[1] for e in examples]
    return ConstraintDataset(prompts, targets, tokenizer.pad_id)


def train(
    config: SFTConfig,
    *,
    checkpoint: str,
    words: list[str] | None = None,
    pattern_matrix: PatternMatrix | None = None,
    split: Split | None = None,
    run_dir: Path | None = None,
    device: torch.device | None = None,
    cache_dir: str = "runs/cache",
) -> Path:
    """Run SFT on golden games warm-started from ``checkpoint``. Returns the run dir."""
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

    ckpt = load_checkpoint(Path(checkpoint), device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"Warm-started from {checkpoint} ({sum(p.numel() for p in model.parameters()):,} params)")

    if run_dir is None:
        logger = MetricsLogger(experiment="sft-v3")
        run_dir = logger.log_dir
    else:
        logger = None
    print(f"Run dir: {run_dir}")
    RunManifest.capture(
        experiment="sft-v3", config=config.model_dump(), seed=tcfg.seed, dataset_id="wordle-v3-golden"
    ).save(run_dir / "manifest.json")

    forward_model = torch.compile(model) if (tcfg.compile and device.type == "cuda") else model
    optimizer = create_optimizer(model, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=tcfg.warmup_steps, total_steps=tcfg.max_steps)

    print(f"Generating {tcfg.n_golden_games} golden games (train answers only)...")
    sft_dataset = generate_sft_dataset(pattern_matrix, tokenizer, split.train_answers, tcfg.n_golden_games, tcfg.seed)
    print(f"  {len(sft_dataset)} SFT examples (after late-turn oversampling)")
    sft_iter = infinite(
        DataLoader(
            sft_dataset,
            batch_size=tcfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda b: collate_padded(b, tokenizer.pad_id),
        )
    )

    replay_iter = None
    if tcfg.replay_frac > 0:
        replay_iter = infinite(
            DataLoader(
                WordOnlyDataset(words, tokenizer),
                batch_size=tcfg.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda b: collate_padded(b, tokenizer.pad_id),
            )
        )

    reporter = MetricReporter(logger, run_dir, model, tokenizer, pattern_matrix, device, tcfg, split)
    vocab_size = tokenizer.vocab_size
    model.train()
    t0 = time.time()
    rrng = random.Random(tcfg.seed + 1)

    for step in range(1, tcfg.max_steps + 1):
        use_replay = replay_iter is not None and rrng.random() < tcfg.replay_frac
        input_ids, target_ids, loss_mask = next(replay_iter if use_replay else sft_iter)
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
            save_checkpoint(run_dir / f"checkpoint-{step}" / "model.pt", model, optimizer, step, model.config)

    if logger is not None:
        logger.close()
    print(f"Done. Run dir: {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    config = SFTConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.n_golden_games is not None:
        config.training.n_golden_games = args.n_golden_games
    train(config, checkpoint=args.checkpoint, cache_dir=args.cache_dir)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
