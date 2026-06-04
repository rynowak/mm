"""V2 pre-training: constraint-state encoding with character-level output.

Usage:
    uv run python wordle2/pretrain.py --config wordle2/configs/small.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time

import numpy as np
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
from mm_wordle import WordleEnv, load_answers
from torch.utils.data import DataLoader

from wordle.config import PretrainConfig
from wordle2.data import collate_constraint, load_constraint_data
from wordle2.tokenizer import ConstraintTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 pre-train with constraint-state encoding")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_model(config: PretrainConfig, tokenizer: ConstraintTokenizer) -> GPT:
    model_cfg = GPTConfig(
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        embed_dim=config.model.embed_dim,
        vocab_size=tokenizer.vocab_size,
        context_len=config.model.context_len,
        dropout=config.model.dropout,
    )
    model = GPT(model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    print(f"  Vocab: {tokenizer.vocab_size} tokens")
    return model


@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (input_ids, target_ids, loss_mask) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits, _, _ = model(input_ids)
        loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
        loss_all = loss_all.view(target_ids.shape)
        masked_loss = (loss_all * loss_mask).sum()
        n_tokens = loss_mask.sum().clamp(min=1)
        total_loss += masked_loss.item()
        total_tokens += n_tokens.item()

    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_games(
    model: GPT,
    tokenizer: ConstraintTokenizer,
    device: torch.device,
    n_games: int = 200,
    temperature: float = 0.1,
) -> dict:
    """Play games. Returns dict with valid_word_rate, info_gain, win_rate by turn."""
    from mm_wordle.reward import _compute_expected_info_gain
    from mm_wordle.solver import filter_candidates
    from mm_wordle.words import all_valid_words

    model.eval()
    answers = load_answers()
    valid = all_valid_words()
    env = WordleEnv()
    targets = random.sample(answers, min(n_games, len(answers)))
    letter_mask = torch.full((tokenizer.vocab_size,), float("-inf"), device=device)
    for lid in tokenizer.letter_ids:
        letter_mask[lid] = 0.0

    wins = 0
    by_turn: dict[int, dict[str, list[float]]] = {}

    for target in targets:
        state = env.reset(target_word=target)
        candidates = list(answers)
        while not state.solved and not state.failed:
            turn = state.turn + 1
            prompt_ids = tokenizer.encode_game_state(state)
            si = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

            generated: list[int] = []
            for _ in range(5):
                logits, _, _ = model(si)
                next_logits = logits[0, -1, :] / temperature
                next_logits = next_logits + letter_mask
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
                generated.append(next_id)
                si = torch.cat([si, torch.tensor([[next_id]], device=device)], dim=1)

            guess = tokenizer.decode_letters(generated)
            if len(guess) != 5:
                guess = "zzzzz"

            if turn not in by_turn:
                by_turn[turn] = {"valid": [], "info_gain": []}
            by_turn[turn]["valid"].append(1.0 if guess in valid else 0.0)

            ig = _compute_expected_info_gain(guess, candidates) if len(candidates) > 1 else 0.0
            by_turn[turn]["info_gain"].append(ig)

            state, _ = env.step(state, guess)
            fb = state.guesses[-1].feedback if state.guesses else []
            candidates = filter_candidates(candidates, guess, fb)

        if state.solved:
            wins += 1

    model.train()
    n = len(targets)
    result: dict = {"win_rate": wins / max(n, 1)}
    for t in sorted(by_turn):
        bt = by_turn[t]
        result[f"t{t}_valid"] = sum(bt["valid"]) / len(bt["valid"])
        result[f"t{t}_info_gain"] = sum(bt["info_gain"]) / len(bt["info_gain"])
    return result


def print_eval_metrics(metrics: dict) -> None:
    turns = sorted(t for t in metrics if t.startswith("t") and t.endswith("_valid"))
    vwr = " ".join(f"{t.split('_')[0]}={metrics[t]:.0%}" for t in turns)
    ig = " ".join(
        f"{t.replace('_valid', '').split('_')[0]}={metrics[t.replace('_valid', '_info_gain')]:.1f}" for t in turns
    )
    print(f"  [valid-words] {vwr}")
    print(f"  [info-gain]   {ig}")
    print(f"  [win-rate]    {metrics['win_rate']:.1%}")


def train(config: PretrainConfig, resume_path: str | None = None) -> None:
    seed = config.training.seed
    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = ConstraintTokenizer()
    model = build_model(config, tokenizer)
    model = model.to(device)

    optimizer = create_optimizer(
        model,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
    )

    start_step = 0
    if resume_path is not None:
        ckpt_path = pathlib.Path(resume_path)
        print(f"Resuming from {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        rng = checkpoint.get("rng_states", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        if "random" in rng:
            random.setstate(rng["random"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed at step {start_step}")

    train_dataset, val_dataset = load_constraint_data(config.model_dump(), tokenizer)
    pad_id = tokenizer.pad_id

    def collate_fn(batch: list) -> tuple:
        return collate_constraint(batch, pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    logger = MetricsLogger(experiment="pretrain-v2")
    print(f"Logging to {logger.log_dir}")

    manifest = RunManifest.capture(
        experiment="pretrain-v2",
        config=config.model_dump(),
        seed=seed,
        dataset_id="wordle-v2-constraint-state",
    )
    manifest.save(logger.log_dir / "manifest.json")

    max_steps = config.training.max_steps
    grad_clip = config.training.grad_clip
    eval_interval = config.training.eval_interval
    checkpoint_interval = config.training.checkpoint_interval

    model.train()
    step = start_step
    tokens_processed = 0
    t_start = time.time()

    print(f"\nStarting V2 training from step {step} to {max_steps}")
    print(f"  Batch size: {config.training.batch_size}\n")

    while step < max_steps:
        for input_ids, target_ids, loss_mask in train_loader:
            if step >= max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            logits, _, _ = model(input_ids)
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
            loss_all = loss_all.view(target_ids.shape)
            loss = (loss_all * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            tokens_processed += int(loss_mask.sum().item())
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            tokens_per_sec = tokens_processed / max(elapsed, 1e-6)

            if step % 10 == 0 or step == 1:
                print(
                    f"step {step:>5d}/{max_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )

            if step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  [eval] val_loss: {val_loss:.4f}")

                metrics = evaluate_games(model, tokenizer, device)
                print_eval_metrics(metrics)

            if step % checkpoint_interval == 0:
                ckpt_dir = logger.log_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
                print(f"  [checkpoint] saved to {ckpt_dir}")

    ckpt_dir = logger.log_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
    print(f"\nTraining complete at step {step}")
    print(f"Final checkpoint: {ckpt_dir}")

    metrics = evaluate_games(model, tokenizer, device)
    print("Final eval:")
    print_eval_metrics(metrics)

    logger.close()


def main() -> None:
    args = parse_args()
    config = PretrainConfig.from_yaml(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
