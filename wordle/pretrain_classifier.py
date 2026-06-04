"""Pre-train a word-classifier GPT on Wordle game transcripts.

The model reads a game state (prompt with feedback tokens) and classifies
which of the 2,315 answer words should be guessed next. One forward pass,
one softmax — no autoregressive character generation.

Usage:
    uv run python wordle/pretrain_classifier.py --config wordle/configs/small-classifier.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from config import PretrainConfig
from data import (
    collate_word_classifier,
    idx_to_word,
    load_word_classifier_data,
    num_words,
)
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from mm_tokenizers import CharTokenizer
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
from mm_wordle.serialize import game_state_to_prompt
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train a word-classifier GPT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_model(config: PretrainConfig, tokenizer: CharTokenizer) -> GPT:
    model_cfg = GPTConfig(
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        embed_dim=config.model.embed_dim,
        vocab_size=tokenizer.vocab_size,
        context_len=config.model.context_len,
        dropout=config.model.dropout,
        n_output_classes=num_words(),
    )
    model = GPT(model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    print(f"  Input vocab: {tokenizer.vocab_size} tokens")
    print(f"  Output classes: {num_words()} words")
    return model


@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Compute average validation loss and top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits, _, _ = model(input_ids)
        last_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_logits, targets)

        preds = last_logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_loss += loss.item() * len(targets)
        total_samples += len(targets)

    model.train()
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_games(
    model: GPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    n_games: int = 200,
    temperature: float = 0.1,
) -> tuple[float, float]:
    """Play games and return (win_rate, avg_guesses)."""
    model.eval()
    answers = load_answers()
    env = WordleEnv()
    targets = random.sample(answers, min(n_games, len(answers)))

    wins = 0
    total_turns = 0

    for target in targets:
        state = env.reset(target_word=target)
        while not state.solved and not state.failed:
            st = game_state_to_prompt(state)
            si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)
            logits, _, _ = model(si.unsqueeze(0))
            word_logits = logits[0, -1, :] / temperature
            probs = F.softmax(word_logits, dim=-1)
            word_idx = int(torch.multinomial(probs, num_samples=1).item())
            guess = idx_to_word(word_idx)
            state, _ = env.step(state, guess)

        if state.solved:
            wins += 1
        total_turns += state.turn

    model.train()
    return wins / len(targets), total_turns / len(targets)


def train(config: PretrainConfig, resume_path: str | None = None) -> None:
    seed = config.training.seed
    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
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

    train_dataset, val_dataset = load_word_classifier_data(config.model_dump(), tokenizer)
    pad_id = tokenizer.pad_id

    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return collate_word_classifier(batch, pad_id)

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

    logger = MetricsLogger(experiment="pretrain-classifier")
    print(f"Logging to {logger.log_dir}")

    manifest = RunManifest.capture(
        experiment="pretrain-classifier",
        config=config.model_dump(),
        seed=seed,
        dataset_id="wordle-transcripts-word-classifier",
    )
    manifest.save(logger.log_dir / "manifest.json")

    max_steps = config.training.max_steps
    grad_clip = config.training.grad_clip
    eval_interval = config.training.eval_interval
    checkpoint_interval = config.training.checkpoint_interval

    model.train()
    step = start_step
    t_start = time.time()

    print(f"\nStarting training from step {step} to {max_steps}")
    print(f"  Batch size: {config.training.batch_size}\n")

    while step < max_steps:
        for input_ids, targets in train_loader:
            if step >= max_steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits, _, _ = model(input_ids)
            last_logits = logits[:, -1, :]
            loss = F.cross_entropy(last_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            logger.log_scalar("train/loss", loss.item(), step)
            logger.log_scalar("train/lr", lr, step)

            if step % 10 == 0 or step == 1:
                print(
                    f"step {step:>5d}/{max_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"elapsed {elapsed:.0f}s"
                )

            if step % eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)
                logger.log_scalar("val/loss", val_loss, step)
                logger.log_scalar("val/accuracy", val_acc, step)
                print(f"  [eval] val_loss: {val_loss:.4f}, accuracy: {val_acc:.1%}")

                win_rate, avg_guesses = evaluate_games(model, tokenizer, device)
                logger.log_scalar("val/win_rate", win_rate, step)
                logger.log_scalar("val/avg_guesses", avg_guesses, step)
                print(f"  [games] win_rate: {win_rate:.1%}, avg_guesses: {avg_guesses:.1f}")

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

    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Final val_loss: {val_loss:.4f}, accuracy: {val_acc:.1%}")

    win_rate, avg_guesses = evaluate_games(model, tokenizer, device, n_games=200)
    print(f"Final win_rate: {win_rate:.1%}, avg_guesses: {avg_guesses:.1f}")

    logger.close()


def main() -> None:
    args = parse_args()
    config = PretrainConfig.from_yaml(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
