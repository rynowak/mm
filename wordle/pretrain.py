"""Pre-train a character-level GPT on TinyStories + Wordle word lists.

Usage:
    uv run python wordle/pretrain.py --config wordle/configs/small.yaml

Resume:
    uv run python wordle/pretrain.py --config wordle/configs/small.yaml \
        --resume runs/pretrain-small-.../checkpoint-1000/model.pt
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
from data import collate_wordle, load_pretrain_data
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
from mm_wordle import all_valid_words
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-train a character-level GPT")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    return parser.parse_args()


def build_model(config: PretrainConfig, tokenizer: CharTokenizer) -> GPT:
    """Build a GPT model from config."""
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
    print(f"Model: {param_count:,} parameters (estimate: {model_cfg.param_count_estimate():,})")
    return model


@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Compute average masked validation loss."""
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
def generate_sample(
    model: GPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    max_chars: int = 200,
) -> str:
    """Generate sample text from a [bos] prompt."""
    model.eval()
    prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    output = model.generate(prompt, max_new_tokens=max_chars, temperature=0.8, top_k=40)
    tokens = output[0].tolist()
    text = tokenizer.decode(tokens)
    model.train()
    return text


@torch.no_grad()
def compute_valid_word_rate(
    model: GPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    n_samples: int = 100,
) -> float:
    """Generate five-character sequences and check what fraction are valid Wordle words."""
    model.eval()
    valid_words = all_valid_words()

    n_valid = 0
    for _ in range(n_samples):
        prompt = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
        output = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=40)
        # Take only the 5 generated characters (skip the bos token)
        generated_ids = output[0, 1:].tolist()
        try:
            word = tokenizer.decode(generated_ids)
            # Only count if it's exactly 5 lowercase letters
            if len(word) == 5 and word.isalpha() and word.islower() and word in valid_words:
                n_valid += 1
        except ValueError:
            pass

    model.train()
    return n_valid / n_samples


def train(config: PretrainConfig, resume_path: str | None = None) -> None:
    """Run the pre-training loop."""
    # Setup
    seed = config.training.seed
    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    model = build_model(config, tokenizer)
    model = model.to(device)

    # Optimizer and scheduler
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

    # Resume from checkpoint
    start_step = 0
    if resume_path is not None:
        ckpt_path = pathlib.Path(resume_path)
        print(f"Resuming from {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        # Restore RNG states
        rng = checkpoint.get("rng_states", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        if "random" in rng:
            random.setstate(rng["random"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        # Advance scheduler to the right step
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed at step {start_step}")

    # Load data
    train_dataset, val_dataset = load_pretrain_data(config.model_dump(), tokenizer)
    pad_id = tokenizer.pad_id

    def collate_fn(batch):
        return collate_wordle(batch, pad_id)

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

    # Determine model size label from config
    embed_dim = config.model.embed_dim
    size_label = "small" if embed_dim <= 256 else "medium"

    # Metrics logger
    logger = MetricsLogger(experiment=f"pretrain-{size_label}")
    print(f"Logging to {logger.log_dir}")

    # Run manifest
    manifest = RunManifest.capture(
        experiment=f"pretrain-{size_label}",
        config=config.model_dump(),
        seed=seed,
        dataset_id="wordle-transcripts",
    )
    manifest.save(logger.log_dir / "manifest.json")

    # Training loop
    max_steps = config.training.max_steps
    grad_clip = config.training.grad_clip
    eval_interval = config.training.eval_interval
    checkpoint_interval = config.training.checkpoint_interval
    batch_size = config.training.batch_size

    model.train()
    step = start_step
    tokens_processed = 0
    t_start = time.time()

    print(f"\nStarting training from step {step} to {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Eval every {eval_interval} steps, Checkpoint every {checkpoint_interval} steps\n")

    while step < max_steps:
        for input_ids, target_ids, loss_mask in train_loader:
            if step >= max_steps:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            # Forward pass — compute masked loss (only on target letter tokens)
            logits, _, _ = model(input_ids)
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
            loss_all = loss_all.view(target_ids.shape)
            loss = (loss_all * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            tokens_processed += int(loss_mask.sum().item())

            # Get current learning rate
            lr = optimizer.param_groups[0]["lr"]

            # Compute tokens/sec
            elapsed = time.time() - t_start
            tokens_per_sec = tokens_processed / max(elapsed, 1e-6)

            # Log training metrics
            logger.log_scalar("train/loss", loss.item(), step)
            logger.log_scalar("train/lr", lr, step)
            logger.log_scalar("train/grad_norm", grad_norm, step)
            logger.log_scalar("train/tokens_per_sec", tokens_per_sec, step)

            # Print progress
            if step % 10 == 0 or step == 1:
                print(
                    f"step {step:>5d}/{max_steps} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec:.0f}"
                )

            # Evaluation
            if step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                logger.log_scalar("val/loss", val_loss, step)
                print(f"  [eval] val_loss: {val_loss:.4f}")

                # Generate sample text
                sample = generate_sample(model, tokenizer, device)
                logger.log_text("samples/generated", sample, step)
                print(f"  [sample] {sample[:100]}...")

                # Valid word rate
                vwr = compute_valid_word_rate(model, tokenizer, device)
                logger.log_scalar("val/valid_word_rate", vwr, step)
                print(f"  [valid-word-rate] {vwr:.2%}")

            # Weight histograms (every 5 eval intervals)
            if step % (eval_interval * 5) == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.log_histogram(f"weights/{name}", param, step)

            # Checkpoint
            if step % checkpoint_interval == 0:
                ckpt_dir = logger.log_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    ckpt_dir / "model.pt",
                    model,
                    optimizer,
                    step,
                    model.config,
                )
                print(f"  [checkpoint] saved to {ckpt_dir}")

    # Final checkpoint
    ckpt_dir = logger.log_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        ckpt_dir / "model.pt",
        model,
        optimizer,
        step,
        model.config,
    )
    print(f"\nTraining complete at step {step}")
    print(f"Final checkpoint: {ckpt_dir}")

    # Final evaluation
    val_loss = evaluate(model, val_loader, device)
    print(f"Final val_loss: {val_loss:.4f}")

    vwr = compute_valid_word_rate(model, tokenizer, device)
    print(f"Final valid-word-rate: {vwr:.2%}")

    logger.close()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = PretrainConfig.from_yaml(args.config)
    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
