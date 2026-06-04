"""RL fine-tune a word-classifier GPT to play Wordle using GRPO.

The model outputs a probability distribution over 2,315 answer words.
GRPO samples multiple words per turn, computes rewards, and updates
the policy to favor high-reward words.

Usage:
    uv run python wordle/finetune_classifier.py \
        --config wordle/configs/finetune-classifier.yaml \
        --checkpoint runs/pretrain-classifier/.../model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from config import FinetuneConfig
from data import idx_to_word
from mm_grpo import compute_group_advantages
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
from mm_wordle import WordleEnv, compute_reward, load_answers
from mm_wordle.serialize import game_state_to_prompt
from mm_wordle.solver import filter_candidates
from torch import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL fine-tune a word-classifier GPT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--opener", type=str, default=None)
    return parser.parse_args()


def load_pretrained_model(checkpoint_path: Path, device: torch.device) -> GPT:
    checkpoint = load_checkpoint(checkpoint_path, device)
    config_dict = checkpoint["config"]
    model_config = GPTConfig(**config_dict)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def sample_words(
    model: GPT,
    state_ids: Tensor,
    device: torch.device,
    n_samples: int = 1,
    temperature: float = 1.0,
) -> tuple[list[int], Tensor]:
    """Sample word indices from the model's output distribution.

    Returns (word_indices, log_probs) where log_probs shape is (n_samples,).
    """
    logits, _, _ = model(state_ids.unsqueeze(0))
    word_logits = logits[0, -1, :] / temperature
    log_probs = F.log_softmax(word_logits, dim=-1)
    probs = log_probs.exp()

    indices = torch.multinomial(probs, num_samples=n_samples, replacement=True)
    sampled_log_probs = log_probs[indices]

    return indices.tolist(), sampled_log_probs.detach()


@torch.no_grad()
def evaluate_games(
    model: GPT,
    tokenizer: CharTokenizer,
    eval_words: list[str],
    device: torch.device,
    temperature: float = 0.1,
) -> tuple[float, float]:
    """Play evaluation games. Returns (win_rate, avg_guesses)."""
    model.eval()
    env = WordleEnv()
    wins = 0
    total_turns = 0

    for target in eval_words:
        state = env.reset(target_word=target)
        while not state.solved and not state.failed:
            st = game_state_to_prompt(state)
            si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)
            logits, _, _ = model(si.unsqueeze(0))
            word_logits = logits[0, -1, :] / temperature
            probs = F.softmax(word_logits, dim=-1)
            word_idx = int(torch.multinomial(probs, num_samples=1).item())
            state, _ = env.step(state, idx_to_word(word_idx))

        if state.solved:
            wins += 1
        total_turns += state.turn

    model.train()
    n = len(eval_words)
    return wins / max(n, 1), total_turns / max(n, 1)


def grpo_step(
    model: GPT,
    ref_model: GPT,
    opener_model: GPT | None,
    env: WordleEnv,
    batch_targets: list[str],
    tokenizer: CharTokenizer,
    answers: list[str],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_clip: float,
    group_size: int,
    clip_epsilon: float,
    kl_beta: float,
    ppo_epochs: int,
    curriculum_phase: int,
    max_turns: int,
    logger: MetricsLogger,
    log_dir: Path,
    step: int,
) -> dict[str, float]:
    """Run one GRPO training step for the word classifier."""
    all_state_ids: list[Tensor] = []
    all_word_indices: list[Tensor] = []
    all_rewards: list[Tensor] = []
    all_ref_log_probs: list[Tensor] = []
    all_old_log_probs: list[Tensor] = []

    batch_game_rewards: list[float] = []
    batch_replays: list[dict] = []

    for target in batch_targets:
        state = env.reset(target_word=target)
        candidates = list(answers)
        game_reward = 0.0
        replay_guesses: list[str] = []
        replay_feedback: list[list[str]] = []

        # Phase 2: opener model plays turns 1-2
        if curriculum_phase == 2 and opener_model is not None:
            for _ in range(2):
                if state.solved or state.failed:
                    break
                st = game_state_to_prompt(state)
                si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, _, _ = opener_model(si.unsqueeze(0))
                    word_idx = int(logits[0, -1, :].argmax().item())
                guess = idx_to_word(word_idx)
                state, _ = env.step(state, guess)
                fb = state.guesses[-1].feedback
                candidates = filter_candidates(candidates, guess, fb)
                replay_guesses.append(guess)
                replay_feedback.append([f.value for f in fb])

            if state.solved:
                batch_replays.append(
                    {
                        "target": target,
                        "guesses": replay_guesses,
                        "feedback": replay_feedback,
                        "solved": True,
                        "turns": state.turn,
                        "reward": 0.0,
                    }
                )
                continue

        while not state.solved and not state.failed and state.turn < max_turns:
            st = game_state_to_prompt(state)
            si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)

            word_indices, _ = sample_words(model, si, device, n_samples=group_size)

            rewards: list[float] = []
            for widx in word_indices:
                guess = idx_to_word(widx)
                sim_state, _ = env.step(state, guess)
                fb = sim_state.guesses[-1].feedback if sim_state.guesses else []
                r, _, _ = compute_reward(guess, fb, candidates, composite=True)
                rewards.append(r)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            word_indices_t = torch.tensor(word_indices, dtype=torch.long, device=device)

            with torch.no_grad():
                logits, _, _ = model(si.unsqueeze(0))
                old_lp = F.log_softmax(logits[0, -1, :], dim=-1)
                old_log_probs = old_lp[word_indices_t]

                ref_logits, _, _ = ref_model(si.unsqueeze(0))
                ref_lp = F.log_softmax(ref_logits[0, -1, :], dim=-1)
                ref_log_probs = ref_lp[word_indices_t]

            all_state_ids.append(si)
            all_word_indices.append(word_indices_t)
            all_rewards.append(rewards_t)
            all_ref_log_probs.append(ref_log_probs)
            all_old_log_probs.append(old_log_probs)

            best_idx = int(rewards_t.argmax().item())
            chosen_word = idx_to_word(word_indices[best_idx])
            game_reward += rewards[best_idx]

            new_state, _ = env.step(state, chosen_word)
            feedback = new_state.guesses[-1].feedback if new_state.guesses else []
            replay_guesses.append(chosen_word)
            replay_feedback.append([f.value for f in feedback])
            candidates = filter_candidates(candidates, chosen_word, feedback)
            state = new_state

        batch_game_rewards.append(game_reward)
        batch_replays.append(
            {
                "target": target,
                "guesses": replay_guesses,
                "feedback": replay_feedback,
                "solved": state.solved,
                "turns": state.turn,
                "reward": game_reward,
            }
        )

    if not all_state_ids:
        return {}

    # PPO epochs
    n_turns = len(all_state_ids)
    last_loss = 0.0
    last_kl = 0.0
    last_clip_frac = 0.0

    for _epoch in range(ppo_epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_kl = 0.0
        epoch_clip = 0.0

        for i in range(n_turns):
            logits, _, _ = model(all_state_ids[i].unsqueeze(0))
            current_lp = F.log_softmax(logits[0, -1, :], dim=-1)
            current_log_probs = current_lp[all_word_indices[i]]

            advantages = compute_group_advantages(all_rewards[i])

            ratio = (current_log_probs - all_old_log_probs[i]).exp()
            clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            kl = (current_log_probs - all_ref_log_probs[i]).exp() - 1 - (current_log_probs - all_ref_log_probs[i])
            kl_loss = kl.mean()

            turn_loss = policy_loss + kl_beta * kl_loss
            epoch_loss = epoch_loss + turn_loss

            epoch_kl += kl.mean().item()
            clip_frac = ((ratio - 1).abs() > clip_epsilon).float().mean().item()
            epoch_clip += clip_frac

        epoch_loss = epoch_loss / n_turns
        optimizer.zero_grad()
        epoch_loss.backward()
        clip_grad_norm(model, grad_clip)
        optimizer.step()

        last_loss = epoch_loss.item()
        last_kl = epoch_kl / n_turns
        last_clip_frac = epoch_clip / n_turns

    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    reward_mean = sum(batch_game_rewards) / len(batch_game_rewards)

    logger.log_scalar("train/loss", last_loss, step)
    logger.log_scalar("train/reward_mean", reward_mean, step)
    logger.log_scalar("train/kl_div", last_kl, step)
    logger.log_scalar("train/clip_fraction", last_clip_frac, step)
    logger.log_scalar("train/lr", lr, step)

    # Save live data
    live_dir = log_dir / "live"
    live_dir.mkdir(exist_ok=True)
    live_data = {
        "step": step,
        "loss": last_loss,
        "reward_mean": reward_mean,
        "kl_div": last_kl,
        "games": batch_replays,
    }
    (live_dir / "latest.json").write_text(json.dumps(live_data))
    history_line = json.dumps({"step": step, "loss": last_loss, "reward_mean": reward_mean, "kl_div": last_kl})
    with open(live_dir / "history.jsonl", "a") as hf:
        hf.write(history_line + "\n")

    return {
        "loss": last_loss,
        "reward_mean": reward_mean,
        "lr": lr,
        "kl_div": last_kl,
        "clip_fraction": last_clip_frac,
    }


def train(
    config: FinetuneConfig,
    checkpoint_path: str,
    resume_path: str | None = None,
    opener_checkpoint: str | None = None,
) -> None:
    rl_cfg = config.rl
    seed_everything(rl_cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    answers = load_answers()
    env = WordleEnv()

    model = load_pretrained_model(Path(checkpoint_path), device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters, output_classes={model.config.output_size}")

    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    opener_model = None
    if rl_cfg.curriculum_phase == 2 and opener_checkpoint:
        opener_model = load_pretrained_model(Path(opener_checkpoint), device)
        opener_model.eval()
        for param in opener_model.parameters():
            param.requires_grad = False
        print(f"Loaded opener model from {opener_checkpoint}")

    optimizer = create_optimizer(model, lr=rl_cfg.learning_rate, weight_decay=rl_cfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=rl_cfg.warmup_steps, total_steps=rl_cfg.max_steps)

    # Precompute expected info gain
    from mm_wordle.reward import precompute_expected_info_gain

    print("Precomputing expected info gain...")
    precompute_expected_info_gain()

    eval_words = random.sample(answers, min(rl_cfg.max_eval_games, len(answers)))

    logger = MetricsLogger(experiment="finetune-classifier")
    print(f"Logging to {logger.log_dir}")

    manifest = RunManifest.capture(
        experiment="finetune-classifier",
        config=config.model_dump(),
        seed=rl_cfg.seed,
        dataset_id="wordle-answers",
    )
    manifest.save(logger.log_dir / "manifest.json")

    eval_words_path = logger.log_dir / "eval_words.json"
    eval_words_path.write_text(json.dumps(eval_words, indent=2))

    max_steps = rl_cfg.max_steps
    recent_wins: deque[bool] = deque(maxlen=100)
    deque(maxlen=100)

    model.train()
    t_start = time.time()

    print(f"\nStarting RL fine-tuning: {max_steps} steps")
    print(f"  Group size: {rl_cfg.group_size}")
    print(f"  Batch size: {rl_cfg.batch_size}")
    print(f"  Max turns: {rl_cfg.max_turns}\n")

    for step in range(1, max_steps + 1):
        batch_targets = random.choices(answers, k=rl_cfg.batch_size)

        metrics = grpo_step(
            model=model,
            ref_model=ref_model,
            opener_model=opener_model,
            env=env,
            batch_targets=batch_targets,
            tokenizer=tokenizer,
            answers=answers,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=rl_cfg.grad_clip,
            group_size=rl_cfg.group_size,
            clip_epsilon=rl_cfg.clip_epsilon,
            kl_beta=rl_cfg.kl_beta,
            ppo_epochs=rl_cfg.ppo_epochs,
            curriculum_phase=rl_cfg.curriculum_phase,
            max_turns=rl_cfg.max_turns,
            logger=logger,
            log_dir=logger.log_dir,
            step=step,
        )

        if not metrics:
            continue

        live_path = logger.log_dir / "live" / "latest.json"
        if live_path.exists():
            live = json.loads(live_path.read_text())
            for g in live["games"]:
                recent_wins.append(g["solved"])

        elapsed = time.time() - t_start
        if step % 10 == 0 or step == 1:
            wr = sum(recent_wins) / max(len(recent_wins), 1)
            print(
                f"step {step:>5d}/{max_steps} | "
                f"loss {metrics['loss']:.4f} | "
                f"win_rate {wr:.0%} | "
                f"reward {metrics['reward_mean']:.2f} | "
                f"kl {metrics['kl_div']:.4f} | "
                f"lr {metrics['lr']:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )

        if step % rl_cfg.eval_interval == 0:
            print(f"\n  [eval] Running {len(eval_words)} evaluation games...")
            eval_wr, eval_ag = evaluate_games(model, tokenizer, eval_words, device)
            logger.log_scalar("eval/win_rate", eval_wr, step)
            logger.log_scalar("eval/avg_guesses", eval_ag, step)
            print(f"  [eval] win_rate: {eval_wr:.1%}, avg_guesses: {eval_ag:.1f}\n")

        if step % rl_cfg.checkpoint_interval == 0:
            ckpt_dir = logger.log_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
            torch.save(ref_model.state_dict(), ckpt_dir / "ref_model.pt")
            print(f"  [checkpoint] saved to {ckpt_dir}")

    # Final
    ckpt_dir = logger.log_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
    torch.save(ref_model.state_dict(), ckpt_dir / "ref_model.pt")
    print(f"\nTraining complete at step {step}")

    eval_wr, eval_ag = evaluate_games(model, tokenizer, eval_words, device, temperature=0.1)
    print(f"Final eval: win_rate={eval_wr:.1%}, avg_guesses={eval_ag:.1f}")

    logger.close()


def main() -> None:
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)
    train(config, checkpoint_path=args.checkpoint, resume_path=args.resume, opener_checkpoint=args.opener)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
