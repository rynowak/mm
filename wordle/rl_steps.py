"""Per-step training logic for REINFORCE and GRPO."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from decoding import sample_constrained
from grpo_train import (
    TurnExperience,
    collect_game_experience,
    compute_grpo_loss,
)
from mm_grpo import MovingAverageBaseline, reinforce_loss
from mm_training import clip_grad_norm
from mm_viz import GameReplay
from mm_wordle import game_state_to_prompt
from mm_wordle.solver import filter_candidates
from torch import Tensor

if TYPE_CHECKING:
    from pathlib import Path

    from mm_model import GPT
    from mm_tokenizers import CharTokenizer
    from mm_training import MetricsLogger
    from mm_wordle import WordleEnv, WordTrie


def reinforce_step(
    model: GPT,
    env: WordleEnv,
    batch_targets: list[str],
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    grad_clip: float,
    baseline: MovingAverageBaseline,
    logger: MetricsLogger,
    step: int,
) -> dict[str, float]:
    """Run one REINFORCE training step. Returns metrics dict."""
    from finetune import play_game_reinforce

    all_log_probs: list[Tensor] = []
    all_rewards: list[float] = []
    batch_size = len(batch_targets)

    for target in batch_targets:
        turn_lps, turn_rewards, _replay = play_game_reinforce(
            model=model,
            env=env,
            target_word=target,
            tokenizer=tokenizer,
            answers=answers,
            trie=trie,
            device=device,
            constrained=constrained,
        )
        total_reward = sum(turn_rewards)
        game_log_probs = torch.cat(turn_lps) if turn_lps else torch.zeros(1, device=device)
        all_log_probs.append(game_log_probs)
        all_rewards.append(total_reward)

    max_len = max(lp.shape[0] for lp in all_log_probs)
    padded_lps = torch.zeros(batch_size, max_len, device=device)
    for i, lp in enumerate(all_log_probs):
        padded_lps[i, : lp.shape[0]] = lp

    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    baseline.update(rewards_tensor.mean().item())
    baseline_val = torch.tensor(baseline.get(), dtype=torch.float32, device=device)

    loss = reinforce_loss(log_probs=padded_lps, rewards=rewards_tensor, baseline=baseline_val)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm(model, grad_clip)
    optimizer.step()
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    reward_mean = rewards_tensor.mean().item()

    logger.log_scalar("train/loss", loss.item(), step)
    logger.log_scalar("train/reward_mean", reward_mean, step)
    logger.log_scalar("train/grad_norm", grad_norm, step)
    logger.log_scalar("train/lr", lr, step)

    return {"loss": loss.item(), "reward_mean": reward_mean, "lr": lr}


def grpo_step(
    model: GPT,
    ref_model: GPT,
    opener_model: GPT | None,
    env: WordleEnv,
    batch_targets: list[str],
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
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
) -> tuple[dict[str, float], list[GameReplay], list[float], list[list[dict]]]:
    """Run one GRPO training step.

    Returns (metrics, replays, rewards, turn_details).
    """
    all_experiences: list[list[TurnExperience]] = []
    batch_rewards: list[float] = []
    batch_replays: list[GameReplay] = []
    batch_turn_details: list[list[dict]] = []

    for target in batch_targets:
        init_state = None
        init_candidates = None
        opener_guesses: list[str] = []
        opener_feedback: list[list[str]] = []

        if curriculum_phase == 2 and opener_model is not None:
            init_state = env.reset(target_word=target)
            init_candidates = list(answers)
            for _ in range(2):
                if init_state.solved or init_state.failed:
                    break
                st = game_state_to_prompt(init_state)
                si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)
                samples = sample_constrained(opener_model, si, trie, tokenizer, device, n_samples=1)
                guess = samples[0][0]
                init_state, _ = env.step(init_state, guess)
                fb = init_state.guesses[-1].feedback
                init_candidates = filter_candidates(init_candidates, guess, fb)
                opener_guesses.append(guess)
                opener_feedback.append([f.value for f in fb])

        if init_state is not None and init_state.solved:
            continue

        experiences, replay, game_reward, turn_details = collect_game_experience(
            model=model,
            ref_model=ref_model,
            env=env,
            target_word=target,
            tokenizer=tokenizer,
            answers=answers,
            trie=trie,
            device=device,
            group_size=group_size,
            constrained=constrained,
            max_turns=max_turns,
            initial_state=init_state,
            initial_candidates=init_candidates,
            composite=curriculum_phase != 1,
        )
        if not experiences:
            continue

        if opener_guesses:
            replay = GameReplay(
                target=replay.target,
                guesses=opener_guesses + replay.guesses,
                feedback=opener_feedback + replay.feedback,
                solved=replay.solved,
                turns=replay.turns,
            )

        all_experiences.append(experiences)
        batch_rewards.append(game_reward)
        batch_replays.append(replay)
        batch_turn_details.append(turn_details)

    if not all_experiences:
        return {}, [], [], []

    batch_size = len(all_experiences)
    last_metrics: dict[str, float] = {}
    last_loss = 0.0
    last_grad_norm = 0.0

    for _epoch in range(ppo_epochs):
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "kl_div": [],
            "entropy": [],
            "clip_fraction": [],
        }

        for game_experiences in all_experiences:
            game_loss, game_metrics = compute_grpo_loss(
                model,
                game_experiences,
                clip_epsilon,
                kl_beta,
                trie=trie if constrained else None,
                tokenizer=tokenizer if constrained else None,
            )
            epoch_loss = epoch_loss + game_loss
            for key in epoch_metrics:
                if key in game_metrics:
                    epoch_metrics[key].append(game_metrics[key])

        epoch_loss = epoch_loss / batch_size
        optimizer.zero_grad()
        epoch_loss.backward()
        last_grad_norm = clip_grad_norm(model, grad_clip)
        optimizer.step()
        last_loss = epoch_loss.item()
        last_metrics = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}

    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    reward_mean = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0

    logger.log_scalar("train/loss", last_loss, step)
    logger.log_scalar("train/reward_mean", reward_mean, step)
    logger.log_scalar("train/grad_norm", last_grad_norm, step)
    logger.log_scalar("train/lr", lr, step)
    for key, value in last_metrics.items():
        logger.log_scalar(f"train/{key}", value, step)

    # Save live data for dashboard
    live_dir = log_dir / "live"
    live_dir.mkdir(exist_ok=True)
    live_data = {
        "step": step,
        "loss": last_loss,
        "reward_mean": reward_mean,
        "rewards": batch_rewards,
        "kl_div": last_metrics.get("kl_div", 0.0),
        "clip_fraction": last_metrics.get("clip_fraction", 0.0),
        "entropy": last_metrics.get("entropy", 0.0),
        "games": [
            {
                "target": r.target,
                "guesses": r.guesses,
                "feedback": r.feedback,
                "solved": r.solved,
                "turns": r.turns,
                "reward": br,
                "turn_details": td,
            }
            for r, br, td in zip(batch_replays, batch_rewards, batch_turn_details, strict=True)
        ],
    }
    (live_dir / "latest.json").write_text(json.dumps(live_data))

    history_line = json.dumps(
        {
            "step": step,
            "loss": last_loss,
            "kl_div": last_metrics.get("kl_div", 0.0),
            "reward_mean": reward_mean,
            "clip_fraction": last_metrics.get("clip_fraction", 0.0),
        }
    )
    with open(live_dir / "history.jsonl", "a") as hf:
        hf.write(history_line + "\n")

    return (
        {"loss": last_loss, "reward_mean": reward_mean, "lr": lr, **last_metrics},
        batch_replays,
        batch_rewards,
        batch_turn_details,
    )
