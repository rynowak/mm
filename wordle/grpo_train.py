"""GRPO experience collection, loss computation, and step data capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from decoding import sample_constrained, sample_unconstrained
from mm_grpo import (
    build_step_data,
    compute_group_advantages,
    grpo_loss,
    sequence_log_probs,
)
from mm_viz import GameReplay
from mm_wordle import compute_reward, game_state_to_prompt
from mm_wordle.solver import filter_candidates
from torch import Tensor

if TYPE_CHECKING:
    from mm_model import GPT
    from mm_tokenizers import CharTokenizer
    from mm_viz import GRPOStepData
    from mm_wordle import GameState, WordleEnv, WordTrie


@dataclass
class TurnExperience:
    """Collected experience from one turn of a GRPO game."""

    state_ids: Tensor
    word_ids_batch: Tensor
    rewards_tensor: Tensor
    ref_log_probs: Tensor
    old_log_probs: Tensor


def collect_game_experience(
    model: GPT,
    ref_model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    group_size: int,
    constrained: bool,
    max_turns: int = 6,
    initial_state: GameState | None = None,
    initial_candidates: list[str] | None = None,
    solve_bonus: bool = True,
) -> tuple[list[TurnExperience], GameReplay, float, list[dict]]:
    """Play a Wordle game and collect experience for GRPO optimization."""
    state = initial_state if initial_state is not None else env.reset(target_word)
    experiences: list[TurnExperience] = []
    replay_guesses: list[str] = []
    replay_feedback: list[list[str]] = []
    turn_details: list[dict] = []
    total_reward = 0.0
    candidates = list(initial_candidates) if initial_candidates is not None else list(answers)
    turns_played = 0

    while not state.solved and not state.failed and turns_played < max_turns:
        state_tokens = game_state_to_prompt(state)
        state_ids = torch.tensor(tokenizer.encode("".join(state_tokens)), dtype=torch.long, device=device)

        if constrained:
            samples = sample_constrained(model, state_ids, trie, tokenizer, device, n_samples=group_size)
        else:
            samples = sample_unconstrained(model, state_ids, device, tokenizer, n_samples=group_size)

        guesses = [s[0] for s in samples]
        word_ids_list = [s[1] for s in samples]
        word_ids_batch = torch.stack(word_ids_list)

        rewards: list[float] = []
        group_details: list[dict] = []
        for guess in guesses:
            sim_state, _ = env.step(state, guess)
            fb = sim_state.guesses[-1].feedback if sim_state.guesses else []
            r, actual, expected = compute_reward(guess, fb, candidates, solve_bonus=solve_bonus)
            rewards.append(r)
            group_details.append(
                {
                    "guess": guess,
                    "reward": round(r, 3),
                    "actual": round(actual, 3),
                    "expected": round(expected, 3),
                    "candidates": len(candidates),
                }
            )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        prompt_expanded = state_ids.unsqueeze(0).expand(group_size, -1)
        full_sequences = torch.cat([prompt_expanded, word_ids_batch], dim=-1)
        prompt_len = state_ids.shape[0]

        with torch.no_grad():
            old_logits, _, _ = model(full_sequences)
            old_completion_logits = old_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
            if constrained:
                old_completion_logits = _apply_trie_masks(old_completion_logits, word_ids_batch, trie, tokenizer)
            old_lp = sequence_log_probs(old_completion_logits, word_ids_batch)

            ref_logits, _, _ = ref_model(full_sequences)
            ref_completion_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
            if constrained:
                ref_completion_logits = _apply_trie_masks(ref_completion_logits, word_ids_batch, trie, tokenizer)
            ref_lp = sequence_log_probs(ref_completion_logits, word_ids_batch)

        experiences.append(
            TurnExperience(
                state_ids=state_ids,
                word_ids_batch=word_ids_batch,
                rewards_tensor=rewards_tensor,
                ref_log_probs=ref_lp,
                old_log_probs=old_lp,
            )
        )

        best_idx = int(rewards_tensor.argmax().item())
        chosen_guess = guesses[best_idx]
        total_reward += rewards[best_idx]

        turn_details.append(
            {
                "chosen": chosen_guess,
                "candidates": len(candidates),
                "group": group_details,
            }
        )

        new_state, _done = env.step(state, chosen_guess)
        feedback = new_state.guesses[-1].feedback if new_state.guesses else []
        replay_guesses.append(chosen_guess)
        replay_feedback.append([fb.value for fb in feedback])

        candidates = filter_candidates(candidates, chosen_guess, feedback)
        state = new_state
        turns_played += 1

    replay = GameReplay(
        target=target_word,
        guesses=replay_guesses,
        feedback=replay_feedback,
        solved=state.solved,
        turns=state.turn,
    )
    return experiences, replay, total_reward, turn_details


def _apply_trie_masks(
    completion_logits: Tensor,
    word_ids_batch: Tensor,
    trie: WordTrie,
    tokenizer: CharTokenizer,
) -> Tensor:
    """Apply trie masks to completion logits using precomputed GPU masks."""
    group_size, seq_len, _vocab_size = completion_logits.shape
    masked = completion_logits.clone()

    for pos in range(seq_len):
        prefixes = []
        for i in range(group_size):
            prefix = tokenizer.decode(word_ids_batch[i, :pos].tolist()) if pos > 0 else ""
            prefixes.append(prefix)
        masked[:, pos] = masked[:, pos] + trie.gpu_mask(prefixes)

    return masked


def compute_grpo_loss(
    model: GPT,
    experiences: list[TurnExperience],
    clip_epsilon: float,
    kl_beta: float,
    trie: WordTrie | None = None,
    tokenizer: CharTokenizer | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute GRPO loss from collected experience."""
    total_loss = torch.tensor(0.0, device=experiences[0].state_ids.device)
    metrics_accum: dict[str, list[float]] = {
        "policy_loss": [],
        "kl_div": [],
        "entropy": [],
        "clip_fraction": [],
    }

    for exp in experiences:
        prompt_expanded = exp.state_ids.unsqueeze(0).expand(exp.word_ids_batch.shape[0], -1)
        full_sequences = torch.cat([prompt_expanded, exp.word_ids_batch], dim=-1)
        prompt_len = exp.state_ids.shape[0]

        logits, _, _ = model(full_sequences)
        completion_logits = logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        if trie is not None and tokenizer is not None:
            completion_logits = _apply_trie_masks(completion_logits, exp.word_ids_batch, trie, tokenizer)
        current_log_probs = sequence_log_probs(completion_logits, exp.word_ids_batch)

        turn_loss, turn_metrics = grpo_loss(
            log_probs=current_log_probs,
            old_log_probs=exp.old_log_probs,
            rewards=exp.rewards_tensor,
            ref_log_probs=exp.ref_log_probs,
            clip_epsilon=clip_epsilon,
            beta=kl_beta,
        )
        total_loss = total_loss + turn_loss
        for key in metrics_accum:
            if key in turn_metrics:
                metrics_accum[key].append(turn_metrics[key])

    if experiences:
        total_loss = total_loss / len(experiences)

    aggregated = {k: sum(v) / max(len(v), 1) for k, v in metrics_accum.items()}
    return total_loss, aggregated


def collect_grpo_step_data(
    model: GPT,
    ref_model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    group_size: int,
    step: int,
) -> GRPOStepData | None:
    """Collect a GRPOStepData snapshot for one game turn (first turn only)."""
    state = env.reset(target_word)
    state_tokens = game_state_to_prompt(state)
    state_text = "".join(state_tokens)
    state_ids = torch.tensor(tokenizer.encode(state_text), dtype=torch.long, device=device)

    samples = sample_constrained(model, state_ids, trie, tokenizer, device, n_samples=group_size)

    guesses = [s[0] for s in samples]
    word_ids_list = [s[1] for s in samples]
    word_ids_batch = torch.stack(word_ids_list)

    rewards: list[float] = []
    reward_breakdowns: list[dict[str, float]] = []
    for guess in guesses:
        sim_state, _ = env.step(state, guess)
        fb = sim_state.guesses[-1].feedback if sim_state.guesses else []
        r, _, _ = compute_reward(guess, fb, list(answers))
        rewards.append(r)
        reward_breakdowns.append({"total": r})

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = compute_group_advantages(rewards_tensor)

    prompt_len = state_ids.shape[0]
    prompt_expanded = state_ids.unsqueeze(0).expand(group_size, -1)
    full_sequences = torch.cat([prompt_expanded, word_ids_batch], dim=-1)

    with torch.no_grad():
        logits_new, _, _ = model(full_sequences)
        completion_logits_new = logits_new[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        new_log_probs = sequence_log_probs(completion_logits_new, word_ids_batch)

        ref_logits, _, _ = ref_model(full_sequences)
        ref_completion_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        ref_log_probs_t = sequence_log_probs(ref_completion_logits, word_ids_batch)

    kl_per_token = new_log_probs - ref_log_probs_t
    kl_div = kl_per_token.sum(dim=-1).mean().item()

    old_probs = ref_log_probs_t.sum(dim=-1).exp().tolist()
    new_probs = new_log_probs.sum(dim=-1).exp().tolist()

    return build_step_data(
        step=step,
        game_state_tokens=state_tokens,
        game_state_text=state_text,
        completion_texts=guesses,
        completion_token_lists=[list(g) for g in guesses],
        log_probs_per_completion=new_log_probs.tolist(),
        rewards=rewards,
        reward_breakdowns=reward_breakdowns,
        advantages=advantages.tolist(),
        group_mean=rewards_tensor.mean().item(),
        group_std=rewards_tensor.std().item(),
        old_probs=old_probs,
        new_probs=new_probs,
        kl_divergence=kl_div,
    )
