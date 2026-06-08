"""Per-step metric engine: valid-word rate, info gain, win rate (§5.9).

Plays a set of games greedily (letter-masked decode) and scores the trio using the
precomputed pattern matrix for info gain and candidate filtering. Shared by all
phases — pre-train/SFT call it as the per-step mini-eval; RL reuses it for eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mm_wordle import WordleEnv
from mm_wordle.solver import filter_candidates

if TYPE_CHECKING:
    from mm_model import GPT
    from mm_wordle import ConstraintTokenizer, PatternMatrix


@dataclass
class EvalResult:
    valid_word_rate: float
    info_gain: float  # mean expected info gain over guesses (candidates > 1)
    win_rate: float
    avg_guesses: float
    by_turn: dict[int, dict[str, float]] = field(default_factory=dict)
    replays: list[dict] = field(default_factory=list)


def build_letter_mask(tokenizer: ConstraintTokenizer, device: torch.device) -> torch.Tensor:
    """A logit-additive mask that is 0 on plain-letter tokens and -inf elsewhere."""
    mask = torch.full((tokenizer.vocab_size,), float("-inf"), device=device)
    for lid in tokenizer.letter_ids:
        mask[lid] = 0.0
    return mask


@torch.no_grad()
def _greedy_guess(model: GPT, prompt_ids: list[int], device: torch.device, letter_mask: torch.Tensor) -> list[int]:
    si = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []
    for _ in range(5):
        logits, _, _ = model(si)
        next_id = int(torch.argmax(logits[0, -1, :] + letter_mask).item())
        generated.append(next_id)
        si = torch.cat([si, torch.tensor([[next_id]], device=device)], dim=1)
    return generated


@dataclass
class OpenerMetrics:
    valid_word_rate: float
    info_gain: float  # mean expected info gain of valid openers over the full universe
    distinct_valid: int


@torch.no_grad()
def sample_openers(
    model: GPT,
    tokenizer: ConstraintTokenizer,
    pattern_matrix: PatternMatrix,
    device: torch.device,
    n_samples: int = 64,
    temperature: float = 1.0,
) -> OpenerMetrics:
    """Cheap per-step pre-train metric: sample words from the empty prompt.

    All samples share the fixed-length empty prompt, so this is a single batched
    decode (no padding). Returns the validity rate, mean opener info gain over the
    full universe, and the number of distinct valid words (mode-collapse guard).
    """
    was_training = model.training
    model.eval()
    letter_mask = build_letter_mask(tokenizer, device)
    prompt = torch.tensor(tokenizer.empty_prompt(), dtype=torch.long, device=device)
    si = prompt.unsqueeze(0).expand(n_samples, -1).contiguous()

    cols: list[torch.Tensor] = []
    for _ in range(5):
        logits, _, _ = model(si)
        next_logits = logits[:, -1, :] / temperature + letter_mask
        nxt = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
        si = torch.cat([si, nxt], dim=1)
        cols.append(nxt)
    gen = torch.cat(cols, dim=1).tolist()

    words = [tokenizer.decode_letters(row) for row in gen]
    valid = [w for w in words if w in pattern_matrix.guess_index]
    full_idx = np.arange(len(pattern_matrix.targets))
    igs = [pattern_matrix.expected_info_gain(w, full_idx) for w in valid]

    if was_training:
        model.train()
    return OpenerMetrics(
        valid_word_rate=len(valid) / max(len(words), 1),
        info_gain=float(sum(igs) / len(igs)) if igs else 0.0,
        distinct_valid=len(set(valid)),
    )


@torch.no_grad()
def play_games(
    model: GPT,
    tokenizer: ConstraintTokenizer,
    pattern_matrix: PatternMatrix,
    targets: list[str],
    device: torch.device,
    max_replays: int = 8,
) -> EvalResult:
    """Play ``targets`` greedily and return the trio + per-turn detail + sample replays."""
    was_training = model.training
    model.eval()
    env = WordleEnv()
    letter_mask = build_letter_mask(tokenizer, device)
    universe = pattern_matrix.targets
    n_words = len(universe)
    valid_index = pattern_matrix.guess_index

    n_valid = 0
    n_guesses = 0
    ig_values: list[float] = []
    wins = 0
    total_turns = 0
    by_turn: dict[int, dict[str, float]] = {}
    replays: list[dict] = []

    for target in targets:
        state = env.reset(target_word=target)
        candidate_idx = np.arange(n_words)
        guesses: list[str] = []
        feedback_rows: list[list[str]] = []
        turn_rewards: list[float] = []

        while not state.solved and not state.failed:
            turn = state.turn + 1
            prompt_ids = tokenizer.encode_game_state(state)
            guess = tokenizer.decode_letters(_greedy_guess(model, prompt_ids, device, letter_mask))
            if len(guess) != 5:
                guess = "zzzzz"
            valid = guess in valid_index

            n_guesses += 1
            n_valid += int(valid)
            bt = by_turn.setdefault(turn, {"valid": 0.0, "n": 0.0, "ig_sum": 0.0, "ig_n": 0.0})
            bt["valid"] += float(valid)
            bt["n"] += 1.0

            turn_ig = 0.0
            if valid and len(candidate_idx) > 1:
                turn_ig = pattern_matrix.expected_info_gain(guess, candidate_idx)
                ig_values.append(turn_ig)
                bt["ig_sum"] += turn_ig
                bt["ig_n"] += 1.0

            state, _ = env.step(state, guess)
            fb = state.guesses[-1].feedback
            guesses.append(guess)
            feedback_rows.append([f.value for f in fb])
            turn_rewards.append(round(turn_ig, 3))

            if valid:
                observed = pattern_matrix.pattern_id(guess, target)
                candidate_idx = pattern_matrix.consistent_idx(guess, observed, candidate_idx)
            else:
                kept = set(filter_candidates([universe[i] for i in candidate_idx], guess, fb))
                candidate_idx = np.array([i for i in candidate_idx if universe[i] in kept], dtype=np.int64)

        wins += int(state.solved)
        total_turns += state.turn
        if len(replays) < max_replays:
            replays.append(
                {
                    "target": target,
                    "guesses": guesses,
                    "feedback": feedback_rows,
                    "solved": state.solved,
                    "turns": state.turn,
                    "turn_rewards": turn_rewards,  # per-guess expected info gain (bits)
                }
            )

    if was_training:
        model.train()

    n = max(len(targets), 1)
    by_turn_out = {
        t: {
            "valid_word_rate": d["valid"] / d["n"] if d["n"] else 0.0,
            "info_gain": d["ig_sum"] / d["ig_n"] if d["ig_n"] else 0.0,
        }
        for t, d in sorted(by_turn.items())
    }
    return EvalResult(
        valid_word_rate=n_valid / max(n_guesses, 1),
        info_gain=float(sum(ig_values) / len(ig_values)) if ig_values else 0.0,
        win_rate=wins / n,
        avg_guesses=total_turns / n,
        by_turn=by_turn_out,
        replays=replays,
    )
