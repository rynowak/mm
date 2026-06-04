"""V2 data pipeline: constraint-state prompts with character-level targets."""

from __future__ import annotations

import random as rng
from typing import TYPE_CHECKING

import torch
from mm_wordle.game import GameState, GuessFeedback, LetterFeedback, WordleEnv
from mm_wordle.transcripts import generate_examples
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from wordle2.tokenizer import ConstraintTokenizer


class ConstraintDataset(Dataset):
    """Dataset of (constraint_state, target_chars) pairs.

    Each item returns (input_ids, target_ids, loss_mask):
      - input_ids: constraint state + target[:-1]
      - target_ids: constraint state[1:] + target
      - loss_mask: 1 for target character positions, 0 elsewhere
    """

    def __init__(
        self,
        prompts: list[torch.Tensor],
        targets: list[torch.Tensor],
        pad_id: int,
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for prompt, target in zip(prompts, targets, strict=True):
            full_seq = torch.cat([prompt, target])
            input_ids = full_seq[:-1]
            target_ids = full_seq[1:]

            loss_mask = torch.zeros_like(target_ids)
            prompt_len = len(prompt)
            loss_mask[prompt_len - 1 :] = 1.0

            self.examples.append((input_ids, target_ids, loss_mask))

        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_constraint(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=pad_id)
    loss_masks = pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0.0)
    return input_ids, target_ids, loss_masks


def _oversample_late_turns(
    examples: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """Oversample later turns."""
    by_turn: dict[int, list[tuple[torch.Tensor, torch.Tensor, int]]] = {}
    for item in examples:
        turn = item[2]
        by_turn.setdefault(turn, []).append(item)

    result: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for turn, items in by_turn.items():
        result.extend(items * turn)

    rng.shuffle(result)
    return result


def load_constraint_data(
    config: dict,
    tokenizer: ConstraintTokenizer,
) -> tuple[ConstraintDataset, ConstraintDataset]:
    """Generate training data with constraint-state encoding."""
    from mm_tokenizers import CharTokenizer

    val_fraction = config["data"]["val_fraction"]
    n_games = config["data"].get("transcript_games", 20000)

    char_tok = CharTokenizer()
    print(f"Generating {n_games} Wordle game transcripts...")
    raw_examples = generate_examples(char_tok, n_games=n_games)
    print(f"  Generated {len(raw_examples)} per-turn examples from {n_games} games")

    env = WordleEnv()
    examples: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    for ex in raw_examples:
        target_word = char_tok.decode(ex.target_ids)
        prompt_text = char_tok.decode(ex.prompt_ids)

        state = _reconstruct_state(prompt_text, env)
        prompt_ids = tokenizer.encode_game_state(state)
        target_ids = [tokenizer.encode_token(ch) for ch in target_word]

        turn = len(state.guesses) + 1

        examples.append(
            (
                torch.tensor(prompt_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long),
                turn,
            )
        )

    examples = _oversample_late_turns(examples)
    print(f"  After oversampling late turns: {len(examples)} examples")

    prompts = [e[0] for e in examples]
    targets = [e[1] for e in examples]

    n_val = int(len(examples) * val_fraction)
    n_train = len(examples) - n_val

    train_ds = ConstraintDataset(prompts[:n_train], targets[:n_train], tokenizer.pad_id)
    val_ds = ConstraintDataset(prompts[n_train:], targets[n_train:], tokenizer.pad_id)

    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    return train_ds, val_ds


def _reconstruct_state(prompt_text: str, env: WordleEnv) -> GameState:
    """Reconstruct a GameState from V1 prompt text.

    V1 prompt format: [bos] or [bos]guess+feedback[sep]...
    """
    parts = prompt_text.replace("[bos]", "").split("[sep]")
    parts = [p for p in parts if p]

    if not parts:
        return GameState(target="?????")

    target_placeholder = "?????"
    state = env.reset(target_word=target_placeholder)

    for part in parts:
        guess = ""
        feedback_strs: list[str] = []
        i = 0
        while i < len(part):
            if part[i] == "[":
                end = part.index("]", i)
                token = part[i : end + 1]
                feedback_strs.append(token)
                i = end + 1
            else:
                guess += part[i]
                i += 1

        if len(guess) != 5 or len(feedback_strs) != 5:
            continue

        feedback = []
        for fs in feedback_strs:
            if fs == "[green]":
                feedback.append(LetterFeedback.GREEN)
            elif fs == "[yellow]":
                feedback.append(LetterFeedback.YELLOW)
            else:
                feedback.append(LetterFeedback.GRAY)

        gf = GuessFeedback(guess=guess, feedback=feedback)
        state = GameState(
            target=state.target,
            guesses=[*state.guesses, gf],
            turn=state.turn + 1,
            solved=False,
            failed=state.turn + 1 >= 6,
        )

    return state
