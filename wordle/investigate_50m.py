"""Investigate the 50M model's failing games."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from decoding import sample_unconstrained
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_wordle import WordleEnv, load_answers
from mm_wordle.game import LetterFeedback
from mm_wordle.reward import _compute_expected_info_gain
from mm_wordle.serialize import game_state_to_prompt
from mm_wordle.solver import filter_candidates


def load_model(path: str, device: torch.device) -> GPT:
    ckpt = load_checkpoint(Path(path), device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def play_game(
    model: GPT,
    env: WordleEnv,
    target: str,
    tokenizer: CharTokenizer,
    answers: list[str],
    device: torch.device,
    temperature: float = 0.1,
) -> dict:
    state = env.reset(target_word=target)
    candidates = list(answers)
    turns = []
    all_valid = set(answers)

    while not state.solved and not state.failed:
        st = game_state_to_prompt(state)
        si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)
        samples = sample_unconstrained(model, si, device, tokenizer, n_samples=1, temperature=temperature)
        guess = samples[0][0]

        n_before = len(candidates)
        consistent = guess in candidates
        is_valid_word = guess in all_valid
        is_repeat = guess in [t["guess"] for t in turns]

        new_state, _ = env.step(state, guess)
        feedback = new_state.guesses[-1].feedback

        if n_before > 1 and len(guess) == 5 and len(feedback) == 5:
            exp_ig = _compute_expected_info_gain(guess, candidates)
            after = filter_candidates(candidates, guess, feedback)
            actual_ig = math.log2(n_before / max(len(after), 1))
        else:
            exp_ig = 0.0
            actual_ig = 0.0
            after = candidates if len(guess) != 5 else filter_candidates(candidates, guess, feedback)

        greens = sum(1 for f in feedback if f == LetterFeedback.GREEN)
        yellows = sum(1 for f in feedback if f == LetterFeedback.YELLOW)

        turns.append(
            {
                "turn": state.turn + 1,
                "guess": guess,
                "consistent": consistent,
                "valid_word": is_valid_word,
                "repeat": is_repeat,
                "candidates_before": n_before,
                "candidates_after": len(after),
                "expected_ig": round(exp_ig, 3),
                "actual_ig": round(actual_ig, 3),
                "greens": greens,
                "yellows": yellows,
                "feedback": [f.value for f in feedback],
            }
        )

        candidates = after
        state = new_state

    return {
        "target": target,
        "solved": state.solved,
        "turns_count": state.turn,
        "guesses": [t["guess"] for t in turns],
        "turns": turns,
    }


def classify_failure(game: dict) -> str:
    """Classify why a game was lost."""
    turns = game["turns"]

    # Check for invalid words
    invalid_count = sum(1 for t in turns if not t["valid_word"])
    if invalid_count >= 2:
        return "invalid_words"

    # Check for repeated guesses
    repeat_count = sum(1 for t in turns if t["repeat"])
    if repeat_count >= 2:
        return "repetition"

    # Check for inconsistent guesses (ignoring feedback)
    inconsistent_late = sum(1 for t in turns[2:] if not t["consistent"])
    if inconsistent_late >= 3:
        return "ignores_feedback"

    # Check if candidates narrowed to 1-2 but couldn't close
    final_cands = turns[-1]["candidates_before"]
    if final_cands <= 2:
        return "cant_close"

    # Check if candidates stayed high (bad information gathering)
    if turns[1]["candidates_after"] > 50:
        return "poor_info_gain"

    return "other"


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = CharTokenizer()
    answers = load_answers()

    random.seed(99)
    eval_words = random.sample(answers, 200)

    print("Loading 50M RL model...")
    model = load_model("runs/finetune-grpo/20260603_062400/checkpoint-3000/model.pt", device)

    print("Playing 200 games (temp=0.1)...")
    torch.manual_seed(42)
    games = []
    for target in eval_words:
        game = play_game(model, WordleEnv(), target, tokenizer, answers, device, 0.1)
        games.append(game)

    wins = [g for g in games if g["solved"]]
    losses = [g for g in games if not g["solved"]]
    print(f"\nResults: {len(wins)}/{len(games)} wins ({len(wins) / len(games):.1%})")
    print(f"Losses: {len(losses)}")

    # Win statistics
    if wins:
        win_turns = [g["turns_count"] for g in wins]
        print(f"\nWin stats: avg={sum(win_turns) / len(win_turns):.1f}, min={min(win_turns)}, max={max(win_turns)}")
        turn_dist = Counter(win_turns)
        for t in sorted(turn_dist):
            print(f"  Solved in {t}: {turn_dist[t]} games")

    # Classify failures
    print("\n=== Failure Classification ===")
    failure_types = Counter()
    failure_examples: dict[str, list] = {}
    for game in losses:
        ftype = classify_failure(game)
        failure_types[ftype] += 1
        if ftype not in failure_examples:
            failure_examples[ftype] = []
        if len(failure_examples[ftype]) < 3:
            failure_examples[ftype].append(game)

    for ftype, count in failure_types.most_common():
        print(f"  {ftype}: {count} ({count / len(losses):.0%})")

    # Show examples per failure type
    for ftype, examples in failure_examples.items():
        print(f"\n--- {ftype} examples ---")
        for game in examples:
            print(f"  Target: {game['target']}")
            for t in game["turns"]:
                flags = []
                if not t["valid_word"]:
                    flags.append("INVALID")
                if not t["consistent"]:
                    flags.append("INCONSISTENT")
                if t["repeat"]:
                    flags.append("REPEAT")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                fb_str = "".join(f[0].upper() for f in t["feedback"])
                print(
                    f"    t{t['turn']}: {t['guess']} {fb_str} "
                    f"cands={t['candidates_before']}→{t['candidates_after']} "
                    f"ig={t['actual_ig']:.1f}{flag_str}"
                )
            print()

    # Consistency analysis
    print("=== Consistency by Turn ===")
    by_turn: dict[int, dict[str, int]] = {}
    for game in games:
        for t in game["turns"]:
            tn = t["turn"]
            if tn not in by_turn:
                by_turn[tn] = {"total": 0, "consistent": 0, "valid": 0, "repeat": 0}
            by_turn[tn]["total"] += 1
            if t["consistent"]:
                by_turn[tn]["consistent"] += 1
            if t["valid_word"]:
                by_turn[tn]["valid"] += 1
            if t["repeat"]:
                by_turn[tn]["repeat"] += 1

    print(f"  {'Turn':<6} {'Consistent':>12} {'Valid Word':>12} {'Repeats':>10}")
    for tn in sorted(by_turn):
        bt = by_turn[tn]
        n = bt["total"]
        print(f"  {tn:<6} {bt['consistent'] / n:>11.1%} {bt['valid'] / n:>11.1%} {bt['repeat'] / n:>9.1%}  (n={n})")

    # Valid word rate overall
    total_guesses = sum(len(g["turns"]) for g in games)
    valid_guesses = sum(1 for g in games for t in g["turns"] if t["valid_word"])
    invalid_examples = [t["guess"] for g in games for t in g["turns"] if not t["valid_word"]]
    print(f"\nValid word rate: {valid_guesses}/{total_guesses} ({valid_guesses / total_guesses:.1%})")
    if invalid_examples:
        inv_counter = Counter(invalid_examples)
        print(f"  Invalid word examples: {inv_counter.most_common(10)}")

    # First guess analysis
    first_guesses = [g["turns"][0]["guess"] for g in games]
    fg_counter = Counter(first_guesses)
    print(f"\nFirst guesses: {len(fg_counter)} unique")
    avg_ig = sum(g["turns"][0]["expected_ig"] for g in games) / len(games)
    print(f"  Avg info gain: {avg_ig:.2f} bits")
    print(f"  Top 5: {fg_counter.most_common(5)}")

    # Save data
    data_dir = Path("runs/investigation")
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "50m_games.json", "w") as f:
        json.dump(games, f, indent=2)
    print("\nSaved to runs/investigation/50m_games.json")


if __name__ == "__main__":
    main()
