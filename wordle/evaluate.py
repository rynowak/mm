"""Evaluate a Wordle-playing model and generate visualizations.

Usage:
    uv run python wordle/evaluate.py --checkpoint path/to/checkpoint.pt --num-games 100
    uv run python wordle/evaluate.py --checkpoint path/to/checkpoint.pt --interactive
    uv run python wordle/evaluate.py --checkpoint path/to/checkpoint.pt --report output.html
    uv run python wordle/evaluate.py --compare ckpt1.pt ckpt2.pt --report comparison.html
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_training import get_device, seed_everything
from mm_viz import EvalSnapshot, GameReplay, render_comparison_html, render_games_report
from mm_wordle import WordleEnv, WordTrie, all_valid_words, game_state_to_prompt, load_answers

if TYPE_CHECKING:
    from mm_wordle.game import GameState


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, GPTConfig]:
    """Load model from checkpoint.

    Returns the model (in eval mode on the given device) and its config.
    """
    path = pathlib.Path(checkpoint_path)
    checkpoint = load_checkpoint(path, device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_guess_unconstrained(
    model: GPT,
    tokenizer: CharTokenizer,
    game_state: GameState,
    device: torch.device,
) -> str:
    """Generate a 5-character guess autoregressively without constraints."""
    state_tokens = game_state_to_prompt(game_state)
    state_ids = tokenizer.encode("".join(state_tokens))
    prompt = torch.tensor([state_ids], dtype=torch.long, device=device)
    output = model.generate(prompt, max_new_tokens=5, temperature=0.1, top_k=5)
    generated_ids = output[0, len(state_ids) :].tolist()
    try:
        text = tokenizer.decode(generated_ids)
    except ValueError:
        text = ""
    letters = [ch for ch in text if "a" <= ch <= "z"]
    return "".join(letters[:5]).ljust(5, "a")


@torch.no_grad()
def generate_guess_constrained(
    model: GPT,
    tokenizer: CharTokenizer,
    game_state: GameState,
    trie: WordTrie,
    device: torch.device,
) -> str:
    """Generate a 5-character guess using trie-constrained decoding.

    Same algorithm as finetune.py's sample_constrained: mask logits at each
    position to only allow characters that continue a valid word in the trie.
    Uses greedy decoding (low temperature) for deterministic evaluation.
    """
    state_tokens = game_state_to_prompt(game_state)
    state_ids = tokenizer.encode("".join(state_tokens))
    prompt = torch.tensor([state_ids], dtype=torch.long, device=device)
    prefix = ""
    vocab_size = model.config.vocab_size

    logits, _, kv_cache = model(prompt)
    logits = logits[:, -1, :] / 0.1

    valid_chars = trie.valid_next_chars(prefix)
    if not valid_chars:
        return "aaaaa"

    mask = torch.full((1, vocab_size), float("-inf"), device=device)
    for ch in valid_chars:
        token_ids = tokenizer.encode(ch)
        if token_ids:
            mask[0, token_ids[0]] = 0.0
    logits = logits + mask
    probs = F.softmax(logits, dim=-1)
    next_token = probs.argmax(dim=-1, keepdim=True)

    try:
        ch = tokenizer.decode([int(next_token.item())])
        prefix += ch
    except ValueError:
        prefix += "?"

    for pos in range(4):
        logits, _, kv_cache = model(next_token, kv_cache=kv_cache, start_pos=len(state_ids) + pos)
        logits = logits[:, -1, :] / 0.1

        valid_chars = trie.valid_next_chars(prefix)
        if not valid_chars:
            break

        mask = torch.full((1, vocab_size), float("-inf"), device=device)
        for ch in valid_chars:
            token_ids = tokenizer.encode(ch)
            if token_ids:
                mask[0, token_ids[0]] = 0.0
        logits = logits + mask
        probs = F.softmax(logits, dim=-1)
        next_token = probs.argmax(dim=-1, keepdim=True)

        try:
            ch = tokenizer.decode([int(next_token.item())])
            prefix += ch
        except ValueError:
            prefix += "?"

    return prefix.ljust(5, "a")[:5]


def select_guess(
    model: GPT,
    tokenizer: CharTokenizer,
    game_state: GameState,
    device: torch.device,
    decoding: str,
    trie: WordTrie,
) -> str:
    """Select the next guess using the model.

    Constrained mode uses trie-masked decoding (same as training).
    Unconstrained mode generates freely.
    """
    if decoding == "constrained":
        return generate_guess_constrained(model, tokenizer, game_state, trie, device)
    return generate_guess_unconstrained(model, tokenizer, game_state, device)


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------


def play_game(
    model: GPT,
    tokenizer: CharTokenizer,
    env: WordleEnv,
    device: torch.device,
    decoding: str,
    trie: WordTrie,
    target_word: str | None = None,
) -> GameReplay:
    """Play a single Wordle game and return a replay."""
    state = env.reset(target_word=target_word)
    target = state.target

    guesses: list[str] = []
    feedback: list[list[str]] = []

    done = False
    while not done:
        guess = select_guess(model, tokenizer, state, device, decoding, trie)
        state, done = env.step(state, guess)

        last_fb = state.guesses[-1]
        guesses.append(last_fb.guess)
        feedback.append([fb.value for fb in last_fb.feedback])

    return GameReplay(
        target=target,
        guesses=guesses,
        feedback=feedback,
        solved=state.solved,
        turns=state.turn,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Computed metrics from an evaluation run."""

    num_games: int
    wins: int
    win_rate: float
    avg_guesses_winners: float
    guess_distribution: dict[int, int]  # turn -> count (7 = failed)
    first_guesses: Counter[str]
    decoding: str


def compute_metrics(replays: list[GameReplay], decoding: str) -> EvalMetrics:
    """Compute evaluation metrics from game replays."""
    num_games = len(replays)
    wins = sum(1 for r in replays if r.solved)
    win_rate = wins / num_games if num_games else 0.0

    winner_guesses = [r.turns for r in replays if r.solved]
    avg_guesses_winners = sum(winner_guesses) / len(winner_guesses) if winner_guesses else 0.0

    # Guess distribution: 1-6 for solved, 7 for failed
    dist: dict[int, int] = {i: 0 for i in range(1, 8)}
    for r in replays:
        if r.solved:
            dist[r.turns] = dist.get(r.turns, 0) + 1
        else:
            dist[7] = dist.get(7, 0) + 1

    first_guesses: Counter[str] = Counter()
    for r in replays:
        if r.guesses:
            first_guesses[r.guesses[0]] += 1

    return EvalMetrics(
        num_games=num_games,
        wins=wins,
        win_rate=win_rate,
        avg_guesses_winners=avg_guesses_winners,
        guess_distribution=dist,
        first_guesses=first_guesses,
        decoding=decoding,
    )


def print_metrics(metrics: EvalMetrics) -> None:
    """Print evaluation results to stdout."""
    bar_width = 30
    max_count = max(metrics.guess_distribution.values()) if metrics.guess_distribution else 1

    print()
    print(f"Evaluation Results ({metrics.num_games} games, {metrics.decoding} decoding)")
    print("━" * 50)
    print(f"Win rate:          {metrics.win_rate:.1%}")
    if metrics.avg_guesses_winners > 0:
        print(f"Avg guesses:       {metrics.avg_guesses_winners:.1f} (winners only)")
    else:
        print("Avg guesses:       N/A (no wins)")
    print("Guess distribution:")

    for turn in range(1, 7):
        count = metrics.guess_distribution.get(turn, 0)
        bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {turn}: {bar} {count}")

    failed = metrics.guess_distribution.get(7, 0)
    bar_len = int(bar_width * failed / max_count) if max_count > 0 else 0
    bar = "█" * bar_len
    print(f"  X: {bar} {failed}")

    print()
    unique_first = len(metrics.first_guesses)
    print(f"First guess diversity: {unique_first} unique first guesses")
    if metrics.first_guesses:
        most_common_word, most_common_count = metrics.first_guesses.most_common(1)[0]
        print(f'Most common first guess: "{most_common_word}" ({most_common_count} times)')
    print()


def evaluate(
    model: GPT,
    tokenizer: CharTokenizer,
    env: WordleEnv,
    device: torch.device,
    num_games: int,
    decoding: str,
    trie: WordTrie,
    target_words: list[str] | None = None,
    checkpoint_path: str = "",
) -> EvalSnapshot:
    """Play num_games games and return metrics + replays.

    If target_words is provided, those words are used as targets (cycling if
    num_games > len(target_words)). Otherwise random answers are used.
    """
    replays: list[GameReplay] = []

    for i in range(num_games):
        target = target_words[i % len(target_words)] if target_words else None

        replay = play_game(model, tokenizer, env, device, decoding, trie, target_word=target)
        replays.append(replay)

        status = "solved" if replay.solved else "failed"
        sys.stdout.write(f"\rPlaying game {i + 1}/{num_games} ... {status} ({replay.target})")
        sys.stdout.flush()

    print()  # newline after progress

    # Compute summary stats for EvalSnapshot
    wins = sum(1 for r in replays if r.solved)
    win_rate = wins / num_games if num_games else 0.0
    winner_guesses = [r.turns for r in replays if r.solved]
    avg_guesses = sum(winner_guesses) / len(winner_guesses) if winner_guesses else 0.0

    # Extract step from checkpoint filename if possible
    step = _extract_step(checkpoint_path)

    return EvalSnapshot(
        step=step,
        checkpoint_path=checkpoint_path,
        win_rate=win_rate,
        avg_guesses=avg_guesses,
        replays=replays,
    )


def _extract_step(checkpoint_path: str) -> int:
    """Try to extract the training step from a checkpoint path.

    Looks for patterns like 'checkpoint-1000/model.pt'.
    Returns 0 if no step can be determined.
    """
    path = pathlib.Path(checkpoint_path)
    for part in reversed(path.parts):
        if part.startswith("checkpoint-"):
            try:
                return int(part.split("-", 1)[1])
            except (ValueError, IndexError):
                pass
    return 0


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def interactive_mode(
    model: GPT,
    tokenizer: CharTokenizer,
    env: WordleEnv,
    device: torch.device,
    decoding: str,
    trie: WordTrie,
) -> None:
    """Interactive loop: user picks a word, model plays, shows the board."""
    answers = set(load_answers())
    all_words = all_valid_words()

    print()
    print("Wordle -- You pick the word, the model plays!")
    print("=" * 50)

    while True:
        print()
        word = input("Enter a 5-letter word (or 'random' for a random word, 'quit' to exit): ").strip().lower()

        if word == "quit":
            print("Thanks for playing!")
            break

        if word == "random":
            word = random.choice(list(answers))
            print(f"Random word: {word}")

        if len(word) != 5 or not word.isalpha():
            print("Please enter exactly 5 letters.")
            continue

        if word not in all_words:
            print(f'"{word}" is not in the word list. Try another.')
            continue

        # Play the game
        state = env.reset(target_word=word)
        done = False
        turn = 0

        print()
        while not done:
            guess = select_guess(model, tokenizer, state, device, decoding, trie)
            state, done = env.step(state, guess)
            turn += 1

            last_fb = state.guesses[-1]
            feedback_strs: list[str] = []
            for ch, fb in zip(last_fb.guess, last_fb.feedback, strict=True):
                feedback_strs.append(f"{ch.upper()}[{fb.value}]")
            feedback_display = " ".join(feedback_strs)
            print(f'  Turn {turn}: Model guesses "{last_fb.guess}" -> {feedback_display}')

        print()
        if state.solved:
            print(f"Model solved it in {state.turn} guess{'es' if state.turn != 1 else ''}!")
        else:
            print(f"Model failed! The word was {state.target.upper()}.")

        print()
        again = input("Play again? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------


def run_comparison(
    checkpoint_paths: list[str],
    num_games: int,
    decoding: str,
    seed: int,
    device: torch.device,
    trie: WordTrie,
    target_words: list[str] | None = None,
) -> list[EvalSnapshot]:
    """Run evaluation on multiple checkpoints with the same target words.

    Uses the same seed so each checkpoint plays against the same targets.
    """
    tokenizer = CharTokenizer()
    env = WordleEnv()

    if target_words is None:
        seed_everything(seed)
        answers = load_answers()
        target_words = [random.choice(answers) for _ in range(num_games)]

    snapshots: list[EvalSnapshot] = []
    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating checkpoint: {ckpt_path}")
        model, _config = load_model(ckpt_path, device)
        snapshot = evaluate(
            model,
            tokenizer,
            env,
            device,
            num_games,
            decoding,
            trie,
            target_words=target_words,
            checkpoint_path=ckpt_path,
        )
        metrics = compute_metrics(snapshot.replays, decoding)
        print_metrics(metrics)
        snapshots.append(snapshot)

    return snapshots


def render_comparison_report(snapshots: list[EvalSnapshot], checkpoint_paths: list[str]) -> str:
    """Generate a self-contained HTML comparison report.

    Shows side-by-side game boards for each target word across checkpoints,
    plus summary statistics.
    """
    labels = [pathlib.Path(p).parent.name for p in checkpoint_paths]

    # Build per-target comparison sections
    sections: list[str] = []

    # Summary table
    summary_rows: list[str] = []
    for label, snap in zip(labels, snapshots, strict=True):
        wins = sum(1 for r in snap.replays if r.solved)
        total = len(snap.replays)
        summary_rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{snap.win_rate:.1%}</td>"
            f"<td>{snap.avg_guesses:.1f}</td>"
            f"<td>{wins}/{total}</td></tr>"
        )

    summary_table = (
        '<table style="border-collapse:collapse;margin:16px 0;">'
        "<tr><th>Checkpoint</th><th>Win Rate</th><th>Avg Guesses</th><th>Wins/Total</th></tr>"
        + "".join(summary_rows)
        + "</table>"
    )
    sections.append(summary_table)

    # Side-by-side boards for each game (show first 20 for readability)
    max_display = min(20, len(snapshots[0].replays))
    for game_idx in range(max_display):
        replays_for_game = [snap.replays[game_idx] for snap in snapshots]
        target = replays_for_game[0].target
        game_labels = [
            f"{label} ({'solved' if r.solved else 'failed'})" for label, r in zip(labels, replays_for_game, strict=True)
        ]
        comparison_html = render_comparison_html(replays_for_game, game_labels)
        sections.append(
            f'<div style="margin:16px 0;"><h3>Game {game_idx + 1} -- target: {target}</h3>{comparison_html}</div>'
        )

    body = "\n".join(sections)
    return (
        "<html><head><title>Wordle Model Comparison</title>"
        "<style>table { font-family: sans-serif; } th, td { padding: 4px 12px; "
        "text-align: left; border-bottom: 1px solid #ddd; }</style></head>"
        f'<body style="font-family:sans-serif;padding:24px;">'
        f"<h1>Wordle Model Comparison</h1>{body}</body></html>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Wordle-playing model and generate visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python wordle/evaluate.py --checkpoint path/to/model.pt --num-games 100\n"
            "  uv run python wordle/evaluate.py --checkpoint path/to/model.pt --interactive\n"
            "  uv run python wordle/evaluate.py --checkpoint path/to/model.pt --report output.html\n"
            "  uv run python wordle/evaluate.py --compare ckpt1.pt ckpt2.pt --report comparison.html\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required unless --compare)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        metavar="PATH",
        help="Compare multiple checkpoints",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of evaluation games (default: 100)",
    )
    parser.add_argument(
        "--target-words",
        type=str,
        default=None,
        metavar="PATH",
        help="File with specific target words, one per line",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="constrained",
        choices=["constrained", "unconstrained"],
        help="Decoding mode (default: constrained)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="PATH",
        help="Generate HTML report at this path",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Play Wordle interactively against the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Validate arguments
    if args.compare is None and args.checkpoint is None:
        print("Error: --checkpoint or --compare is required.", file=sys.stderr)
        sys.exit(1)

    if args.compare is not None and args.interactive:
        print("Error: --interactive cannot be used with --compare.", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    # Load target words from file if specified
    target_words: list[str] | None = None
    if args.target_words:
        path = pathlib.Path(args.target_words)
        target_words = [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(target_words)} target words from {path}")

    # Build trie for constrained decoding
    trie = WordTrie.from_words(load_answers())

    # --- Comparison mode ---
    if args.compare is not None:
        snapshots = run_comparison(
            args.compare,
            args.num_games,
            args.decoding,
            args.seed,
            device,
            trie,
            target_words=target_words,
        )
        if args.report:
            html = render_comparison_report(snapshots, args.compare)
            report_path = pathlib.Path(args.report)
            report_path.write_text(html)
            print(f"Comparison report saved to {report_path}")
        return

    # --- Single checkpoint mode ---
    assert args.checkpoint is not None
    model, _config = load_model(args.checkpoint, device)
    tokenizer = CharTokenizer()
    env = WordleEnv()

    # Interactive mode
    if args.interactive:
        seed_everything(args.seed)
        interactive_mode(model, tokenizer, env, device, args.decoding, trie)
        return

    # Evaluation mode
    seed_everything(args.seed)
    snapshot = evaluate(
        model,
        tokenizer,
        env,
        device,
        args.num_games,
        args.decoding,
        trie,
        target_words=target_words,
        checkpoint_path=args.checkpoint,
    )

    metrics = compute_metrics(snapshot.replays, args.decoding)
    print_metrics(metrics)

    # Generate HTML report if requested
    if args.report:
        html = render_games_report(snapshot.replays, title=f"Wordle Evaluation ({args.decoding} decoding)")
        report_path = pathlib.Path(args.report)
        report_path.write_text(html)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
