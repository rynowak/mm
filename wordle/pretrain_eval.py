"""Pre-training evaluation criteria.

Runs the checks defined in docs/pretrain-eval.md to determine if the
pre-trained model is ready for RL fine-tuning.

Usage:
    uv run python wordle/pretrain_eval.py --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_training import get_device, seed_everything
from mm_wordle import WordleEnv, all_valid_words, game_state_to_prompt, load_answers


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    path = Path(checkpoint_path)
    checkpoint = load_checkpoint(path, device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_words(
    model: GPT, tokenizer: CharTokenizer, device: torch.device, prompt_ids: list[int], n: int
) -> list[str]:
    """Generate n five-character words from a prompt (unconstrained)."""
    words = []
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(n):
        output = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=40)
        generated = output[0, len(prompt_ids) :].tolist()
        try:
            text = tokenizer.decode(generated)
        except ValueError:
            text = ""
        letters = [ch for ch in text if "a" <= ch <= "z"]
        words.append("".join(letters[:5]).ljust(5, "?"))
    return words


@torch.no_grad()
def top_predictions(
    model: GPT, tokenizer: CharTokenizer, device: torch.device, prompt_ids: list[int], k: int = 10
) -> list[str]:
    """Return the top-k predicted token strings from a prompt."""
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _ = model(prompt)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_ids = probs.topk(k).indices.tolist()
    result = []
    for tid in top_ids:
        try:
            result.append(tokenizer.decode([tid]))
        except ValueError:
            result.append(f"<id={tid}>")
    return result


def run_eval(checkpoint_path: str) -> bool:
    device = get_device()
    seed_everything(42)
    print(f"Device: {device}")

    model = load_model(checkpoint_path, device)
    tokenizer = CharTokenizer()
    valid_words = all_valid_words()
    answers = set(load_answers())
    env = WordleEnv()

    all_passed = True

    # === Criterion 1: Valid words from [bos] ===
    print("\n=== Criterion 1: Generates valid Wordle words from [bos] ===")
    words = generate_words(model, tokenizer, device, [tokenizer.bos_id], 100)
    valid_count = sum(1 for w in words if w in valid_words)
    answer_count = sum(1 for w in words if w in answers)
    rate = valid_count / 100
    print(f"  Valid words: {valid_count}/100 ({rate:.0%})")
    print(f"  Answer words: {answer_count}/100")
    print(f"  Sample: {words[:10]}")
    if rate >= 0.30:
        print("  PASS (>= 30%)")
    else:
        print("  FAIL (< 30%)")
        all_passed = False

    # === Criterion 2: Diverse word generation ===
    print("\n=== Criterion 2: Diverse word generation ===")
    counts = Counter(words)
    unique = len(counts)
    most_common_word, most_common_count = counts.most_common(1)[0]
    dominance = most_common_count / 100
    print(f"  Unique words: {unique}/100")
    print(f"  Most common: '{most_common_word}' ({most_common_count} times, {dominance:.0%})")
    print(f"  Top 5: {counts.most_common(5)}")
    if unique >= 20 and dominance <= 0.10:
        print("  PASS (>= 20 unique, no word > 10%)")
    else:
        reasons = []
        if unique < 20:
            reasons.append(f"only {unique} unique words")
        if dominance > 0.10:
            reasons.append(f"'{most_common_word}' is {dominance:.0%}")
        print(f"  FAIL ({', '.join(reasons)})")
        all_passed = False

    # === Criterion 3: Letters after [sep] ===
    print("\n=== Criterion 3: Generates letters after [sep] ===")
    state = env.reset(target_word="crane")
    state, _ = env.step(state, "slate")
    prompt_tokens = game_state_to_prompt(state)
    prompt_ids = tokenizer.encode("".join(prompt_tokens))
    top = top_predictions(model, tokenizer, device, prompt_ids, k=10)
    letter_count = sum(1 for t in top[:5] if len(t) == 1 and "a" <= t <= "z")
    print(f"  Prompt: {' '.join(prompt_tokens)}")
    print(f"  Top 10 predictions: {top}")
    print(f"  Letters in top 5: {letter_count}/5")
    if letter_count >= 3:
        print("  PASS (>= 3 letters in top 5)")
    else:
        print("  FAIL (< 3 letters in top 5)")
        all_passed = False

    # === Criterion 4: No degenerate mid-game output ===
    print("\n=== Criterion 4: No degenerate output from game prompts ===")
    test_games = [
        ("crane", "slate"),
        ("piano", "house"),
        ("blaze", "train"),
        ("ghost", "crane"),
        ("plumb", "slate"),
        ("vivid", "house"),
        ("jumbo", "train"),
        ("quirk", "crane"),
        ("waxen", "stare"),
        ("youth", "adieu"),
    ]
    valid_midgame = 0
    for target, guess in test_games:
        state = env.reset(target_word=target)
        state, _ = env.step(state, guess)
        prompt_tokens = game_state_to_prompt(state)
        prompt_ids = tokenizer.encode("".join(prompt_tokens))
        gen_words = generate_words(model, tokenizer, device, prompt_ids, 10)
        valid_in_batch = sum(1 for w in gen_words if w in valid_words)
        valid_midgame += valid_in_batch
        print(f"  {target}/{guess}: {valid_in_batch}/10 valid, sample: {gen_words[:3]}")
    rate = valid_midgame / 100
    print(f"  Total valid: {valid_midgame}/100 ({rate:.0%})")
    if rate >= 0.50:
        print("  PASS (>= 50%)")
    else:
        print("  FAIL (< 50%)")
        all_passed = False

    # === Criterion 5: No single-word dominance across states ===
    print("\n=== Criterion 5: No single-word dominance across game states ===")
    all_words_generated: list[str] = list(words)  # reuse criterion 1 words
    for target, guess in test_games:
        state = env.reset(target_word=target)
        state, _ = env.step(state, guess)
        prompt_tokens = game_state_to_prompt(state)
        prompt_ids = tokenizer.encode("".join(prompt_tokens))
        all_words_generated.extend(generate_words(model, tokenizer, device, prompt_ids, 100))
    total = len(all_words_generated)
    counts_all = Counter(all_words_generated)
    top_word, top_count = counts_all.most_common(1)[0]
    dominance_all = top_count / total
    print(f"  Total words generated: {total}")
    print(f"  Unique words: {len(counts_all)}")
    print(f"  Most common: '{top_word}' ({top_count} times, {dominance_all:.1%})")
    if dominance_all <= 0.05:
        print("  PASS (no word > 5%)")
    else:
        print(f"  FAIL ('{top_word}' is {dominance_all:.1%})")
        all_passed = False

    # === Summary ===
    print("\n" + "=" * 50)
    if all_passed:
        print("ALL CRITERIA PASSED — model is ready for RL fine-tuning")
    else:
        print("SOME CRITERIA FAILED — model needs more or different training")
    print("=" * 50)

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-trained model readiness for RL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    passed = run_eval(args.checkpoint)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
