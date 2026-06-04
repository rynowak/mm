"""Round 2 investigation: fill evidence gaps with controlled experiments.

Experiments:
A. Eval all 3 models at temp=0.1 (near-greedy) on same 200 words
B. Eval all 3 models at temp=1.0 (stochastic) on same 200 words (repeat of R1 for confirmation)
C. Eval Phase 2 model with opener playing turns 1-2 (matches training setup)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from decoding import sample_constrained
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_wordle import WordleEnv, WordTrie, load_answers
from mm_wordle.serialize import game_state_to_prompt


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    config_dict = checkpoint["config"]
    model_config = GPTConfig(**config_dict)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def play_game(
    model: GPT,
    env: WordleEnv,
    target: str,
    tokenizer: CharTokenizer,
    trie: WordTrie,
    device: torch.device,
    temperature: float = 1.0,
    opener_model: GPT | None = None,
    opener_turns: int = 2,
) -> dict:
    """Play a game and return result."""
    state = env.reset(target_word=target)
    guesses: list[str] = []

    while not state.solved and not state.failed:
        st = game_state_to_prompt(state)
        si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)

        # Use opener model for first N turns if provided
        if opener_model is not None and state.turn < opener_turns:
            samples = sample_constrained(
                opener_model,
                si,
                trie,
                tokenizer,
                device,
                n_samples=1,
                temperature=temperature,
            )
        else:
            samples = sample_constrained(
                model,
                si,
                trie,
                tokenizer,
                device,
                n_samples=1,
                temperature=temperature,
            )

        guess = samples[0][0]
        guesses.append(guess)
        state, _ = env.step(state, guess)

    return {
        "target": target,
        "solved": state.solved,
        "turns": state.turn,
        "guesses": guesses,
    }


def run_eval(
    model: GPT,
    eval_words: list[str],
    tokenizer: CharTokenizer,
    trie: WordTrie,
    device: torch.device,
    temperature: float,
    label: str,
    opener_model: GPT | None = None,
) -> list[dict]:
    env = WordleEnv()
    results = []
    for target in eval_words:
        game = play_game(model, env, target, tokenizer, trie, device, temperature, opener_model)
        results.append(game)

    wins = sum(1 for g in results if g["solved"])
    n = len(results)
    avg_g = sum(g["turns"] for g in results) / n
    print(f"  {label}: {wins}/{n} ({wins / n:.1%}), avg_guesses={avg_g:.2f}")
    return results


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    answers = load_answers()
    trie = WordTrie.from_words(answers)
    char_to_id = {chr(ord("a") + i): i for i in range(26)}
    trie.build_gpu_masks(tokenizer.vocab_size, char_to_id, device)

    # Same 200 words as R1
    random.seed(99)
    eval_words = random.sample(answers, 200)

    # Load models
    print("Loading models...")
    pretrained = load_model("runs/pretrain-small/20260602_204312/checkpoint-10000/model.pt", device)
    phase1 = load_model("runs/finetune-grpo/20260602_225128/checkpoint-2000/model.pt", device)
    phase2 = load_model("runs/finetune-grpo/20260603_022713/checkpoint-2000/model.pt", device)
    print("  Loaded.\n")

    all_results: dict[str, list[dict]] = {}

    # Experiment A: Near-greedy (temp=0.1) - matches evaluate.py methodology
    print("=== Experiment A: Near-Greedy Eval (temp=0.1, 200 words) ===")
    torch.manual_seed(42)
    all_results["A_pretrained_greedy"] = run_eval(
        pretrained,
        eval_words,
        tokenizer,
        trie,
        device,
        0.1,
        "pretrained temp=0.1",
    )
    torch.manual_seed(42)
    all_results["A_phase1_greedy"] = run_eval(
        phase1,
        eval_words,
        tokenizer,
        trie,
        device,
        0.1,
        "phase1 temp=0.1",
    )
    torch.manual_seed(42)
    all_results["A_phase2_greedy"] = run_eval(
        phase2,
        eval_words,
        tokenizer,
        trie,
        device,
        0.1,
        "phase2 temp=0.1",
    )

    # Experiment B: Stochastic (temp=1.0) - matches training eval methodology
    # Reset seed for each model so they get the same random draws
    print("\n=== Experiment B: Stochastic Eval (temp=1.0, 200 words) ===")
    torch.manual_seed(42)
    all_results["B_pretrained_stoch"] = run_eval(
        pretrained,
        eval_words,
        tokenizer,
        trie,
        device,
        1.0,
        "pretrained temp=1.0",
    )
    torch.manual_seed(42)
    all_results["B_phase1_stoch"] = run_eval(
        phase1,
        eval_words,
        tokenizer,
        trie,
        device,
        1.0,
        "phase1 temp=1.0",
    )
    torch.manual_seed(42)
    all_results["B_phase2_stoch"] = run_eval(
        phase2,
        eval_words,
        tokenizer,
        trie,
        device,
        1.0,
        "phase2 temp=1.0",
    )

    # Experiment C: Phase 2 with opener model (matches training setup)
    print("\n=== Experiment C: Phase 2 with Opener (temp=0.1 and 1.0) ===")
    torch.manual_seed(42)
    all_results["C_phase2_opener_greedy"] = run_eval(
        phase2, eval_words, tokenizer, trie, device, 0.1, "phase2+opener temp=0.1", opener_model=phase1
    )
    torch.manual_seed(42)
    all_results["C_phase2_opener_stoch"] = run_eval(
        phase2, eval_words, tokenizer, trie, device, 1.0, "phase2+opener temp=1.0", opener_model=phase1
    )

    # Also test: Phase 1 opener + pretrained solver (to isolate opener effect)
    print("\n=== Experiment D: Opener Effect Isolation ===")
    torch.manual_seed(42)
    all_results["D_pretrained_with_opener_greedy"] = run_eval(
        pretrained, eval_words, tokenizer, trie, device, 0.1, "pretrained+opener temp=0.1", opener_model=phase1
    )
    torch.manual_seed(42)
    all_results["D_pretrained_with_opener_stoch"] = run_eval(
        pretrained, eval_words, tokenizer, trie, device, 1.0, "pretrained+opener temp=1.0", opener_model=phase1
    )

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Configuration':<35} {'Wins':>5} {'Rate':>6} {'Avg Guesses':>12}")
    print("-" * 60)
    for key, results in all_results.items():
        wins = sum(1 for g in results if g["solved"])
        n = len(results)
        avg_g = sum(g["turns"] for g in results) / n
        label = key.replace("_", " ")
        print(f"{label:<35} {wins:>5} {wins / n:>6.1%} {avg_g:>12.2f}")

    # Save raw data
    data_dir = Path("runs/investigation")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "r2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nRaw data saved to runs/investigation/r2_results.json")


if __name__ == "__main__":
    main()
