"""Round 3: Fill remaining gaps. Consistency + per-turn quality under near-greedy.

Also compute first-guess analysis and game comparisons under near-greedy
to complete the picture.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from decoding import sample_constrained
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_wordle import WordleEnv, WordTrie, load_answers
from mm_wordle.game import LetterFeedback
from mm_wordle.reward import _compute_expected_info_gain
from mm_wordle.serialize import game_state_to_prompt
from mm_wordle.solver import filter_candidates


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def play_game_detailed(
    model: GPT,
    env: WordleEnv,
    target: str,
    tokenizer: CharTokenizer,
    trie: WordTrie,
    answers: list[str],
    device: torch.device,
    temperature: float = 0.1,
    opener_model: GPT | None = None,
    opener_turns: int = 2,
) -> dict:
    state = env.reset(target_word=target)
    candidates = list(answers)
    turns = []

    while not state.solved and not state.failed:
        st = game_state_to_prompt(state)
        si = torch.tensor(tokenizer.encode("".join(st)), dtype=torch.long, device=device)

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
        n_before = len(candidates)
        consistent = guess in candidates

        new_state, _ = env.step(state, guess)
        feedback = new_state.guesses[-1].feedback

        if n_before > 1:
            exp_ig = _compute_expected_info_gain(guess, candidates)
            after = filter_candidates(candidates, guess, feedback)
            actual_ig = math.log2(n_before / max(len(after), 1))
        else:
            exp_ig = 0.0
            actual_ig = 0.0
            after = filter_candidates(candidates, guess, feedback)

        greens = sum(1 for f in feedback if f == LetterFeedback.GREEN)
        yellows = sum(1 for f in feedback if f == LetterFeedback.YELLOW)

        turns.append(
            {
                "turn": state.turn + 1,
                "guess": guess,
                "consistent": consistent,
                "candidates_before": n_before,
                "candidates_after": len(after),
                "expected_ig": round(exp_ig, 3),
                "actual_ig": round(actual_ig, 3),
                "greens": greens,
                "yellows": yellows,
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


def run_detailed_eval(
    model: GPT,
    eval_words: list[str],
    tokenizer: CharTokenizer,
    trie: WordTrie,
    answers: list[str],
    device: torch.device,
    temperature: float,
    label: str,
    opener_model: GPT | None = None,
) -> list[dict]:
    env = WordleEnv()
    results = []
    for target in eval_words:
        game = play_game_detailed(model, env, target, tokenizer, trie, answers, device, temperature, opener_model)
        results.append(game)

    wins = sum(1 for g in results if g["solved"])
    n = len(results)
    avg_g = sum(g["turns_count"] for g in results) / n
    print(f"  {label}: {wins}/{n} ({wins / n:.1%}), avg_guesses={avg_g:.2f}")
    return results


def analyze_consistency(games: list[dict], label: str) -> dict:
    by_turn: dict[int, dict[str, int]] = {}
    total, consistent = 0, 0
    for game in games:
        for t in game["turns"]:
            tn = t["turn"]
            if tn not in by_turn:
                by_turn[tn] = {"total": 0, "consistent": 0}
            by_turn[tn]["total"] += 1
            total += 1
            if t["consistent"]:
                by_turn[tn]["consistent"] += 1
                consistent += 1

    print(f"  {label}: {consistent}/{total} ({consistent / total:.1%}) overall")
    for tn in sorted(by_turn):
        bt = by_turn[tn]
        r = bt["consistent"] / max(bt["total"], 1)
        print(f"    Turn {tn}: {bt['consistent']}/{bt['total']} ({r:.1%})")
    return {
        "overall": round(consistent / total, 3),
        "by_turn": {k: round(v["consistent"] / max(v["total"], 1), 3) for k, v in by_turn.items()},
    }


def analyze_first_guess(games: list[dict], label: str) -> dict:
    guesses = [g["turns"][0]["guess"] for g in games]
    counter = Counter(guesses)
    unique = len(counter)
    avg_ig = sum(g["turns"][0]["expected_ig"] for g in games) / len(games)
    top5 = counter.most_common(5)
    print(f"  {label}: {unique} unique, avg_ig={avg_ig:.2f}, top={', '.join(f'{w}({c})' for w, c in top5[:3])}")
    return {"unique": unique, "avg_ig": round(avg_ig, 3), "top5": top5}


def analyze_per_turn(games: list[dict], label: str) -> dict:
    by_turn: dict[int, dict[str, list]] = {}
    for game in games:
        for t in game["turns"]:
            tn = t["turn"]
            if tn not in by_turn:
                by_turn[tn] = {"exp_ig": [], "act_ig": [], "cands_before": [], "cands_after": [], "greens": []}
            by_turn[tn]["exp_ig"].append(t["expected_ig"])
            by_turn[tn]["act_ig"].append(t["actual_ig"])
            by_turn[tn]["cands_before"].append(t["candidates_before"])
            by_turn[tn]["cands_after"].append(t["candidates_after"])
            by_turn[tn]["greens"].append(t["greens"])

    print(f"  {label}:")
    data = {}
    for tn in sorted(by_turn):
        bt = by_turn[tn]
        n = len(bt["exp_ig"])
        avg_exp = sum(bt["exp_ig"]) / n
        avg_act = sum(bt["act_ig"]) / n
        avg_cb = sum(bt["cands_before"]) / n
        avg_ca = sum(bt["cands_after"]) / n
        avg_g = sum(bt["greens"]) / n
        print(f"    Turn {tn} (n={n:>3d}): exp_ig={avg_exp:.2f}, cands={avg_cb:.0f}→{avg_ca:.0f}, greens={avg_g:.1f}")
        data[tn] = {
            "n": n,
            "exp_ig": round(avg_exp, 3),
            "act_ig": round(avg_act, 3),
            "cands_before": round(avg_cb, 1),
            "cands_after": round(avg_ca, 1),
            "greens": round(avg_g, 2),
        }
    return data


def analyze_game_comparison(baseline: list[dict], test: list[dict], label: str) -> dict:
    bl = {g["target"]: g for g in baseline}
    te = {g["target"]: g for g in test}
    both_win = both_lose = bl_win = te_win = 0
    regressions = []
    improvements = []
    for target in bl:
        b, t = bl[target], te[target]
        if b["solved"] and t["solved"]:
            both_win += 1
        elif not b["solved"] and not t["solved"]:
            both_lose += 1
        elif b["solved"] and not t["solved"]:
            bl_win += 1
            regressions.append({"target": target, "bl": b["guesses"], "te": t["guesses"]})
        else:
            te_win += 1
            improvements.append({"target": target, "bl": b["guesses"], "te": t["guesses"]})

    net = te_win - bl_win
    print(f"  {label}: both_win={both_win}, both_lose={both_lose}, regress={bl_win}, improve={te_win}, net={net:+d}")
    return {
        "both_win": both_win,
        "both_lose": both_lose,
        "regressions": bl_win,
        "improvements": te_win,
        "net": net,
        "regression_examples": regressions[:5],
        "improvement_examples": improvements[:5],
    }


def analyze_repetition(games: list[dict], label: str) -> dict:
    repeats = 0
    examples = []
    for g in games:
        guesses = g["guesses"]
        if len(set(guesses)) < len(guesses):
            repeats += 1
            if len(examples) < 3:
                examples.append({"target": g["target"], "guesses": guesses})
    n = len(games)
    print(f"  {label}: {repeats}/{n} ({repeats / n:.1%}) games with repeats")
    return {"count": repeats, "rate": round(repeats / n, 3), "examples": examples}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = CharTokenizer()
    answers = load_answers()
    trie = WordTrie.from_words(answers)
    trie.build_gpu_masks(tokenizer.vocab_size, {chr(ord("a") + i): i for i in range(26)}, device)

    random.seed(99)
    eval_words = random.sample(answers, 200)

    print("Loading models...")
    pretrained = load_model("runs/pretrain-small/20260602_204312/checkpoint-10000/model.pt", device)
    phase1 = load_model("runs/finetune-grpo/20260602_225128/checkpoint-2000/model.pt", device)
    phase2 = load_model("runs/finetune-grpo/20260603_022713/checkpoint-2000/model.pt", device)

    configs = [
        ("pretrained", pretrained, None, 0.1),
        ("phase1", phase1, None, 0.1),
        ("phase2", phase2, None, 0.1),
        ("phase2+opener", phase2, phase1, 0.1),
    ]

    all_games: dict[str, list[dict]] = {}
    print("\n=== Running detailed evals (temp=0.1) ===")
    for label, model, opener, temp in configs:
        torch.manual_seed(42)
        all_games[label] = run_detailed_eval(model, eval_words, tokenizer, trie, answers, device, temp, label, opener)

    print("\n=== Consistency ===")
    consistency = {}
    for label, games in all_games.items():
        consistency[label] = analyze_consistency(games, label)

    print("\n=== First Guess ===")
    first_guess = {}
    for label, games in all_games.items():
        first_guess[label] = analyze_first_guess(games, label)

    print("\n=== Per-Turn Quality ===")
    per_turn = {}
    for label, games in all_games.items():
        per_turn[label] = analyze_per_turn(games, label)

    print("\n=== Game Comparison vs Pretrained ===")
    comparisons = {}
    for label in ["phase1", "phase2", "phase2+opener"]:
        comparisons[label] = analyze_game_comparison(
            all_games["pretrained"],
            all_games[label],
            f"pretrained vs {label}",
        )

    print("\n=== Repetition ===")
    repetition = {}
    for label, games in all_games.items():
        repetition[label] = analyze_repetition(games, label)

    # Save everything
    data_dir = Path("runs/investigation")
    data_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "consistency": consistency,
        "first_guess": {k: {**v, "top5": v["top5"]} for k, v in first_guess.items()},
        "per_turn": per_turn,
        "comparisons": comparisons,
        "repetition": repetition,
    }
    with open(data_dir / "r3_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Also save full game data
    with open(data_dir / "r3_games.json", "w") as f:
        json.dump(all_games, f, indent=2, default=str)

    print(f"\nSaved to {data_dir}/r3_analysis.json and r3_games.json")


if __name__ == "__main__":
    main()
