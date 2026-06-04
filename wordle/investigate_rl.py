"""Investigation: Why is RL decreasing win rate?

Runs experiments comparing pretrained, Phase 1, and Phase 2 models
on the same eval set to understand exactly what RL changes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from decoding import sample_constrained
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_tokenizers import CharTokenizer
from mm_wordle import WordleEnv, WordTrie, load_answers
from mm_wordle.game import LetterFeedback
from mm_wordle.reward import _compute_expected_info_gain
from mm_wordle.solver import filter_candidates


class _LayerDiff(TypedDict):
    abs_diff: float
    count: int
    rel_diffs: list[float]


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    config_dict = checkpoint["config"]
    model_config = GPTConfig(**config_dict)
    model = GPT(model_config)
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
) -> dict:
    """Play a game and return detailed per-turn info."""
    from mm_wordle.serialize import game_state_to_prompt

    state = env.reset(target_word=target)
    candidates = list(answers)
    turns = []

    while not state.solved and not state.failed:
        state_tokens = game_state_to_prompt(state)
        state_ids = torch.tensor(tokenizer.encode("".join(state_tokens)), dtype=torch.long, device=device)

        samples = sample_constrained(model, state_ids, trie, tokenizer, device, n_samples=1)
        guess = samples[0][0]

        n_candidates_before = len(candidates)
        is_consistent = guess in candidates

        new_state, _ = env.step(state, guess)
        feedback = new_state.guesses[-1].feedback

        # Compute info gain
        if n_candidates_before > 1:
            exp_ig = _compute_expected_info_gain(guess, candidates)
            candidates_after = filter_candidates(candidates, guess, feedback)
            n_after = max(len(candidates_after), 1)
            import math

            actual_ig = math.log2(n_candidates_before / n_after)
        else:
            exp_ig = 0.0
            actual_ig = 0.0
            candidates_after = filter_candidates(candidates, guess, feedback)

        greens = sum(1 for f in feedback if f == LetterFeedback.GREEN)
        yellows = sum(1 for f in feedback if f == LetterFeedback.YELLOW)
        grays = sum(1 for f in feedback if f == LetterFeedback.GRAY)

        turns.append(
            {
                "turn": state.turn + 1,
                "guess": guess,
                "greens": greens,
                "yellows": yellows,
                "grays": grays,
                "candidates_before": n_candidates_before,
                "candidates_after": len(candidates_after),
                "expected_ig": round(exp_ig, 3),
                "actual_ig": round(actual_ig, 3),
                "consistent": is_consistent,
            }
        )

        candidates = candidates_after
        state = new_state

    return {
        "target": target,
        "solved": state.solved,
        "num_turns": state.turn,
        "guesses": [t["guess"] for t in turns],
        "turns": turns,
    }


def run_eval(
    model: GPT,
    eval_words: list[str],
    tokenizer: CharTokenizer,
    trie: WordTrie,
    answers: list[str],
    device: torch.device,
    label: str,
) -> list[dict]:
    """Run full eval and return detailed game results."""
    env = WordleEnv()
    results = []
    for target in eval_words:
        game = play_game_detailed(model, env, target, tokenizer, trie, answers, device)
        results.append(game)

    wins = sum(1 for g in results if g["solved"])
    total_turns = sum(g["num_turns"] for g in results)
    n = len(results)
    print(f"  {label}: {wins}/{n} wins ({wins / n:.1%}), avg_guesses={total_turns / n:.2f}")
    return results


def experiment_1_baseline(models: dict, eval_words: list[str], tokenizer, trie, answers, device) -> dict:
    """Experiment 1: Baseline win rates on same eval set."""
    print("\n=== Experiment 1: Baseline Win Rates (200 games) ===")
    results = {}
    for name, model in models.items():
        results[name] = run_eval(model, eval_words, tokenizer, trie, answers, device, name)
    return results


def experiment_2_first_guess(all_results: dict) -> dict:
    """Experiment 2: First guess diversity and quality."""
    print("\n=== Experiment 2: First Guess Analysis ===")
    data = {}
    for name, games in all_results.items():
        first_guesses = [g["turns"][0]["guess"] for g in games]
        counter = Counter(first_guesses)
        unique = len(counter)
        top5 = counter.most_common(5)

        # Average expected info gain of first guesses
        avg_ig = sum(g["turns"][0]["expected_ig"] for g in games) / len(games)

        print(f"  {name}:")
        print(f"    Unique first guesses: {unique}")
        print(f"    Avg first-guess info gain: {avg_ig:.2f} bits")
        print(f"    Top 5: {', '.join(f'{w}({c})' for w, c in top5)}")

        data[name] = {
            "unique": unique,
            "avg_ig": round(avg_ig, 3),
            "top5": top5,
            "distribution": dict(counter),
        }
    return data


def experiment_3_game_comparison(all_results: dict) -> dict:
    """Experiment 3: Head-to-head game comparison."""
    print("\n=== Experiment 3: Game-by-Game Comparison ===")

    pretrained = {g["target"]: g for g in all_results["pretrained"]}
    data = {}

    for name in all_results:
        if name == "pretrained":
            continue

        rl_games = {g["target"]: g for g in all_results[name]}

        both_win = 0
        both_lose = 0
        pt_win_rl_lose = 0
        pt_lose_rl_win = 0
        regressions = []

        for target in pretrained:
            pt = pretrained[target]
            rl = rl_games[target]

            if pt["solved"] and rl["solved"]:
                both_win += 1
            elif not pt["solved"] and not rl["solved"]:
                both_lose += 1
            elif pt["solved"] and not rl["solved"]:
                pt_win_rl_lose += 1
                regressions.append(
                    {
                        "target": target,
                        "pt_guesses": pt["guesses"],
                        "rl_guesses": rl["guesses"],
                        "pt_turns": pt["num_turns"],
                        "rl_turns": rl["num_turns"],
                    }
                )
            else:
                pt_lose_rl_win += 1

        print(f"  pretrained vs {name}:")
        print(f"    Both win:  {both_win}")
        print(f"    Both lose: {both_lose}")
        print(f"    PT wins, RL loses: {pt_win_rl_lose} (REGRESSIONS)")
        print(f"    PT loses, RL wins: {pt_lose_rl_win} (IMPROVEMENTS)")
        print(f"    Net: {pt_lose_rl_win - pt_win_rl_lose:+d}")

        if regressions:
            print("    Sample regressions:")
            for r in regressions[:5]:
                print(f"      {r['target']}: PT={r['pt_guesses']} vs RL={r['rl_guesses']}")

        data[name] = {
            "both_win": both_win,
            "both_lose": both_lose,
            "pt_win_rl_lose": pt_win_rl_lose,
            "pt_lose_rl_win": pt_lose_rl_win,
            "regressions": regressions[:10],
        }
    return data


def experiment_4_consistency(all_results: dict) -> dict:
    """Experiment 4: How often does each model guess consistently with feedback?"""
    print("\n=== Experiment 4: Feedback Consistency Rate ===")
    data = {}

    for name, games in all_results.items():
        total_turns = 0
        consistent_turns = 0
        by_turn = {}

        for game in games:
            for t in game["turns"]:
                turn_num = t["turn"]
                total_turns += 1
                if t["consistent"]:
                    consistent_turns += 1

                if turn_num not in by_turn:
                    by_turn[turn_num] = {"total": 0, "consistent": 0}
                by_turn[turn_num]["total"] += 1
                if t["consistent"]:
                    by_turn[turn_num]["consistent"] += 1

        rate = consistent_turns / max(total_turns, 1)
        print(f"  {name}: {consistent_turns}/{total_turns} consistent ({rate:.1%})")
        print("    By turn:")
        for turn_num in sorted(by_turn.keys()):
            tb = by_turn[turn_num]
            tr = tb["consistent"] / max(tb["total"], 1)
            print(f"      Turn {turn_num}: {tb['consistent']}/{tb['total']} ({tr:.1%})")

        data[name] = {
            "overall_rate": round(rate, 3),
            "total_turns": total_turns,
            "consistent_turns": consistent_turns,
            "by_turn": {k: round(v["consistent"] / max(v["total"], 1), 3) for k, v in by_turn.items()},
        }
    return data


def experiment_5_turn_quality(all_results: dict) -> dict:
    """Experiment 5: Per-turn info gain and candidate reduction."""
    print("\n=== Experiment 5: Per-Turn Quality ===")
    data = {}

    for name, games in all_results.items():
        by_turn: dict[int, dict[str, list]] = {}

        for game in games:
            for t in game["turns"]:
                turn_num = t["turn"]
                if turn_num not in by_turn:
                    by_turn[turn_num] = {
                        "expected_ig": [],
                        "actual_ig": [],
                        "candidates_before": [],
                        "candidates_after": [],
                        "greens": [],
                    }
                by_turn[turn_num]["expected_ig"].append(t["expected_ig"])
                by_turn[turn_num]["actual_ig"].append(t["actual_ig"])
                by_turn[turn_num]["candidates_before"].append(t["candidates_before"])
                by_turn[turn_num]["candidates_after"].append(t["candidates_after"])
                by_turn[turn_num]["greens"].append(t["greens"])

        print(f"  {name}:")
        turn_data = {}
        for turn_num in sorted(by_turn.keys()):
            tb = by_turn[turn_num]
            n = len(tb["expected_ig"])
            avg_exp = sum(tb["expected_ig"]) / n
            avg_act = sum(tb["actual_ig"]) / n
            avg_cand_before = sum(tb["candidates_before"]) / n
            avg_cand_after = sum(tb["candidates_after"]) / n
            avg_greens = sum(tb["greens"]) / n
            print(
                f"    Turn {turn_num} (n={n:>3d}): "
                f"exp_ig={avg_exp:.2f}, act_ig={avg_act:.2f}, "
                f"cands={avg_cand_before:.0f}→{avg_cand_after:.0f}, "
                f"greens={avg_greens:.1f}"
            )
            turn_data[turn_num] = {
                "n": n,
                "avg_expected_ig": round(avg_exp, 3),
                "avg_actual_ig": round(avg_act, 3),
                "avg_candidates_before": round(avg_cand_before, 1),
                "avg_candidates_after": round(avg_cand_after, 1),
                "avg_greens": round(avg_greens, 2),
            }
        data[name] = turn_data
    return data


def experiment_6_repetition(all_results: dict) -> dict:
    """Experiment 6: Does the model repeat guesses or get stuck in loops?"""
    print("\n=== Experiment 6: Guess Repetition Analysis ===")
    data = {}

    for name, games in all_results.items():
        games_with_repeats = 0
        total_repeated_guesses = 0
        repeat_examples = []

        for game in games:
            guesses = game["guesses"]
            unique = set(guesses)
            if len(unique) < len(guesses):
                games_with_repeats += 1
                repeats = len(guesses) - len(unique)
                total_repeated_guesses += repeats
                if len(repeat_examples) < 5:
                    repeat_examples.append(
                        {
                            "target": game["target"],
                            "guesses": guesses,
                        }
                    )

        n = len(games)
        print(f"  {name}:")
        print(f"    Games with repeated guesses: {games_with_repeats}/{n} ({games_with_repeats / n:.1%})")
        print(f"    Total repeated guesses: {total_repeated_guesses}")
        if repeat_examples:
            print("    Examples:")
            for ex in repeat_examples[:3]:
                print(f"      {ex['target']}: {' → '.join(ex['guesses'])}")

        data[name] = {
            "games_with_repeats": games_with_repeats,
            "repeat_rate": round(games_with_repeats / n, 3),
            "total_repeated": total_repeated_guesses,
            "examples": repeat_examples,
        }
    return data


def experiment_7_weight_divergence(models: dict, pretrained_path: str, device: torch.device) -> dict:
    """Experiment 7: How much did the weights actually change?"""
    print("\n=== Experiment 7: Weight Divergence from Pretrained ===")

    pretrained_state = models["pretrained"].state_dict()
    data = {}

    for name, model in models.items():
        if name == "pretrained":
            continue

        state = model.state_dict()
        layer_diffs: dict[str, _LayerDiff] = {}
        total_diff = 0.0
        total_params = 0

        for key in pretrained_state:
            pt_w = pretrained_state[key].float()
            rl_w = state[key].float()
            diff = (pt_w - rl_w).abs().mean().item()
            rel_diff = diff / (pt_w.abs().mean().item() + 1e-8)
            n = pt_w.numel()
            total_diff += diff * n
            total_params += n

            # Summarize by layer group
            parts = key.split(".")
            group = ".".join(parts[:2]) if len(parts) >= 2 else key
            if group not in layer_diffs:
                layer_diffs[group] = {"abs_diff": 0.0, "count": 0, "rel_diffs": []}
            layer_diffs[group]["abs_diff"] += diff * n
            layer_diffs[group]["count"] += n
            layer_diffs[group]["rel_diffs"].append(rel_diff)

        avg_diff = total_diff / total_params
        print(f"  {name}: avg absolute weight diff = {avg_diff:.6f}")
        print("    Per layer group:")

        layer_summary = {}
        for group in sorted(layer_diffs.keys()):
            ld = layer_diffs[group]
            avg = ld["abs_diff"] / ld["count"]
            avg_rel = sum(ld["rel_diffs"]) / len(ld["rel_diffs"])
            print(f"      {group}: abs={avg:.6f}, rel={avg_rel:.4f}")
            layer_summary[group] = {"abs_diff": round(avg, 6), "rel_diff": round(avg_rel, 4)}

        data[name] = {
            "avg_abs_diff": round(avg_diff, 6),
            "layers": layer_summary,
        }
    return data


def write_report(experiments: dict, output_path: Path) -> None:
    """Write the investigation report as markdown."""
    lines = []
    lines.append("# Investigation: Why RL Decreases Win Rate")
    lines.append("")
    lines.append("## Background")
    lines.append("")
    lines.append("Pre-trained model achieves ~30% win rate on Wordle. Every RL fine-tuning run")
    lines.append("has decreased the win rate: Phase 1 → ~24%, Phase 2 → ~16%. This investigation")
    lines.append("runs 7 experiments on the same 200-word eval set to understand what RL changes.")
    lines.append("")
    lines.append("Models compared:")
    lines.append("- **pretrained**: Final pre-training checkpoint (10K steps)")
    lines.append("- **phase1**: Phase 1 GRPO (turns 1-2, info gain reward, 2000 steps)")
    lines.append("- **phase2**: Phase 2 GRPO (turns 3-6, composite reward, 2000 steps)")
    lines.append("")

    # Experiment 1
    e1 = experiments["experiment_1"]
    lines.append("## Experiment 1: Baseline Win Rates")
    lines.append("")
    lines.append("All three models evaluated on the same 200 target words with constrained decoding.")
    lines.append("")
    lines.append("| Model | Wins | Win Rate | Avg Guesses |")
    lines.append("|-------|------|----------|-------------|")
    for name, games in e1.items():
        wins = sum(1 for g in games if g["solved"])
        n = len(games)
        avg_g = sum(g["num_turns"] for g in games) / n
        lines.append(f"| {name} | {wins}/{n} | {wins / n:.1%} | {avg_g:.2f} |")
    lines.append("")

    # Experiment 2
    e2 = experiments["experiment_2"]
    lines.append("## Experiment 2: First Guess Analysis")
    lines.append("")
    lines.append("| Model | Unique First Guesses | Avg Info Gain | Top 3 |")
    lines.append("|-------|---------------------|---------------|-------|")
    for name, d in e2.items():
        top3 = ", ".join(f"{w}({c})" for w, c in d["top5"][:3])
        lines.append(f"| {name} | {d['unique']} | {d['avg_ig']:.2f} bits | {top3} |")
    lines.append("")
    lines.append("**Key question:** Did RL collapse the first-guess distribution to a few words?")
    lines.append("")

    # Experiment 3
    e3 = experiments["experiment_3"]
    lines.append("## Experiment 3: Game-by-Game Comparison")
    lines.append("")
    for name, d in e3.items():
        lines.append(f"### pretrained vs {name}")
        lines.append("")
        lines.append("| Outcome | Count |")
        lines.append("|---------|-------|")
        lines.append(f"| Both win | {d['both_win']} |")
        lines.append(f"| Both lose | {d['both_lose']} |")
        lines.append(f"| Pretrained wins, RL loses | {d['pt_win_rl_lose']} |")
        lines.append(f"| Pretrained loses, RL wins | {d['pt_lose_rl_win']} |")
        lines.append(f"| **Net** | **{d['pt_lose_rl_win'] - d['pt_win_rl_lose']:+d}** |")
        lines.append("")

        if d["regressions"]:
            lines.append("**Regression examples** (pretrained won, RL lost):")
            lines.append("")
            for r in d["regressions"][:5]:
                pt = "→".join(r["pt_guesses"])
                rl = "→".join(r["rl_guesses"])
                lines.append(f"- **{r['target']}**: PT=`{pt}` ({r['pt_turns']}t) vs RL=`{rl}` ({r['rl_turns']}t)")
            lines.append("")

    # Experiment 4
    e4 = experiments["experiment_4"]
    lines.append("## Experiment 4: Feedback Consistency Rate")
    lines.append("")
    lines.append("Does the model guess words consistent with the feedback it has received?")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append("| Model | Consistent Rate |")
    lines.append("|-------|----------------|")
    for name, d in e4.items():
        lines.append(f"| {name} | {d['overall_rate']:.1%} |")
    lines.append("")

    lines.append("### By Turn")
    lines.append("")
    header = "| Turn |"
    sep = "|------|"
    for name in e4:
        header += f" {name} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)
    all_turns = sorted(set(t for d in e4.values() for t in d["by_turn"]))
    for turn in all_turns:
        row = f"| {turn} |"
        for name in e4:
            rate = e4[name]["by_turn"].get(turn, 0)
            row += f" {rate:.1%} |"
        lines.append(row)
    lines.append("")

    # Experiment 5
    e5 = experiments["experiment_5"]
    lines.append("## Experiment 5: Per-Turn Quality")
    lines.append("")
    for name, turn_data in e5.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Turn | N | Exp IG | Act IG | Cands Before | Cands After | Avg Greens |")
        lines.append("|------|---|--------|--------|-------------|-------------|------------|")
        for turn, td in sorted(turn_data.items()):
            lines.append(
                f"| {turn} | {td['n']} | {td['avg_expected_ig']:.2f} | {td['avg_actual_ig']:.2f} | "
                f"{td['avg_candidates_before']:.0f} | {td['avg_candidates_after']:.0f} | {td['avg_greens']:.1f} |"
            )
        lines.append("")

    # Experiment 6
    e6 = experiments["experiment_6"]
    lines.append("## Experiment 6: Guess Repetition")
    lines.append("")
    lines.append("| Model | Games with Repeats | Repeat Rate | Total Repeated |")
    lines.append("|-------|--------------------|-------------|----------------|")
    for name, d in e6.items():
        lines.append(f"| {name} | {d['games_with_repeats']} | {d['repeat_rate']:.1%} | {d['total_repeated']} |")
    lines.append("")
    for name, d in e6.items():
        if d["examples"]:
            lines.append(f"**{name} repeat examples:**")
            for ex in d["examples"][:3]:
                lines.append(f"- {ex['target']}: `{'→'.join(ex['guesses'])}`")
            lines.append("")

    # Experiment 7
    e7 = experiments["experiment_7"]
    lines.append("## Experiment 7: Weight Divergence")
    lines.append("")
    lines.append("How much did RL change the model weights?")
    lines.append("")
    for name, d in e7.items():
        lines.append(f"### {name}")
        lines.append(f"Average absolute weight difference: {d['avg_abs_diff']:.6f}")
        lines.append("")
        lines.append("| Layer Group | Abs Diff | Relative Diff |")
        lines.append("|-------------|----------|---------------|")
        for group, ld in d["layers"].items():
            lines.append(f"| {group} | {ld['abs_diff']:.6f} | {ld['rel_diff']:.4f} |")
        lines.append("")

    # Conclusions
    lines.append("## Raw Data")
    lines.append("")
    lines.append("Full game-level data saved to `runs/investigation/games.json`.")
    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nReport written to {output_path}")


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    answers = load_answers()

    # Build trie
    trie = WordTrie.from_words(answers)
    char_to_id = {chr(ord("a") + i): i for i in range(26)}
    trie.build_gpu_masks(tokenizer.vocab_size, char_to_id, device)

    # Fixed eval set — 200 words, seeded
    import random

    random.seed(99)
    eval_words = random.sample(answers, 200)

    # Load models
    print("Loading models...")
    pretrained_path = "runs/pretrain-small/20260602_204312/checkpoint-10000/model.pt"
    phase1_path = "runs/finetune-grpo/20260602_225128/checkpoint-2000/model.pt"
    phase2_path = "runs/finetune-grpo/20260603_022713/checkpoint-2000/model.pt"

    models = {
        "pretrained": load_model(pretrained_path, device),
        "phase1": load_model(phase1_path, device),
        "phase2": load_model(phase2_path, device),
    }
    print("  All models loaded.")

    # Run experiments
    torch.manual_seed(42)
    e1 = experiment_1_baseline(models, eval_words, tokenizer, trie, answers, device)

    e2 = experiment_2_first_guess(e1)
    e3 = experiment_3_game_comparison(e1)
    e4 = experiment_4_consistency(e1)
    e5 = experiment_5_turn_quality(e1)
    e6 = experiment_6_repetition(e1)
    e7 = experiment_7_weight_divergence(models, pretrained_path, device)

    experiments = {
        "experiment_1": e1,
        "experiment_2": e2,
        "experiment_3": e3,
        "experiment_4": e4,
        "experiment_5": e5,
        "experiment_6": e6,
        "experiment_7": e7,
    }

    # Save raw data
    data_dir = Path("runs/investigation")
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "games.json", "w") as f:
        json.dump(e1, f, indent=2, default=str)

    # Write report
    write_report(experiments, Path("docs/rl-investigation.md"))


if __name__ == "__main__":
    main()
