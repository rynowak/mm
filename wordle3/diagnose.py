"""Hold-out failure-mode diagnostic (read-only).

Classifies why hold-out games are lost: (A) the model can't narrow the candidate
set, vs (B) it narrows but won't commit to the hold-out word (a train-answer
bias). The true answer is always consistent with its own feedback, so it never
leaves the candidate set — the only failure modes are (A) and (B).

Key signals printed per target set:
  * win rate
  * final-candidate-size distribution among losses (large -> A, small -> B)
  * direct probe: P(guess target | only the target remains) -- low on hold-out = B
  * endgame guess-class mix (train / hold-out / invalid) when <=2 candidates remain

Usage:
    uv run python -m wordle3.diagnose --checkpoint runs/finetune-v3/<ts>/checkpoint-1000/model.pt --n 500
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from mm_model import GPT, GPTConfig, load_checkpoint
from mm_wordle import ConstraintTokenizer, PatternMatrix, WordleEnv, load_full_word_set
from mm_wordle.solver import filter_candidates

from wordle3.metrics import _greedy_guess, build_letter_mask
from wordle3.splits import load_split


def _classify(word: str, train_set: set[str], holdout_set: set[str]) -> str:
    if word in train_set:
        return "train"
    if word in holdout_set:
        return "holdout"
    return "invalid"


@torch.no_grad()
def _play(model, tok, pm, target, device, mask, train_set, holdout_set) -> dict:
    env = WordleEnv()
    state = env.reset(target_word=target)
    target_idx = pm.target_index[target]
    cand = np.arange(len(pm.targets))
    records: list[dict] = []
    only_target_commit: list[bool] = []  # when cand == {target}, did it guess target?

    while not state.solved and not state.failed:
        cand_before = int(len(cand))
        guess = tok.decode_letters(_greedy_guess(model, tok.encode_game_state(state), device, mask))
        if len(guess) != 5:
            guess = "zzzzz"
        valid = guess in pm.guess_index
        if cand_before == 1:
            only_target_commit.append(guess == target)
        records.append({"cand_before": cand_before, "cls": _classify(guess, train_set, holdout_set)})

        state, _ = env.step(state, guess)
        fb = state.guesses[-1].feedback
        if valid:
            cand = pm.consistent_idx(guess, pm.pattern_id(guess, target), cand)
        else:
            kept = set(filter_candidates([pm.targets[i] for i in cand], guess, fb))
            cand = np.array([i for i in cand if pm.targets[i] in kept], dtype=np.int64)

    return {
        "solved": state.solved,
        "final_cand": int(len(cand)),
        "target_in_final": bool((cand == target_idx).any()),
        "records": records,
        "only_target_commit": only_target_commit,
    }


def _report(label: str, games: list[dict]) -> None:
    n = len(games)
    wins = sum(g["solved"] for g in games)
    losses = [g for g in games if not g["solved"]]
    print(f"\n=== {label} (n={n}) ===")
    print(f"win rate: {wins / n:.1%}   losses: {len(losses)}")
    assert all(g["target_in_final"] for g in games), "target left candidate set (impossible) — bug"

    if losses:
        lf = [g["final_cand"] for g in losses]
        narrowed = sum(c <= 2 for c in lf)
        big = sum(c > 5 for c in lf)
        print(
            f"loss final-candidate size: median {statistics.median(lf):.0f} "
            f"mean {statistics.mean(lf):.1f} min {min(lf)} max {max(lf)}"
        )
        print(f"  (B) narrowed to <=2 but lost: {narrowed}/{len(losses)} = {narrowed / len(losses):.0%}")
        print(f"  (A) still >5 candidates:      {big}/{len(losses)} = {big / len(losses):.0%}")

    commits = [c for g in games for c in g["only_target_commit"]]
    if commits:
        p = sum(commits) / len(commits)
        print(f"direct probe P(guess target | only target remains): {p:.0%}  (n={len(commits)})")
    else:
        print("direct probe: never reached a 1-candidate state")

    endgame = Counter(r["cls"] for g in games for r in g["records"] if r["cand_before"] <= 2)
    allcls = Counter(r["cls"] for g in games for r in g["records"])
    print(f"endgame (<=2 cand) guess mix: {dict(endgame)}")
    print(f"all-turn guess mix:           {dict(allcls)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold-out failure-mode diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n", type=int, default=500, help="games per set (hold-out and train)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default="runs/cache")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = ConstraintTokenizer()
    words = load_full_word_set()
    split = load_split()
    pm = PatternMatrix.load_or_build(words, args.cache_dir)

    ckpt = load_checkpoint(Path(args.checkpoint), device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    mask = build_letter_mask(tok, device)
    train_set, holdout_set = set(split.train_answers), set(split.holdout)

    rng = random.Random(args.seed)
    holdout_targets = rng.sample(split.holdout, min(args.n, len(split.holdout)))
    train_targets = rng.sample(split.train_answers, min(args.n, len(split.train_answers)))

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}  | hold-out games: {len(holdout_targets)}  train games: {len(train_targets)}")

    holdout_games = [_play(model, tok, pm, t, device, mask, train_set, holdout_set) for t in holdout_targets]
    _report("HOLD-OUT", holdout_games)
    train_games = [_play(model, tok, pm, t, device, mask, train_set, holdout_set) for t in train_targets]
    _report("TRAIN", train_games)


if __name__ == "__main__":
    main()
