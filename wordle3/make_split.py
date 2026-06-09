"""Generate the canonical V3 train/hold-out answer split.

This writes ``wordle3/data/split.json`` — the single source of truth for the
hold-out, loaded by every phase (pre-train ignores it; SFT/RL/eval honor it). The
split is deterministic in ``(seed, holdout_frac)``; re-running with the same args
reproduces it exactly.

Usage:
    uv run python wordle3/make_split.py [--seed 1234] [--holdout-frac 0.10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_wordle import load_full_word_set, split_answers

DEFAULT_OUT = Path(__file__).parent / "data" / "split.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the V3 train/hold-out split")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--holdout-frac", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = load_full_word_set()
    train, holdout = split_answers(holdout_frac=args.holdout_frac, seed=args.seed)

    # Invariants (the hold-out hard gate) — fail loudly rather than write a bad split.
    train_set, holdout_set = set(train), set(holdout)
    assert train_set.isdisjoint(holdout_set), "train and hold-out overlap"
    assert train_set | holdout_set == set(universe), "split does not cover the universe"

    payload = {
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "n_total": len(universe),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "train_answers": train,
        "holdout": holdout,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=0))
    print(
        f"Wrote {args.out}\n"
        f"  universe={len(universe)}  train={len(train)}  holdout={len(holdout)} "
        f"(frac={args.holdout_frac}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
