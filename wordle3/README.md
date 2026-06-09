# wordle3 — V3 Wordle (three-phase curriculum on the 14,855-word set)

Full design: [`docs/v3-design.md`](../docs/v3-design.md).

V3 keeps the V2 dense constraint-state representation but scales to the full
**14,855-word** valid set (a much harder game than V2's 2,315 curated answers) and
splits training into three phases with a held-out answer set for measuring
generalization:

1. **Pre-train** — word-only (empty constraint prompt), over *all* 14,855 words.
2. **SFT** — behavior cloning on golden games whose answers are train-only.
3. **RL** — the existing two-phase GRPO curriculum, train-only answers.

The model sees every word in pre-training, but **hold-out words are never an
answer** in SFT/RL — solving them at eval is true generalization.

## Hold-out split (single source of truth)

`wordle3/data/split.json` is the canonical train/hold-out split, loaded by every
phase via `wordle3.splits.load_split()`. It is deterministic in `(seed,
holdout_frac)`; regenerate with:

```bash
uv run python wordle3/make_split.py            # defaults: seed=1234, holdout_frac=0.10
```

Current split: 14,855 total → **13,370 train / 1,485 hold-out** (10%).

## Build status

- [x] **Phase 0** — 14,855-word set in `mm_wordle` (`load_full_word_set`,
      `split_answers`); canonical `split.json`; hold-out hard-gate tests.
- [x] **Phase 1** — `mm_wordle.pattern.PatternMatrix` (vectorized feedback,
      info-gain, filtering, best-word) with equivalence tests vs the game/reward.
      Also fixed a duplicate-letter bug in `solver.filter_candidates`.
- [x] **Phase 2** — pre-train (word-only).
      - [x] Tokenizer promoted to `mm_wordle.ConstraintTokenizer` (ADR-5).
      - [x] Word-only data pipeline (`wordle3/data.py`).
      - [x] Per-step metric engine (`wordle3/metrics.py`): valid-word rate + opener
            info gain (cheap, every step) and win-rate game mini-eval (spaced).
      - [x] Training loop (`wordle3/pretrain.py`) with bf16 autocast + `torch.compile`
            (CUDA-gated), pattern-matrix disk cache, UI live/snapshot writes.
      - [x] Configs: `pretrain-large.yaml` (~10M), `pretrain-xxl.yaml` (~38M).

Run pre-training:

```bash
uv run python -m wordle3.pretrain --config wordle3/configs/pretrain-large.yaml
# quick smoke: add --max-steps 300
# watch it:  uv run python wordle/dashboard/app.py --run-dir runs/pretrain-v3/<timestamp>
```
- [x] **Phase 3** — SFT on golden games.
      - [x] Pattern-matrix `GoldenSolver` (`mm_wordle.golden`, ADR-8) — strong play
            over the full universe (replaces the degenerate `play_game_good`).
      - [x] Golden game generation (train answers only) + `ConstraintDataset` +
            late-turn oversampling + word-only replay (N1).
      - [x] `wordle3/sft.py`, `configs/sft-large.yaml`. Validated: win 0%→34% in
            250 steps, opener-IG 4.6→5.7 bits.
- [x] **Phase 4** — two-phase GRPO RL.
      - [x] Batched KV-cache group rollout (`wordle3/rollout.py`, §5.7-B) +
            gradient-capable log-probs.
      - [x] Pattern-matrix reward (`wordle3/reward.py`); hold-out-aware targets;
            two-phase curriculum; reuses `MetricReporter`.
      - [x] `wordle3/finetune.py`, `configs/finetune-phase{1,2}.yaml`.

Run SFT then RL:

```bash
uv run python -m wordle3.sft --config wordle3/configs/sft-large.yaml \
    --checkpoint runs/pretrain-v3/<ts>/checkpoint-15000/model.pt
uv run python -m wordle3.finetune --config wordle3/configs/finetune-phase1.yaml \
    --checkpoint runs/sft-v3/<ts>/checkpoint-8000/model.pt
uv run python -m wordle3.finetune --config wordle3/configs/finetune-phase2.yaml \
    --checkpoint runs/sft-v3/<ts>/checkpoint-8000/model.pt --opener runs/finetune-v3/<ts>/checkpoint-2000/model.pt
```

Shared training machinery: `wordle3/trainutil.py` (model/autocast/IO) and
`wordle3/steplog.py` (`MetricReporter` — the per-step trio + eval snapshots),
reused by all three phase scripts.
