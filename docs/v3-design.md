# V3 Design: Three-Phase Curriculum on the 14,855-Word Set

**Status:** Reviewed (Conditional Approve — critical items resolved)
**Author:** Wordle RL exercise (rynowak/mm)

## Version History

| Version | Date | Summary |
|---------|------|---------|
| 0.1 | 2026-06-08 | Initial design |
| 0.2 | 2026-06-08 | Incorporated design-review feedback |
| 0.3 | 2026-06-08 | Added explicit RL eval block + per-phase evaluation matrix (§5.4, §5.8); hold-out = 10% confirmed |
| 0.4 | 2026-06-08 | Per-step trio (valid-word rate · info gain · win rate) for ALL phases via 16-game mini-eval, with cost analysis (§5.9); dashboard step-charts in-scope (§5.5) |
| 0.5 | 2026-06-09 | Eval fixes: avg_guesses over wins only; eval-target seed decoupled from training (hold-out is the generalization metric) |
| 0.6 | 2026-06-10 | Post-implementation findings (§11) + V3.1 constraint-conditioned retrieval pre-training objective (§12) |

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture (V2)](#2-current-architecture-v2)
3. [Requirements](#3-requirements)
4. [Options Evaluation](#4-options-evaluation)
5. [Recommended Approach](#5-recommended-approach)
6. [Build Plan](#6-build-plan)
7. [Test Strategy](#7-test-strategy)
8. [Risk Assessment](#8-risk-assessment)
9. [Decision Records (ADRs)](#9-decision-records-adrs)
10. [Open Questions](#10-open-questions)
11. [Post-Implementation Findings (V3.0 results)](#11-post-implementation-findings-v30-results)
12. [V3.1: Constraint-Conditioned (Retrieval) Pre-training](#12-v31-constraint-conditioned-retrieval-pre-training)

---

## 1. Problem Statement

V2 trains a decoder-only transformer to play Wordle using a dense constraint-state
encoding. It works on the curated NYT answer set (2,315 words) plus the extended
guess list (10,657 words) for validity checks. Two things limit it as a learning
exercise:

1. **The game is artificially easy.** Strong Wordle solvers exploit the fact that
   answers come from a curated 2,315-word pool. With only 2,315 possible answers,
   candidate sets collapse fast and 6-guess wins are nearly free. There is little
   room to study *strategy* or *generalization*.

2. **There is no held-out set and no clean generalization story.** Every answer
   word is used as both a pre-training target and an RL target
   (`wordle2/finetune.py:193` samples targets from the full `load_answers()` list;
   `mm_wordle/transcripts.py:73` does the same for pre-training). The only split is
   an example-level validation slice sized by `data.val_fraction`
   (`wordle2/data.py:122-127`, 5% in `large.yaml`). We can't ask "can the model
   solve a word it was never trained to solve?"

3. **V2 conflates two distinct skills in one phase.** What V2 calls "pre-training"
   is actually behavior cloning on game transcripts (constraint-state → guess,
   `wordle2/data.py:99-115`). The model never learns the *lexicon* (what counts as
   a valid English 5-letter word) independently of game play. The two skills —
   "know the words" and "play the game" — are entangled.

4. **The RL rollout path is slow.** Group sampling runs `group_size` independent
   autoregressive rollouts, each re-encoding the full sequence on every decode step
   with no KV cache (`wordle2/finetune.py:55-76`, `245-248`). Log-probs are
   recomputed one guess at a time in Python loops (`finetune.py:264-270`,
   `319-321`). Reward computation walks the full answer list in pure Python for the
   "best available word" normalization (`mm_wordle/reward.py:50-66`). Nothing uses
   AMP or `torch.compile` (confirmed absent across `libs/` and `wordle2/`).

**Impact of not fixing:** the exercise plateaus. We can't study scaling, can't
measure generalization, and RL iteration is wall-clock-bound on a problem that
should be cheap.

---

## 2. Current Architecture (V2)

### Representation (to be preserved — see R1)

The "dense encoding" is implemented by `ConstraintTokenizer`
(`wordle2/tokenizer.py`). Vocabulary is **265 tokens**:

| Range | Content | Count |
|-------|---------|-------|
| 0–25 | plain letters `a`–`z` (generated guesses) | 26 |
| 26 | `?` (unknown green slot) | 1 |
| 27–52 | `a-green` … `z-green` | 26 |
| 53–182 | yellows, **position-major**: `a-yellow-1`,`b-yellow-1`,…,`z-yellow-1`,`a-yellow-2`,… | 130 |
| 183–260 | grays, **count-major**: all `*-gray-0`, then all `*-gray-1`, then all `*-gray-2` | 78 |
| 261–264 | `[bos] [sep] [pad] [eos]` | 4 |

Construction order matters for ADR-5 (promoting the tokenizer to a lib): yellows
are built `for pos in 1..5: for ch in a..z` (`tokenizer.py:41-46`) and grays
`for count in 0..2: for ch in a..z` (`tokenizer.py:48-53`) — not letter-major.

A game state encodes to `[bos] <5 green slots> [sep] <sorted facts>`
(`tokenizer.py:97-152`). Facts are deduplicated yellows (`{ch}-yellow-{pos}`) and
gray counts (`{ch}-gray-{min(count,2)}`), sorted alphabetically so the same
constraint set always yields the same token sequence. Guesses are generated as 5
plain-letter tokens; at inference a `-inf` mask on non-letter logits forces valid
letters (`finetune.py:142-144`).

### Model

`mm_model.GPT` (`libs/mm-model/src/mm_model/model.py`): decoder-only, **RoPE**
(no learned positional embeddings), fused QKV, `F.scaled_dot_product_attention`
with `is_causal` (dispatches to FlashAttention on CUDA), KV-cache support
(`model.py:75-80`), pre-LN blocks, GELU MLP at 4× expansion, **no weight tying**
between `token_emb` and `lm_head` (`config.py:63-66`). Config presets in
`config.py:25-33`. Wordle2 configs use `context_len: 128`.

| Config | layers | heads | embed | ≈ params |
|--------|--------|-------|-------|----------|
| `small.yaml` | 4 | 4 | 128 | ~1M |
| `medium.yaml` | 6 | 8 | 256 | ~5M |
| `large.yaml` | 8 | 8 | 320 | ~10M |

### Training pipeline

- **Pre-train** (`wordle2/pretrain.py`): behavior cloning on transcripts generated
  by a 30/40/30 random/decent/good solver mix (`mm_wordle/transcripts.py:59-91`),
  one example per turn, late turns oversampled N× (`wordle2/data.py:64-78`).
  AdamW + cosine schedule (`mm_training/optim.py`), loss masked to the 5 target
  chars. ~100k games → ~1.3M examples for `large.yaml`.
- **RL** (`wordle2/finetune.py`): custom PPO-clip loop using
  `mm_grpo.compute_group_advantages` for within-group normalization. Two phases
  controlled by `rl.curriculum_phase`:
  - **Phase 1** (`max_turns: 2`): policy plays turns 1–2, reward = raw expected
    info gain (`compute_reward(..., composite=False)`, `reward.py:142-143`).
  - **Phase 2** (`max_turns: 6`): a frozen opener (the Phase-1 checkpoint) plays
    turns 1–2 at temp 0.1; policy plays turns 3–6 with the composite reward —
    normalized info gain × 10 + endgame/solve bonuses − invalid penalty
    (`reward.py:145-157`).

### UI contract (must keep working — see R7)

The dashboard (`wordle/dashboard/app.py`, FastAPI + HTMX) is fully decoupled from
the model. It reads three things from a run directory:

1. `live/latest.json` (every 3s): `{step, loss, kl_div, clip_fraction, games:[...]}`
   where each game is `{target, guesses, feedback, solved, turns, turn_rewards?}`
   and `feedback` uses the **string** values `"green"|"yellow"|"gray"`
   (`LetterFeedback.value`). V2 already writes this (`finetune.py:344-353`).
2. `live/history.jsonl` (every 10s): one `{step, loss, kl_div}` per line.
3. `eval-{N}/snapshot.json` (every 10s): `{step, win_rate, avg_guesses}`.
   **V2 never writes this** — it only logs `eval/win_rate` to TensorBoard
   (`finetune.py:364`), so the dashboard's eval chart is empty today.

The data models live in `libs/mm-viz/src/mm_viz/data.py` (`GameReplay`,
`EvalSnapshot`). The renderer `render_game_html` zips `guesses`/`feedback` with
`strict=True` (`board.py`), so the two arrays must be equal length.

---

## 3. Requirements

### Must-have

- **R1 — Preserve the V2 representation.** The constraint-state encoding
  (`ConstraintTokenizer`, vocab 265) is unchanged in every phase. No new tokens.
- **R2 — Adopt the 14,855-word set.** Use the canonical full valid-guess list
  (dracos/tabatkins, verified 14,855 entries). This is both the guess universe and
  the answer universe (minus hold-out).
- **R3 — Three phases.** (1) **Pre-train** on word data only, *no game state*;
  (2) **SFT** introducing the game format on "golden" games; (3) **RL**, the
  existing two-phase curriculum.
- **R4 — Hold-out set.** A reproducible hold-out subset of answers. **All** words
  (including hold-out) are seen during pre-training; hold-out words are **never the
  answer** in SFT or RL. Enforced and tested across phases.
- **R5 — Generalization evaluation.** Report the eval trio (valid-word rate /
  info gain / win rate) — plus avg guesses **over wins only** — separately on
  in-distribution and hold-out answers; the win-rate *generalization gap* is a
  first-class metric.
- **R6 — Speed.** Concrete, measured optimizations to RL rollout, reward
  computation, and the training loop. Larger model supported.
- **R7 — UI support.** Preserve the `live/latest.json` + `live/history.jsonl`
  contract exactly, and additionally emit `eval-{N}/snapshot.json` so the eval
  chart populates. Surface the hold-out metric to the UI.

### Nice-to-have

- **N1** — Cache the generated dataset + precomputed feedback matrix to disk so
  repeated runs skip regeneration.
- **N2** — Optional prefix/word-completion augmentation during pre-training (still
  word-only, no Wordle feedback).
- **N3** — A small dashboard tweak to show hold-out vs in-distribution win rate
  side by side.

### Constraints

- **C1** — Python 3.12+, PyTorch, `uv`. `make check` (ruff + ty + pytest) must pass.
- **C2** — Type hints everywhere; seed all RNGs; log hyperparameters
  (`RunManifest`).
- **C3** — V3 lives in its own top-level `wordle3/` directory (repo invariant:
  one directory per exercise). Reusable pieces are extracted into `libs/`.
- **C4** — Do not modify or break V1 (`wordle/`) or V2 (`wordle2/`).

---

## 4. Options Evaluation

The core decision is the overall training strategy. Three end-to-end options:

### Option A — "Bigger of the same"

Keep V2's two-phase pipeline (BC-pretrain + two-phase RL); only swap the word list
to 14,855, add a hold-out, and bump the model.

- ➕ Least new code; reuses `wordle2/` almost verbatim.
- ➖ Violates **R3** (no word-only pre-train, no SFT/golden distinction).
- ➖ The lexicon and game-play skills stay entangled; weak generalization story —
  the model only ever "knows" a word because it cloned a game that used it.

### Option B — Three-phase curriculum *(RECOMMENDED)*

Word-only pre-train over **all** 14,855 words → SFT (behavior cloning on golden
games whose answers are train-only) → two-phase RL (train-only answers). Hold-out
enforced from SFT onward. Pre-training uses the **empty constraint prompt**
(`[bos] ????? [sep] → word`) so the V2 representation is reused unchanged.

- ➕ Meets R1–R7. Cleanly separates "know the lexicon" (pre-train) from "play the
  game" (SFT+RL), which is exactly what enables the generalization test: hold-out
  words live in the weights as *known words* but are never reinforced as answers.
- ➕ Smooth curriculum — pre-train is literally the turn-1 special case of SFT.
- ➖ Most new scaffolding (three scripts, hold-out plumbing, faster reward).

### Option C — Three-phase + architecture change

As B, but swap the character decoder for a word-classifier or retrieval head to
handle the 6× larger lexicon.

- ➕ A classifier over 14,855 words could be more sample-efficient.
- ➖ **Violates R1** (changes the representation/output). Large scope. The whole
  point of keeping the char decoder is to test the *curriculum*, not the head.

| Criterion | A | B | C |
|-----------|---|---|---|
| Meets R1 (keep representation) | ✅ | ✅ | ❌ |
| Meets R3 (three phases) | ❌ | ✅ | ✅ |
| Generalization story | weak | strong | strong |
| New code | low | medium | high |
| Risk | low | medium | high |

**Recommendation: Option B.** It is the only option that satisfies all
must-haves while preserving the representation.

### Sub-decision: model size

Pre-training must memorize the full lexicon. Napkin math: representing 14,855
words as letter patterns is ~14,855 × 5 × 26 ≈ 1.9M values (6.4× V2's ~300K). V2's
10M model held 2,315 words comfortably; scale capacity for the bigger lexicon.
Param counts (vocab=265, no bias, `context_len: 128`, formula
`2·v·d + n·(12d² + 4d) + 2d`):

| Tier | layers | heads | embed | ≈ params | role |
|------|--------|-------|-------|----------|------|
| `large` (exists) | 8 | 8 | 320 | ~10.0M | fast baseline / sanity |
| `xl` | 10 | 6 | 384 | ~17.9M | intermediate |
| `xxl` *(recommended)* | 12 | 8 | 512 | ~38.0M | primary for the 14,855 challenge |

**Recommendation:** train `xxl` (~38M) as primary, keep `large` (~10M) as a fast
baseline to confirm the pipeline and quantify the capacity effect. `context_len`
stays at 128 — even the worst-case constraint prompt (5 greens + ~10 yellows + ≤26
grays + 5 target chars) is well under 128, and short contexts keep training cheap.

### Sub-decision: pre-train example format

| Variant | Example | Verdict |
|---------|---------|---------|
| Pure word-LM | `[bos] w o r d s [eos]` | Works, but the constraint/`[sep]` tokens get no gradient until SFT (cold embeddings). |
| **Empty constraint prompt** | `[bos] ? ? ? ? ? [sep] w o r d s` | **Chosen.** Identical to the V2 turn-1 state; pre-train is the zero-constraint special case of the game, so SFT is a pure continuation. |
| Prefix-completion augmented | also train on word prefixes | Nice-to-have (N2); stronger lexical features, still feedback-free. |

---

## 5. Recommended Approach

### 5.0 Architecture overview

```
                 14,855-word valid set  (mm_wordle: full_word_set)
                          │
        deterministic split (seed) ──► split.json {train_answers, holdout}
                          │                         │
                          ▼                         │ (holdout NEVER an answer
   Phase 1  PRE-TRAIN ─ ALL 14,855 words ─┐         │  below this line; still a
   (word-only, empty constraint prompt)   │         │  valid guess + candidate)
   metric: valid-word rate, lexicon recall│         ▼
                          ▼               │   Phase 2  SFT
                 pretrain checkpoint ──────┴─► golden games on train_answers
                                              (behavior cloning, constraint→guess)
                                                      │
                                                      ▼
                                              Phase 3  RL (two-phase)
                                              P1 openers (turns 1-2, info gain)
                                              P2 mid/late (turns 3-6, composite)
                                              targets ∈ train_answers
                                                      │
                                                      ▼
                              EVAL: win rate / avg guesses on
                              {train_answers sample}  vs  {holdout}
                              + writes live/*.json, eval-N/snapshot.json (UI)
```

Candidate pool (the set the reward reasons about as "remaining possible answers")
is the **full** 14,855-word universe `U` in every phase and at eval — hold-out
words remain valid candidates so a hold-out game is genuinely solvable. Only the
*target sampling* is restricted to `train_answers`.

### 5.1 Dataset acquisition & splits

Add the 14,855-word list to `mm_wordle` as a package data file
(`libs/mm-wordle/src/mm_wordle/data/valid_words_14855.txt`, loaded via
`importlib.resources`) rather than a giant Python tuple. New API in
`mm_wordle/words.py` (V1/V2 `ANSWERS`/`VALID_GUESSES` stay untouched, C4):

```python
def load_full_word_set() -> list[str]:               # 14,855 sorted words = U = G
    ...

def split_answers(holdout_frac: float = 0.10,
                  seed: int = 1234) -> tuple[list[str], list[str]]:
    """Deterministic (train_answers, holdout) split of U. Same seed => same split."""
    words = load_full_word_set()
    rng = random.Random(seed)
    shuffled = words.copy(); rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * holdout_frac)
    holdout = sorted(shuffled[:n_holdout])
    train = sorted(shuffled[n_holdout:])
    return train, holdout
```

- **U (answer universe / candidate pool)** = all 14,855 words.
- **G (valid guess set)** = all 14,855 words. A guess ∉ G gets `INVALID_WORD_PENALTY`.
- **H (hold-out)** ≈ 1,486 words (10%). Tunable via `holdout_frac`.
- **train_answers** = U − H ≈ 13,369 words.

**Source of truth = one file.** A tiny script `wordle3/make_split.py` writes
`wordle3/data/split.json` (`{seed, holdout_frac, train_answers:[...], holdout:[...]}`)
**once**, and is committed. Every phase loads *this file*, never re-derives the
split. This guarantees pre-train, SFT, RL, and eval agree on the same hold-out.

### 5.2 Phase 1 — Pre-train (word-only)

**Goal:** bake the entire lexicon into the weights. The model must learn that all
14,855 strings (hold-out included) are real words and be able to generate them.

- **Data:** every word in U exactly, encoded as the empty constraint prompt
  `[bos] ? ? ? ? ? [sep]` (the canonical turn-1 state) with target = the 5 letters.
  Loss masked to the 5 target positions (reuse `ConstraintDataset` from
  `wordle2/data.py:18-45`). Uniform over all words — hold-out words appear at the
  same frequency as train words, so the model is never biased away from them here.
- **No game state, no feedback, no transcripts** — this is purely
  `P(word | no constraints)`.
- **Objective:** cross-entropy; train to convergence on the eval metrics below.
- **(N2, optional)** add prefix-completion examples (`[bos] ? ? ? ? ? [sep] w o r`
  → next letter) to strengthen lexical features. Still feedback-free.

**Eval (primary = valid-word rate):**
- *Valid-word rate*: sample N words at temp ≈ 1.0; fraction that are in G. Target ≈ 100%.
- *Lexicon recall*: per-word mean NLL over **all** of U; specifically check that
  hold-out words have NLL comparable to train words (they must, since pre-train is
  uniform). This is the precondition for the whole generalization test.
- *Diversity*: count of distinct valid words generated in K samples (guard against
  mode collapse onto a few words).

### 5.3 Phase 2 — SFT on golden games

**Goal:** teach the game format — map a non-empty constraint state to a strong
next guess — by behavior cloning on high-quality demonstrations.

- **Golden games need a new solver — the existing one degrades at this scale.**
  `play_game_good` falls back to `random.choice(candidates)` whenever
  `len(candidates) > 500` (`solver.py:99`), so over the 14,855-word universe
  *every golden game would get a random opener* and random play for the first
  couple of turns — useless as a demonstration. `entropy_guess` also only samples
  200 candidates (`solver.py:58`) and resets empty candidate lists to `answers`
  rather than U (`solver.py:103-104`). V3 therefore builds a **pattern-matrix
  solver** (`mm_wordle/golden.py`): (1) a single **precomputed best opener over U**
  (argmax expected info gain across G, computed once from the §5.7-A matrix and
  cached); (2) for turns 2+, exact info-gain selection over `candidates ∪ top-K
  global probes` using the matrix — no 200-sample cap, no >500 fallback; (3) when
  ahead-of-game candidate filtering empties out, reset to **U**, not train_answers.
- **`generate_golden_examples(...)`** then (a) samples targets **only from
  `train_answers`**, (b) plays with the pattern-matrix solver over candidate pool
  **U** (the hard game), (c) converts each turn to a `(ConstraintTokenizer state,
  guess)` example. Late-turn oversampling kept (`data.py:64-78`).
- **Hold-out enforcement:** assert every golden game's `target ∈ train_answers`.
  Hold-out words may still appear as *guesses* (a strong solver may probe with one)
  — that is permitted by R4 ("not as **answers**"). See ADR-4 for the strict
  variant and its trade-off.
- **Training:** identical loop to pre-train (cross-entropy, masked to guess
  letters), warm-started from the pre-train checkpoint. The empty-prompt pre-train
  task is exactly the turn-1 subset of this data, so it's a smooth continuation.
- **(N1)** keep a small replay fraction (e.g., 5%) of word-only pre-train batches
  mixed into SFT to retain hold-out lexical coverage (mitigates the forgetting
  risk in §8).

**Eval:** valid-word rate **by turn** (target ≈ 100% at all turns),
*constraint-consistency rate* (fraction of guesses that satisfy the current
constraint state — a clean signal the model reads the encoding), and greedy win
rate on a `train_answers` sample vs the `holdout` sample.

### 5.4 Phase 3 — RL (two-phase, unchanged algorithm)

Port `wordle2/finetune.py` to `wordle3/finetune.py` with three changes: (1)
targets sampled from `train_answers` (not all answers); (2) the reward universe is
U (14,855), wired through the reward module (§5.6); (3) the faster rollout/reward
paths (§5.7) and `eval-{N}/snapshot.json` writing (§5.5). The curriculum is
unchanged:

- **Phase 3a (openers, turns 1–2):** reward = raw expected info gain
  (`composite=False`). Over U, good openers have higher absolute info gain than
  over the 2,315 set, so the "good opener" threshold is re-derived from U
  (precompute the top openers over U; report the achieved percentile rather than a
  hard-coded 5.5 bits).
- **Phase 3b (mid/late, turns 3–6):** frozen opener (3a checkpoint) plays turns
  1–2; policy plays 3–6 with the composite reward, normalized against the best
  available word **over G** (see ADR-3). KL-to-reference and PPO-clip unchanged
  (`finetune.py:314-340`).

**Eval** (greedy, temp ≈ 0.1, like `finetune.py:96-125`, run every
`eval_interval` and written to `eval-{N}/snapshot.json`):
- *Phase 3a:* expected info gain for turns 1–2 and the achieved **opener
  percentile** vs the best opener over U; valid-word rate (must not regress).
- *Phase 3b / overall:* **win rate** and **avg guesses (over wins only)** on a
  `train_answers` sample *and* on the `holdout` sample; the **generalization gap** =
  `win_rate(train) − win_rate(holdout)` is the headline metric (R5); info gain by
  turn; invalid-word rate (should stay ≈ 0); KL-to-reference (policy-drift
  tripwire). Also track hold-out NLL as the forgetting tripwire (§8 R-1).

### 5.5 UI support (R7)

- **Unchanged contract:** keep writing `live/latest.json` and `live/history.jsonl`
  exactly as `wordle2/finetune.py:344-355` does (same keys, `feedback` as
  `"green"|"yellow"|"gray"` strings, `guesses`/`feedback` equal length).
- **New (fixes the empty eval chart):** at every eval, write
  `runs/.../eval-{step}/snapshot.json` as a **plain dict** carrying the eval trio +
  avg guesses for train and hold-out — `{step, win_rate, valid_word_rate, info_gain,
  avg_guesses, holdout_win_rate, holdout_valid_word_rate, holdout_info_gain,
  holdout_avg_guesses, opener_*}` (not `EvalSnapshot.save()` — that dataclass
  requires `checkpoint_path`/`replays` we don't need here, `data.py:72-91`). The
  dashboard reads `win_rate`/`avg_guesses` via `.get()` (`app.py:37-42`) and ignores
  extras, so this is backward-compatible. **`avg_guesses` is over wins only** —
  losses are excluded, not counted as 6 (which would just re-encode win rate).
- **Per-step trio (see §5.9):** every phase writes `valid_word_rate`, `info_gain`,
  and `win_rate` into both `live/latest.json` (top bar) and each `live/history.jsonl`
  line, in addition to `loss` (and `kl_div`/`clip_fraction` in RL). Extra keys are
  backward-compatible (dashboard reads by key).
- **Dashboard edit (now in-scope, was N3):** add three step-series charts
  (valid-word rate, info gain, win rate) fed from `history.jsonl`, and plot
  `holdout_win_rate` alongside `win_rate` from `snapshot.json`. This is the only
  dashboard code change V3 requires.

### 5.6 Reward universe wiring

`mm_wordle/reward.py` currently hardcodes the universe at import
(`_ANSWERS = load_answers()`, `_VALID_WORDS = all_valid_words()`,
`reward.py:15-16`) and searches `_ANSWERS` for the best word
(`reward.py:60`). At 14,855 words this is doubly broken: it's slow, and the
`_BEST_IG_CACHE` keyed by `tuple(candidates)` (`reward.py:47-66`) is effectively
**disabled** — after turn 1 every game has a unique candidate set, so it's a
cache-miss every time (and each key is a ~74 KB tuple of strings). V3 replaces
this with an explicit configuration entry point backed by the pattern matrix:

```python
def configure_reward(
    candidate_universe: list[str],   # U = all 14,855 (best-word search runs over this)
    best_word_search: list[str],     # G = all 14,855 valid guesses (denominator domain)
    valid_words: set[str],           # G, for the invalid-word penalty
    pattern_matrix: "PatternMatrix",
) -> None: ...
```

called once at the start of SFT/RL. **The best-available-word denominator must
search over G (all 14,855), and candidate filtering over U (all 14,855) — never
`train_answers`.** Restricting either to `train_answers` would bias normalization
against hold-out games (where a hold-out word can be the true best guess) and is a
hold-out leak in reverse — see the test in §7. This removes the hidden global
coupling and lets V3 use U/G without touching V1/V2 behavior (which keeps calling
the existing helpers).

### 5.7 Speed optimizations (R6)

Ordered by expected wall-clock impact on RL (the bottleneck).

**A. Precomputed feedback-pattern matrix (biggest reward win).** Wordle feedback is
a function only of (guess, target). Precompute `P[g, t] ∈ [0, 243)` (one of 3⁵
patterns) as a `uint8` matrix over G×U. Size = 14,855 × 14,855 × 1 byte ≈ 220 MB —
build once, `np.memmap` to disk (N1), keyed by the word-set hash. Then:
  - *expected info gain* of guess g over candidate mask = entropy of
    `np.bincount(P[g, cand_idx])` — a vectorized op, no Python feedback loop.
  - *candidate filtering* after playing g with observed pattern p =
    `cand_idx[P[g, cand_idx] == p]` — a boolean mask, replacing
    `solver.filter_candidates`'s per-word Python loop.
  - *best available word* (normalization denominator) = argmax info gain over rows;
    bound the search to `candidates ∪ top-K global probes` (K≈300) so it's O(K·|cand|)
    instead of O(|G|·|cand|). Building P itself is vectorized per-row over targets
    with numpy (compare letters as arrays), not the scalar `compute_feedback`.

This replaces the pure-Python hot loops at `reward.py:29-44`, `50-66` and the
solver filtering used throughout RL, and makes the larger universe affordable.

**B. Batched group sampling with KV cache.** Today each of `group_size` guesses is
a separate rollout and each decode step re-runs the full prefix with no cache
(`finetune.py:55-76`, `245-248`). This needs a **new decode loop**, not just
wiring: the model's `generate()` is single-sequence (`model.py:163-214`). But
group sampling is the easy batched case — all members share the prompt and decode
in **lockstep at the same position**, so we can prime the shared KV cache once with
the `(group_size, prompt_len)` prompt, then run 5 single-token batched steps
(`forward` already accepts `(batch, seq)` with a shared `kv_cache` list and
per-position RoPE via `start_pos`, `model.py:75-80`). No per-sequence position
bookkeeping is required. This is ≈ `group_size` × (prefix re-encode savings) fewer
FLOPs on the dominant path.

**C. Batched log-probs.** `compute_guess_log_probs` is called in Python loops for
old/ref/current log-probs (`finetune.py:264-270`, `319-321`). For **old and ref**
log-probs use `mm_grpo.collect_completions_log_probs` (`mm_grpo/utils.py:34`) — but
note it hardcodes `torch.no_grad()` (`utils.py:67`), so it is **only** valid for
the detached old/ref pass. The **current-policy** log-probs in the PPO update need
gradients, so add a sibling gradient-enabled batched function (same shape logic,
no `no_grad`) — do not reuse `collect_completions_log_probs` there.

**D. AMP / bf16 autocast.** Wrap forward/backward in `torch.autocast`. Use `bfloat16`
on CUDA (no GradScaler needed); fall back to fp32 on MPS where bf16 coverage is
partial. Add `torch.backends.cuda.matmul.allow_tf32 = True` (free on Ampere+).

**E. `torch.compile`.** Wrap the model once after construction
(`model = torch.compile(model)`), CUDA-gated (skip on MPS where it's unreliable).
Biggest gain on the many repeated small forwards in RL.

**F. Dataset & matrix caching (N1).** Cache generated golden examples and the
pattern matrix to disk keyed by (word-set hash, seed, config). Avoids the
`ProcessPoolExecutor` regeneration (`transcripts.py:84-85`) on every run.

**G. Larger pre-train batch.** Pre-train examples are tiny (≤ ~12 tokens); push
`batch_size` up (512+) to saturate the device.

Each optimization is independently testable; land A–C first (RL-dominant), then
D–G.

### 5.8 Evaluation matrix (which evals run per phase)

| Metric | Pre-train | SFT | RL 3a | RL 3b | Primary gate for |
|--------|:---:|:---:|:---:|:---:|---|
| Valid-word rate (overall) | ✅ ≈100% | — | ✅ no-regress | ✅ no-regress | Pre-train |
| Valid-word rate **by turn** | — | ✅ ≈100% all turns | — | — | SFT |
| Lexicon recall / per-word NLL (train vs **hold-out**) | ✅ gate | tripwire | — | tripwire | Pre-train (+forgetting tripwire after) |
| Generation diversity (distinct valid words) | ✅ | — | — | — | Pre-train (mode-collapse guard) |
| Constraint-consistency rate | — | ✅ | — | — | SFT |
| Expected info gain **by turn** | — | — | ✅ (turns 1–2) | ✅ | RL 3a |
| Opener percentile vs best-over-U | — | — | ✅ | — | RL 3a |
| **Win rate** (train sample) | — | ✅ greedy | — | ✅ | RL 3b |
| **Win rate (hold-out sample)** | — | ✅ greedy | — | ✅ | R5 generalization |
| Avg guesses (train / hold-out), **over wins only** | — | ✅ | — | ✅ | RL 3b |
| **Generalization gap** = WR(train) − WR(hold-out) | — | reported | — | ✅ **headline** | R5 |
| KL-to-reference | — | — | ✅ | ✅ | RL drift tripwire |
| Invalid-word rate | — | — | ✅ ≈0 | ✅ ≈0 | RL |

Notes: every phase writes `live/history.jsonl` and, from SFT onward,
`eval-{N}/snapshot.json` (win/avg-guesses incl. hold-out) so the dashboard charts
populate for any phase (§5.5). Eval games are greedy at temp ≈ 0.1; train- and
hold-out-eval target sets are disjoint by construction (the §7 hard gate).

### 5.9 Per-step metrics — valid-word rate · info gain · win rate (ALL phases)

**Requirement:** log all three of *valid-word rate*, *info gain*, and *win rate*
**every training step, in every phase** (pre-train, SFT, RL 3a, RL 3b). These three
form the standard per-step trio plotted by the dashboard (§5.5); the heavier
phase-specific diagnostics in §5.8 (by-turn breakdowns, NLL, generalization gap)
remain on the `eval_interval` cadence.

The three differ sharply in cost, so each is computed by the cheapest source that
still produces it every step:

| Metric | Cost | Per-step source |
|--------|------|-----------------|
| **valid-word rate** | cheap | The fraction of guesses the model just produced that are in G. In SFT/RL, the batch already generated guesses → free. In pre-train, greedy-decode the step's batch prompts (all empty-state) and check membership → one extra batched forward. |
| **info gain** | cheap *with the §5.7-A matrix* | In RL it's already computed for the reward → free. In pre-train/SFT, score the just-generated (or mini-eval) guesses against the live candidate pool via the pattern matrix → vectorized, ~free. |
| **win rate** | **expensive** (needs full games) | In **RL**, the batch plays full games every step → use the **batch solve-rate**, free. In **pre-train/SFT** the step is teacher-forced (no game play), so play a small fixed **mini-eval set** every step. |

**The mini-eval (pre-train/SFT win rate + info gain at every step).** A fixed set
of `step_eval_games` targets (default **16**, drawn from `train_answers`, plus the
same count from `holdout`), played greedily at temp ≈ 0.1, batched with the §5.7-B
KV-cached decode so a step's mini-eval is ~`6 turns × 5 tokens = 30` batched
single-token forwards over a 16-wide batch — cheap next to a 512-row training
forward/backward. Config knob `step_eval_games` (set `0` to fall back to
interval-only). The fixed set is seeded and held constant across a run so the
per-step series is comparable step-to-step.

**Noise & interpretation.** Per-step win rate is intentionally a *small-sample,
smoothed-trend* signal (16–32 games, or one RL batch of `batch_size`); it is **not**
the headline number. The precise win rate is the `eval_interval` full eval over the
hold-out set and a train sample, written to `eval-{N}/snapshot.json` (§5.5). The
per-step series gives a live training pulse; the interval eval gives the number we
report.

**Explicit eval hold-out + decoupled seed (eval hygiene).** The **hold-out set is
the generalization eval set** — its words are *never* used as a training answer
(R4), so hold-out win rate is a clean, seed-robust measure of skill on unseen
words. Eval target sets are sampled with a **fixed seed decoupled from the training
seed** (`steplog.EVAL_SEED`). This matters: an earlier version sampled the train
eval with the *training* seed, which correlated the sample with the training-target
stream so it landed on **memorized** answers and inflated "train win rate" ~2×
(observed 93% vs a true ~47%). The train number is only a secondary in-distribution
monitor and still includes whatever fraction of the sample was used as a training
answer (the model memorizes those, ~92%, vs ~36% on never-trained words) — so
**generalization is read from hold-out, not train**.

**Plumbing.** A shared `mm_training` helper (`StepMetrics` →
`{valid_word_rate, info_gain, win_rate}`) is logged to TensorBoard and written into
`live/latest.json` + `live/history.jsonl` by all three training scripts, so the
same dashboard charts work whichever phase's run dir it points at.

**Cost.** Measured in forward-pass *token-units* relative to **one pre-train step**
(batch 512 × ~12 tokens, fwd+bwd ≈ ~18k token-units):

| Phase | valid-word | info gain | win rate | trio overhead / step |
|-------|-----------|-----------|----------|----------------------|
| **RL** | free (batch guesses) | free (reward already computes it) | free (batch solve-rate) | **~0%** |
| **SFT** | free (batch guesses) | ~free (pattern matrix) | mini-eval | **≈ mini-eval** |
| **Pre-train** | +1 batched fwd (greedy-decode the batch) | ~free (pattern matrix) | mini-eval | **≈ mini-eval + ~2%** |

The only non-trivial term is the pre-train/SFT **win-rate mini-eval**: 16 games
(8 train + 8 hold-out) × ≤6 turns × (~20 prompt + 5 decode) ≈ **~4–5k token-units,
forward-only** (no backward). That's ≈ **10–15%** of a batch-512 pre-train step in
FLOPs. Two caveats and two levers:

- *Wall-clock caveat:* the mini-eval is ~30 **sequential** single-token decodes, so
  on GPU it's kernel-launch-bound — wall-clock overhead can run a bit above the FLOP
  share. RL is unaffected (no extra eval).
- *Lever 1 — training batch:* the mini-eval is **fixed-size**, so its relative cost
  shrinks as the pre-train batch grows (speed-opt G): ~15% at batch 512 → ~7% at
  1024 → ~4% at 2048.
- *Lever 2 — `step_eval_games`:* halve to 8 (~6–8%), or set `0` to drop per-step
  win rate and fall back to interval-only (valid-word rate + info gain stay free
  every step). Default **16**.

Net: the cheap two (valid-word rate, info gain) are **<2% always, all phases**;
per-step win rate is **free in RL** and a tunable **~5–15%** in pre-train/SFT. The
full headline eval is unchanged on the `eval_interval` cadence.

---

## 6. Build Plan

Phased, each with a verification gate. Estimates in engineering-days for this
solo learning repo.

| Phase | Work | Gate | Est. |
|-------|------|------|------|
| **0. Data + split** | Add 14,855 list to `mm_wordle`; `load_full_word_set`, `split_answers`; `wordle3/make_split.py` → `split.json`; unit tests. | `split.json` reproducible; `train ∩ holdout = ∅`; `set(train) ∪ set(holdout) == set(U)`; \|U\| = 14,855. | 0.5 |
| **1. Pattern matrix + reward wiring** | `mm_wordle/pattern.py` (build/memmap, info-gain, filter); `configure_reward`; equivalence tests vs current `reward.py`/`solver.py`. | New path matches old reward/filter on the 2,315 set bit-for-bit; ≥10× faster best-IG over U. | 1.5 |
| **2. Pre-train** | `wordle3/tokenizer.py` (reuse V2 — see ADR-5), `wordle3/data.py`, `wordle3/pretrain.py`, `configs/pretrain-{large,xxl}.yaml`. | Valid-word rate ≈ 100%; hold-out NLL ≈ train NLL. | 1.5 |
| **3. SFT** | `mm_wordle/golden.py` (pattern-matrix solver: precomputed best opener over U + exact mid-game selection); `generate_golden_examples`; `wordle3/sft.py`; replay mix (N1). | Golden games beat `play_game_good` avg-guesses on U; valid-word rate ≈ 100% by turn; constraint-consistency high; no holdout word as a golden answer (asserted). | 1.5 |
| **4. RL** | `wordle3/finetune.py` (port + faster rollout B/C + AMP/compile D/E + `eval-{N}/snapshot.json`); `configs/finetune-phase{1,2}-{large,xxl}.yaml`. | Win rate improves over SFT; UI live + eval charts populate; targets ∈ train_answers (asserted). | 2.5 |
| **5. Eval + UI** | Generalization eval (train vs holdout) wired into all phases; optional dashboard hold-out chart (N3); `wordle3/README.md`; `docs/v3-results.md`. | Generalization gap reported; `make check` green. | 1.0 |

Total ≈ **8.5 days**. Phases 0–1 are prerequisites; 2→3→4 are sequential
(checkpoints chain); 5 overlaps 2–4.

**Backward compatibility:** V3 is a new `wordle3/` tree. `mm_wordle` only gains
additive APIs; `ANSWERS`/`VALID_GUESSES`/`load_answers`/`all_valid_words` and the
existing `reward.py` behavior are untouched, so V1/V2 keep running (C4).

---

## 7. Test Strategy

- **Hold-out hard gate (R4).** A unit test asserts
  `holdout ∩ (SFT golden answers ∪ RL sampled targets) = ∅` over a seeded run, and
  that `split_answers` is deterministic for a fixed seed. This is the single most
  important correctness invariant — it is what makes the generalization number
  meaningful.
- **Pattern-matrix equivalence.** Property test: for random (guess, target) pairs,
  `P[g,t]` decodes to the same pattern as `WordleEnv.compute_feedback`; expected
  info gain and candidate filtering match `reward.py`/`solver.py` exactly on the
  2,315 set.
- **Reward-universe correctness (C1 guard).** Assert the best-available-word
  denominator searches G/U, not `train_answers`: for a candidate set whose true
  best guess is a hold-out word, `best_ig(candidates, domain=G) ≥
  best_ig(candidates, domain=train_answers)` and the configured reward uses the
  former. Guards against silently biasing normalization against hold-out games.
- **Split union.** Assert `set(train_answers) ∪ set(holdout) == set(load_full_word_set())`
  and `|U| == 14,855`, in addition to the disjointness gate below.
- **Tokenizer parity.** Reuse `wordle2/tests/test_tokenizer.py` against the
  shared/V3 tokenizer to guarantee R1 (representation unchanged).
- **Tiny end-to-end smoke tests** (repo guardrail: training tested by running with
  tiny models/datasets, fixed seed): a ~50-word toy universe runs pre-train → SFT →
  a few RL steps and produces `live/latest.json` + `eval-*/snapshot.json` matching
  the `GameReplay`/`EvalSnapshot` schema (guards R7).
- **UI contract test.** Validate emitted `latest.json`/`snapshot.json` against the
  exact keys the dashboard reads (`app.py:37-42`, `100-161`); assert `feedback`
  values ∈ {green,yellow,gray} and `len(guesses)==len(feedback)`.
- **Generalization metric test.** Eval harness returns distinct train/holdout
  numbers and the gap; assert it never samples holdout as a train-eval target.
- `make check` (ruff format/check, ty, pytest) is the umbrella gate per repo
  invariants.

---

## 8. Risk Assessment

### Risks of implementing

- **R-1 Distribution shift away from hold-out (highest).** SFT/RL reinforce
  train_answers as answers, which can pull the policy's word distribution toward
  train words and degrade hold-out recall (catastrophic forgetting) — directly
  attacking R5. *Mitigations:* (a) keep the candidate pool = full U so hold-out
  words are never marked impossible; (b) KL-to-reference in RL already anchors the
  policy near the SFT model; (c) replay a small fraction of word-only pre-train
  batches during SFT (§5.3, N1); (d) track hold-out NLL across phases as a tripwire.
- **R-2 Harder game → low absolute win rate.** With 14,855 possible answers, even
  optimal play wins less often in 6 guesses than on the curated 2,315 set. This is
  intended ("a challenge"), but tune expectations: report avg-guesses and info-gain
  trends, not just win rate. The *gap* (train vs holdout) is the headline, not the
  absolute.
- **R-3 Reward cost blow-up.** Best-available-word over U is O(|G|·|cand|) in the
  naive form. *Mitigation:* §5.7-A pattern matrix + bounded denominator search;
  this is a prerequisite (build Phase 1), not an afterthought.
- **R-4 Pattern matrix memory (220 MB).** Acceptable on disk via memmap; if RAM is
  tight, chunk row access. Built once, cached (N1).
- **R-5 Capacity vs lexicon.** ~38M may still under-fit 14,855 words. *Mitigation:*
  pre-train eval (valid-word rate, full-lexicon NLL) is the gate before SFT; bump
  to a larger tier if hold-out NLL stays high.
- **R-6 torch.compile / MPS flakiness.** *Mitigation:* CUDA-gate compile and bf16;
  fp32 eager fallback on MPS keeps local dev working.

### Risks of NOT implementing

- The exercise stalls on an easy problem with no generalization signal and slow RL;
  we can't study scaling, curriculum, or transfer — the stated learning goals.

---

## 9. Decision Records (ADRs)

**ADR-1 — Three-phase curriculum (pre-train / SFT / RL).** *Context:* V2 entangles
lexicon and game play. *Decision:* separate word-only pre-train from
golden-game SFT from RL. *Rationale:* enables the generalization test — hold-out
words enter the model as known words in pre-train but are never reinforced as
answers. *Consequences:* more scripts; a clean curriculum and a meaningful
generalization metric.

**ADR-2 — Empty constraint prompt for pre-training.** *Context:* "word data, no
game state" must coexist with R1. *Decision:* encode each word as the turn-1 state
`[bos] ????? [sep] → word`. *Rationale:* reuses the V2 representation verbatim and
makes pre-train the zero-constraint special case of the game, so SFT is a pure
continuation. *Consequences:* pre-train teaches only the marginal word
distribution; SFT must teach constraint-conditioning (intended).

**ADR-3 — Hold-out restricts answers, not candidates/guesses.** *Context:* a fair
generalization test needs the true answer to be reachable. *Decision:* hold-out
words are excluded only as *sampled targets* in SFT/RL; they remain in the
candidate pool U and the valid guess set G everywhere. *Rationale:* if the
candidate pool excluded hold-out, a hold-out game would be unsolvable by a
candidate-aware policy, making the test vacuous. *Consequences:* the model may
*guess* a hold-out word; that's realistic and consistent with R4 ("not as
**answers**").

**ADR-4 — Hold-out leakage definition = answers only (with a strict variant
noted).** *Context:* a strong golden/RL solver may probe with a hold-out word as a
*guess*, so the model is briefly trained to *emit* it. *Decision:* follow the
user's literal requirement — hold-out barred only as the *answer/target*. *Rationale:*
barring hold-out words as guesses too would distort golden play and shrink the
candidate pool, contradicting ADR-3; the model already "knows" every word from
pre-train, so a probe leak is minor. *Consequences:* the generalization claim is
"solves games whose **answer** it never trained on." A strict mode (also bar
hold-out guesses) is available behind a flag if we want the stronger claim, at the
cost of weaker golden play.

**ADR-5 — Promote `ConstraintTokenizer` into `mm_wordle`.** *Context:* the
representation is now stable and shared by V2 and V3 (R1, C3 "extract reusable
pieces"). *Decision:* move/duplicate `ConstraintTokenizer` into
`mm_wordle.constraint_tokenizer`; `wordle3/tokenizer.py` re-exports it; V2 keeps
its local copy to honor C4 (no V2 changes). *Rationale:* one source of truth for
the representation; the existing `test_tokenizer.py` becomes the shared spec.
*Consequences:* a small additive lib module; V2 untouched.

**ADR-6 — Precomputed pattern matrix as the reward/solver backend.** *Context:*
the 6× larger universe makes per-call Python feedback loops the RL bottleneck.
*Decision:* precompute a `uint8` G×U pattern matrix (memmap, cached) and serve
info gain / filtering / best-word from it. *Rationale:* turns the hot loops into
vectorized numpy and makes the larger universe affordable; equivalence-tested
against the current implementation. *Consequences:* 220 MB artifact; a one-time
build cost amortized by caching.

**ADR-7 — Model size: `xxl` (~38M) primary, `large` (~10M) baseline.** *Context:*
6.4× lexicon to memorize. *Decision:* primary runs at ~38M (12L/8H/512d), with a
~10M baseline. *Rationale:* capacity to store the lexicon while staying small
enough for fast iteration; the baseline quantifies the capacity effect.
*Consequences:* longer pre-train; `context_len` stays 128.

**ADR-8 — Purpose-built golden solver instead of `play_game_good`.** *Context:*
`play_game_good` plays a *random* opener whenever `len(candidates) > 500`
(`solver.py:99`) and `entropy_guess` samples only 200 candidates
(`solver.py:58`); over the 14,855-word universe this makes turn-1 (and early)
golden play effectively random — the opposite of "golden". *Decision:* build
`mm_wordle/golden.py` on top of the §5.7-A pattern matrix: one precomputed best
opener over U, then exact info-gain selection over `candidates ∪ top-K probes` for
later turns, resetting empty candidate lists to U (not train_answers). *Rationale:*
SFT quality is bounded by demonstration quality; the matrix makes exact selection
cheap enough to drop both the 200-sample cap and the >500 random fallback.
*Consequences:* a new solver module (reused by golden-game generation and the RL
eval baseline); `play_game_good` stays for V1/V2.

---

## 10. Open Questions

- **OQ-1 hold-out fraction.** 10% (~1,486 words) is the default. Larger → stronger
  generalization signal but fewer training answers and a harder game. Revisit after
  the first end-to-end run.
- **OQ-2 candidate-pool size at RL time.** Filtering against the full 14,855 every
  turn is correct but heavier than the 2,315 baseline even with the pattern matrix.
  If RL is still rollout-bound after §5.7, consider capping per-game candidate
  tracking (e.g., sample a fixed-size consistent subset) — but only if measured
  necessary, since it changes reward semantics.
- **OQ-3 strict hold-out mode (ADR-4).** Decide whether the stronger
  "never even guessed" claim is worth the distorted golden play; defaulting to off.

---

## 11. Post-Implementation Findings (V3.0 results)

Running the full V3.0 pipeline and a read-only diagnostic (`wordle3/diagnose.py`)
overturned the initial headline numbers, surfaced two eval bugs, and isolated the
real failure mode.

### 11.1 Two eval bugs (both fixed)

- **`avg_guesses` counted losses as 6** (the guess cap), so it just re-encoded win
  rate. Fixed to average over **wins only** (v0.5).
- **Eval sampled targets with the *training* seed.** The "train" eval sample
  correlated with the training-target stream and landed on **memorized** answers,
  inflating train win rate ~2× (reported **93%** vs a true **~47%**). Fixed by
  decoupling the eval seed (`steplog.EVAL_SEED`); **hold-out is the generalization
  metric** and was unaffected (its words are never training answers).

### 11.2 The honest result

| solve rate (final RL checkpoint) | value |
|----------------------------------|-------|
| words trained on as answers | ~92% |
| untrained train-split words | ~38% |
| hold-out words | ~34% |

The model **memorizes** the answers it trains on and **generalizes weakly (~36%)** —
identically whether the unseen word is in the *train* split or the *hold-out*
split. **The train/hold-out split is nearly irrelevant; the real axis is
seen-as-answer vs not.** Only ~26% of train words (3,485/13,370) were ever used as
a golden answer, and RL (phase 2, 1k steps) was effectively a no-op (KL-pinned).

### 11.3 The failure mode

The model **narrows to the answer but won't commit**: losses have a **median of 1**
remaining candidate, yet it guesses the lone candidate only **9–18%** of the time.
It reliably emits only *memorized* answers; for unseen words it can't invert
"constraints → the one consistent word", so it keeps probing and times out.

**Conclusion:** the bottleneck is **constraint→word retrieval / endgame
commitment**, not capacity (pre-training already fits the lexicon; narrowing is
good). Capacity (MoE/38M) and inference crutches (constrained decoding) do not
address it — and a crutch defeats the point (the exercise is to *train* the model).

## 12. V3.1: Constraint-Conditioned (Retrieval) Pre-training

**Root cause.** No phase ever trained the missing skill. Pre-training learned only
the marginal word distribution (`empty prompt → word`); SFT/RL only needed
retrieval on the ~26% of train words used as answers, so **memorization sufficed**
and the general skill never formed.

**Fix.** Add a retrieval/commit objective to **pre-training** — the phase allowed to
see *all* words — over the full 14,855-word lexicon:

- For each word `W`, simulate decent play with `W` as the answer and collect
  constraint states that narrow to **≤ `max_candidates` (≈1–3)** consistent words;
  train `(constraint-state → W)`.
- Restricting to **tight states** keeps the target unambiguous and **conflict-free
  with SFT** (which also commits at ≤2 candidates). Loose mid-game states stay
  SFT/RL's job (probe for information).
- Keep the marginal `empty prompt → word` examples for lexicon coverage; combined
  pre-training teaches both *the lexicon* and *constraint→word retrieval*.

**Why it's in-bounds (pure training, no shortcut).** It changes *what the model
learns*, not the task or the inference path — no constrained decoding, no candidate
set handed to the model, no memorize-everything. It uses pre-training as designed
("show the model all words").

**Hold-out semantics (signed off).** Training retrieval over all words means the
model *can* produce hold-out words from constraints, so the hold-out now tests
whether the **learned game strategy** (narrow + commit, trained on train answers)
generalizes to answers it never *played* — the meaningful generalization for a
game-player. Lexical generalization (producing a never-seen string) is impossible
by construction, which is exactly why pre-training sees all words.

**ADR-9 — Teach constraint→word retrieval in pre-training, on tight states only.**
*Context:* the model memorizes answers and can't commit to unseen ones (§11).
*Decision:* add `(tight constraint-state → answer)` examples over the full lexicon
to pre-training, restricted to ≤ few candidates, alongside the marginal examples.
*Rationale:* directly trains the missing skill in the phase allowed all words,
without crutches or task changes, and conflict-free with SFT. *Consequences:*
hold-out tests strategy generalization (not lexical); larger pre-train dataset; new
config knobs (`retrieval_pretrain`, `games_per_word`, `max_candidates`).

**Eval.** Headline stays **hold-out win rate** (honest eval, §5.9). Add a
**retrieval-accuracy probe**: for hold-out words, construct a 1-candidate state and
check whether the model greedily produces the word — the direct measure of the
trained skill.

**Build plan.** Extend `wordle3/data.py` (retrieval example generation),
`PretrainTrainingConfig` (knobs), and `wordle3/pretrain.py` (combined dataset);
re-run pre-train → SFT → RL; read hold-out win rate + retrieval accuracy.

### 12.1 Results (V3.1, 10M model)

The objective worked — it turned the memorizer into a generalizing solver.

| measure | V3.0 (no retrieval) | V3.1 (retrieval pre-train) |
|---------|---------------------|----------------------------|
| commit@1, hold-out (produce the answer when only it remains) | 5% | **27%** |
| commit@1, train | 11% | 38% |
| pre-train-alone hold-out win | n/a (≈0, can't play) | **~38% (≈ train — gap eliminated)** |
| **hold-out win after SFT** | **~34%** | **52%** |
| train win after SFT (honest) | ~47% | ~61% |
| train/hold-out gap | memorization regime (92% on memorized) | **~9%** |

- The commit skill rose **~5×** on hold-out (5%→27%), the mechanism that was missing.
- **Hold-out win rate 34% → 52% (+18 pts).** Pre-train alone already eliminates the
  gap (~38% hold-out ≈ train); SFT adds strategy (opener IG 4.5→6.0) that transfers
  to unseen answers instead of just memorized ones.
- Headroom remains (commit@1 only 27%) → next lever is **capacity** (38M/MoE), now
  justified: the objective is right, so capacity is the bottleneck rather than a guess.
- (RL phase-2 on top: in progress at time of writing.)

## Review Incorporation Summary

Design review verdict: **CONDITIONAL APPROVE**. Resolved in v0.2:

- **Critical (2):**
  - *C2* — `play_game_good` plays random openers at >500 candidates
    (`solver.py:99`), breaking golden quality on the 14,855 set. Added a
    pattern-matrix golden solver (§5.3, ADR-8, build Phase 3).
  - *C1* — `configure_reward` now explicitly searches the best-word denominator
    over G/U, never `train_answers`, with a guard test (§5.6, §7).
- **High (2):**
  - *H5* — clarified batched group sampling needs a new lockstep decode loop
    (`generate()` is batch-1), not just wiring (§5.7-B).
  - *H6* — `collect_completions_log_probs` is `no_grad` (`utils.py:67`); a separate
    gradient-enabled function is required for current-policy log-probs (§5.7-C).
- **Medium/Low:** corrected tokenizer token-ordering description (§2), the
  `data.py:122-127` val-split reference and dropped the fixed-"5%" claim (§1),
  noted `snapshot.json` is a plain dict not `EvalSnapshot.save()` (§5.5), added the
  split-union gate (§5.1, §7), and noted `_BEST_IG_CACHE` is effectively disabled
  at scale (§5.6). Verified-correct claims (vocab=265, UI `.get()` extra-key
  tolerance, KV-cache/log-prob line refs) left as-is.

Deferred: none — all CRITICAL/HIGH addressed. Strict hold-out mode remains OQ-3.
