# V2 Design: Dense Game State Encoding

## Overview

V2 replaces the raw game-history token format with a dense constraint-state
encoding. Instead of replaying every guess and feedback token, the model sees
a summary of what's known: confirmed positions, eliminated letters, and
positional constraints. This cuts context length roughly in half and gives
the model pre-digested information instead of requiring it to parse history.

V2 keeps character-level autoregressive decoding (not the word classifier).
The goal is to test whether the data format was the bottleneck before changing
the output architecture.

## Tokenizer

### Vocabulary (~265 tokens)

| Category | Tokens | Count | Description |
|----------|--------|-------|-------------|
| Letters | `a` - `z` | 26 | Plain letters for generating guesses |
| Unknown | `?` | 1 | Unknown position in green slots |
| Green | `a-green` - `z-green` | 26 | Confirmed letter at a position |
| Yellow | `a-yellow-1` - `z-yellow-5` | 130 | Letter exists, excluded from position N |
| Gray-0 | `a-gray-0` - `z-gray-0` | 26 | Letter not in word at all |
| Gray-1 | `a-gray-1` - `z-gray-1` | 26 | Exactly 1 of this letter (no more) |
| Gray-2 | `a-gray-2` - `z-gray-2` | 26 | Exactly 2 of this letter (no more) |
| Specials | `[bos]`, `[sep]`, `[pad]`, `[eos]` | 4 | Structural tokens |

Total: ~265

### Prompt Format

```
[bos] <green-state> [sep] <facts>
```

**Green state** (always 5 tokens): What's confirmed at each position.

```
? ? a-green ? ?
```

`?` for unknown, `X-green` for confirmed. Updated cumulatively across turns.

**Facts** (variable length): Accumulated yellow and gray constraints.

```
l-yellow-1 o-yellow-3 s-gray-0 t-gray-0 e-gray-1
```

- `l-yellow-1`: letter 'l' exists but not at position 1
- `s-gray-0`: zero s's in the word
- `e-gray-1`: exactly one 'e' (no more beyond what's green/yellow)

Facts grow as turns progress but are always a flat list — no per-turn
structure. Order within facts doesn't matter semantically.

### Example Game

Target: FOGGY. Guess sequence: SLATE → HUMOR → OVOID

**Turn 1 prompt** (no history):
```
[bos] ? ? ? ? ? [sep]
```

**Turn 2 prompt** (after SLATE → all gray):
```
[bos] ? ? ? ? ? [sep] s-gray-0 l-gray-0 a-gray-0 t-gray-0 e-gray-0
```

**Turn 3 prompt** (after SLATE, HUMOR → o yellow at pos 4):
```
[bos] ? ? ? ? ? [sep] s-gray-0 l-gray-0 a-gray-0 t-gray-0 e-gray-0 h-gray-0 u-gray-0 m-gray-0 r-gray-0 o-yellow-4
```

**Turn 4 prompt** (after SLATE, HUMOR, OVOID → o yellow at pos 0, second o gray):
```
[bos] ? ? ? ? ? [sep] s-gray-0 l-gray-0 a-gray-0 t-gray-0 e-gray-0 h-gray-0 u-gray-0 m-gray-0 r-gray-0 o-yellow-4 o-yellow-1 i-gray-0 d-gray-0 o-gray-1
```

Note: `o-yellow-4` (from HUMOR), `o-yellow-1` (from OVOID pos 0), and
`o-gray-1` (exactly one 'o', from the second 'o' in OVOID being gray).
The model sees all three facts and can deduce: one 'o', not at positions
1 or 4.

### Context Length Comparison

| Turn | V1 (raw history) | V2 (constraint state) |
|------|------------------|-----------------------|
| 1 | 1 token | 7 tokens |
| 2 | 12 tokens | ~12 tokens |
| 3 | 23 tokens | ~17 tokens |
| 4 | 34 tokens | ~22 tokens |
| 5 | 45 tokens | ~27 tokens |
| 6 | 56 tokens | ~32 tokens |

V2 is comparable at turn 2 and shorter at turns 3-6. More importantly,
the information density per token is much higher — every token carries a
constraint, not raw history.

## Training Data

### Reuse from V1

Game transcript generation (`mm_wordle.transcripts`) is unchanged. The same
solver mix (30% random, 40% decent, 30% good) produces game transcripts.
The difference is how transcripts become training examples.

**V1**: Each example is `(raw game state tokens, next 5 chars)`.
**V2**: Each example is `(constraint state summary, next 5 chars)`.

A new function converts a `GameState` into the V2 constraint-state prompt
by accumulating greens, yellows, and grays from all previous turns.

### Turn Distribution

Keep the late-turn oversampling from V1. The distribution weights:

| Turn | Weight | Share |
|------|--------|-------|
| 1 | 1x | ~7% |
| 2 | 2x | ~13% |
| 3 | 3x | ~20% |
| 4 | 4x | ~23% |
| 5 | 5x | ~20% |
| 6 | 6x | ~17% |

## Model Architecture

### What stays the same

- Decoder-only transformer with causal attention
- RoPE positional encoding
- Pre-training on game transcripts with cross-entropy loss
- Loss masked to only the 5 target character positions
- Character-level autoregressive generation (5 tokens per guess)

### What changes

- `token_emb`: `nn.Embedding(265, embed_dim)` instead of `nn.Embedding(50, embed_dim)`
- `lm_head`: `nn.Linear(embed_dim, 265)` — outputs over full V2 vocab, but at
  inference we mask to only the 26 plain letter tokens

The model generates 5 plain letter tokens autoregressively, same as V1. The
larger vocab is only for the input encoding. At generation time, logits for
non-letter tokens are masked to `-inf`.

### Smallest Model Estimate

**What the model needs to learn:**

1. Read 5 green-state tokens (fixed structure)
2. Read ~10-20 fact tokens (variable, unordered constraints)
3. Attend to constraints while generating a valid 5-letter word
4. Ideally, generate a word that maximizes information gain or solves

**Capacity analysis:**

- Embedding table: 265 × embed_dim. At embed_dim=128: 34K params.
- The constraint state is simpler than raw history — no need to learn
  feedback-to-letter correspondence (it's baked into the tokens).
- The main challenge is learning which 5-letter combinations are valid
  words AND satisfy the constraints. This is ~2,315 valid patterns.

**Napkin math for minimum model size:**

The model needs to store ~2,315 word patterns and learn to select based on
constraints. A lookup table for this would be 2,315 × 5 × 26 ≈ 300K values.
A transformer compresses this through learned attention patterns.

| Config | Params | Notes |
|--------|--------|-------|
| 2 layers, 4 heads, 64 dim | ~200K | Probably too small — can't hold word list |
| 4 layers, 4 heads, 128 dim | ~1.5M | Might work for basic play |
| 6 layers, 8 heads, 256 dim | ~5M | Known to work from V1 |

Recommendation: start at **4 layers, 4 heads, 128 dim (~1.5M params)**.
If that doesn't converge, step up to 6 layers. The denser encoding should
need less model capacity than V1 because the model doesn't have to learn
game-state parsing.

## Training Pipeline

### Step 1: Build V2 Tokenizer

New tokenizer class alongside `CharTokenizer`. Handles encoding constraint
states and decoding generated letters.

### Step 2: Build V2 Data Pipeline

New function that converts game transcripts into V2 training examples.
Reuses `generate_examples()` for transcript generation, adds a conversion
layer that computes constraint state from game history.

### Step 3: Pre-train

Same flow as V1: cross-entropy loss on the 5 target characters.

**Primary eval metric: valid word rate per turn.** The goal of pretraining
is 100% valid word generation at all turns (1-6). Win rate is reported
but is secondary — it's a side effect of valid word generation, not the
training objective. RL handles strategy.

Eval plays 200 games and reports valid word rate at each turn. Training
is not done until late-turn valid word rate is near 100%.

Start with the 5M model (6 layers, 8 heads, 256 embed dim). The 856K
model (4 layers, 128 dim) achieved 65% win rate but produced invalid
words at later turns — insufficient capacity for the word list.

### Step 4: RL Phase 1 (Openers, turns 1-2)

GRPO with info gain reward (composite=False). Goal: the model chooses
openers with ≥5.5 bits info gain 100% of the time for turns 1-2 without
regressing on valid word rate.

### Step 5: RL Phase 2 (Mid/Late game, turns 3-6)

GRPO with composite reward (normalized info gain + endgame bonus +
solve bonus + invalid word penalty). Opener model frozen from Phase 1.

Invalid words receive -10.0 penalty in both phases. See
`docs/reward-function.md` for the full reward specification.

## Eval Metrics

All three metrics reported at every eval across all phases:

1. **Valid word rate by turn** — primary for pretraining. Goal: 100% at all turns.
2. **Info gain by turn** — primary for Phase 1. Good opener threshold: ≥5.5 bits.
3. **Win rate** — single-sample games (200 games, temp=0.1). Primary for Phase 2.

A "good opener" is defined as ≥5.5 bits expected info gain. This is the
top ~50 words out of 2315 (top 2%). The best openers (raise, slate, crate)
score 5.8-5.9 bits.

The key question: does the dense encoding help the model learn mid-game
strategy better than V1? If the model can read constraints directly
from the tokens, it should learn feedback consistency (guessing words
that match constraints) much faster.

## Project Structure

V2 code lives alongside V1. No V1 code is modified or deleted.

### New files

```
wordle2/                          # V2 exercise directory
  README.md
  config.py                       # V2 configs (may extend V1)
  tokenizer.py                    # V2 constraint-state tokenizer
  data.py                         # V2 data pipeline
  pretrain.py                     # V2 pre-training script
  finetune.py                     # V2 RL fine-tuning script
  configs/
    small.yaml                    # 1.5M model config
    finetune-phase1.yaml
    finetune-phase2.yaml
```

### Shared libraries (no changes needed)

- `libs/mm-model/` — GPT model with RoPE (works with any vocab size)
- `libs/mm-grpo/` — GRPO algorithm (works with any log prob shape)
- `libs/mm-training/` — optimizer, scheduler, checkpointing
- `libs/mm-wordle/` — game environment, reward function, word lists, solvers

### Shared with minor additions

- `libs/mm-wordle/` — may need a new `constraint_state.py` module that
  computes the accumulated constraint state from a `GameState`. This is
  pure game logic, not V2-specific, and could be useful for V1 analysis too.

## PyTorch Primitives

Areas where we should use PyTorch built-ins instead of custom code:

- **`nn.TransformerDecoderLayer`**: PyTorch provides pre-built transformer
  blocks with attention, feed-forward, and layer norms. Evaluate whether
  switching from our custom `TransformerBlock` reduces code and improves
  performance. Our custom block exists because we needed KV cache support,
  but PyTorch's `nn.MultiheadAttention` now supports this.

- **`torch.compile`**: Enable for the forward pass on supported backends.
  Can significantly speed up training without code changes.

- **Mixed precision (`torch.amp`)**: Use automatic mixed precision for
  faster training on MPS/CUDA. The model is small but the data throughput
  would increase.

- **`torch.nn.utils.parametrize`**: Consider for weight tying between
  token embeddings and the output head if we want to share letter
  representations.

## Design Decisions

- **Fact ordering**: sorted alphabetically. The model doesn't care about turns —
  canonical ordering means the same constraint state always produces the same
  token sequence regardless of which turn discovered the fact.
- **Previous guesses**: not included. The constraint state captures all information.
- **Constrained decoding**: not included in V2. Test the format change in isolation.
