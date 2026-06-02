# Architecture: Wordle Exercise

This document describes the technical architecture for the Wordle training exercise. It covers model design, tokenization, pre-training, RL fine-tuning, the Wordle environment, observability, and how the code is organized into libraries.

See [docs/prd.md](prd.md) for goals and success criteria.

## Overview

The exercise has two training phases and produces two models:

1. **Pre-train** a small GPT from scratch on English text to learn language structure.
2. **RL fine-tune** the pre-trained model with GRPO against a Wordle environment to learn the game.

Everything runs on Apple Silicon (MacBook Pro / Mac Studio) using PyTorch with the MPS backend.

## Model Architecture

Decoder-only GPT-style transformer. Two size configs, both character-level:

| Config | Layers | Heads | Embed Dim | Context | Params | Train Time (MPS) |
|--------|--------|-------|-----------|---------|--------|-------------------|
| small  | 6      | 8     | 256       | 256     | ~5M    | ~10-30 min        |
| medium | 6      | 6     | 384       | 256     | ~10M   | ~1-3 hrs          |

Start with `small` for fast iteration. Scale to `medium` if the model needs more capacity for Wordle. Whether 5M params is sufficient for Wordle is an open question — no one has published results at this scale. Finding out is part of the exercise.

Components:
- Token embedding + learned positional embedding
- N transformer blocks (pre-norm: LayerNorm → attention → residual → LayerNorm → MLP → residual)
- Final LayerNorm → linear head projecting to vocab size

The context window of 256 tokens is more than enough. A full Wordle game is at most ~90 tokens (6 guesses × 5 letters × ~3 tokens per letter-feedback pair).

## Tokenization

Two tokenizers, built as a library exercise:

### Character-level tokenizer (primary)

Used for Wordle pre-training and RL. Vocab of ~50 tokens:

| Token Type | Examples | Count |
|------------|----------|-------|
| Letters | `a`-`z` | 26 |
| Feedback | `[green]`, `[yellow]`, `[gray]` | 3 |
| Structural | `[bos]`, `[eos]`, `[pad]`, `[sep]`, `[newline]` | 5 |
| Reserved | spare slots for future use | ~16 |
| **Total** | | **~50** |

This is a lookup table, not a learned tokenizer. ~20 lines of code.

### BPE tokenizer (learning exercise)

A from-scratch BPE implementation as a separate library, to understand how subword tokenization works. Not used for Wordle — built for learning. Trained on a small text corpus, produces a configurable vocab size.

## Pre-Training

### Data

Character-level pre-training on a mix of datasets from HuggingFace:

| Dataset | Source | Purpose |
|---------|--------|---------|
| TinyStories | `roneneldan/TinyStories` (HuggingFace) | Broad English letter patterns and word structure. Use a subsample if full set is too slow. |
| Wordle word lists | Bundled in `mm-wordle` package data | Dense exposure to valid 5-letter words. ~2,300 answers + ~13,000 valid guesses. Repeated many times to compensate for small size. |

The data pipeline converts text to character-level token sequences, chunks into fixed-length blocks (context window size), and produces (input, target) pairs with causal shift.

### Training Loop

Standard autoregressive language model training:

- **Loss:** cross-entropy on next-character prediction
- **Optimizer:** AdamW
- **Schedule:** linear warmup + cosine decay
- **Batch size:** 64 (adjust to fit memory)
- **Precision:** fp32 (fp16/bf16 offers minimal benefit at this scale on MPS)
- **Device:** MPS with CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`)

Metrics to log: training loss, validation loss, learning rate, tokens/sec.

Checkpoints saved periodically. The final checkpoint is the input to Phase 2.

### Evaluation

- Validation loss on held-out TinyStories split
- Sample text generation (qualitative — does it produce English-like character sequences?)
- Valid-word rate: generate 5-character sequences, check what fraction are real English words

## RL Fine-Tuning with GRPO

### GRPO Algorithm

Implemented from scratch as a library. No HuggingFace trl dependency.

This is an **on-policy** algorithm — each training step samples fresh completions from the current policy, uses them once, then discards them. On-policy is simpler to implement and debug, making it the right starting point. Off-policy methods (replay buffers, importance sampling) are a future exercise (see Future Direction).

GRPO (Group Relative Policy Optimization) works by:

1. For each prompt (game state), sample a **group** of completions (candidate guesses).
2. Score each completion with a reward function.
3. Compute advantages by normalizing rewards within the group (subtract mean, divide by std).
4. Update the policy with a clipped surrogate objective (same as PPO) + KL penalty against a reference policy.

No critic/value network needed — advantages come from within-group comparison.

The implementation must emit structured per-step data (game state, all completions, rewards, advantages, probability changes) so the visualization layer can reconstruct any training step for inspection.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Group size | 8-16 | Number of candidate guesses per game state |
| KL coefficient (beta) | 0.01-0.04 | Prevents divergence from pre-trained model |
| Clip epsilon | 0.1-0.2 | Standard PPO clipping |
| Learning rate | 1e-5 to 5e-5 | Lower than pre-training |
| Batch size (prompts) | 16-32 | Number of game states per batch |

### Reward Function

The reward function scores a single guess given the game state. Rewards are shaped to guide learning:

| Signal | Reward | Rationale |
|--------|--------|-----------|
| Invalid word (not in word list) | -1.0 | Must learn to guess real words (unconstrained mode only) |
| Repeated guess | -0.5 | Must not waste turns on words already tried |
| Contradicts known clues | -0.3 | Must not use letters known to be gray, or ignore known green positions |
| Valid word, no new information | 0.0 | Baseline — didn't help |
| Each new green letter | +0.2 | Correct letter in correct position |
| Each new yellow letter | +0.1 | Correct letter, wrong position |
| Solved the puzzle | +1.0 | Bonus for winning |
| Failed after 6 guesses | -0.5 | Penalty for losing |

These are starting values — expect to tune them during training. Reward function iteration (observing reward hacking, adjusting signals) is an explicit part of the exercise.

### Decoding Modes

Two approaches to generating guesses, explored as a learning exercise:

**Trie-constrained decoding.** At each of the 5 character positions, the model's logits are masked using a prefix trie built from the answer word list (~2,315 words). Only characters that continue a valid word are allowed. This guarantees every guess is a valid word with just 5 forward passes, regardless of vocabulary size.

The trie mask is applied consistently in all contexts:
- **Sampling:** generating candidate guesses during GRPO
- **Old policy log probs:** computed at sampling time, frozen during optimization
- **Current policy log probs:** recomputed each PPO epoch
- **Reference policy log probs:** from the frozen reference model
- **Evaluation:** greedy trie-constrained decoding

This consistency ensures the GRPO importance sampling ratio is computed over the same constrained distribution everywhere. See `docs/game-format.md` for the full specification.

### REINFORCE Baseline

Before implementing GRPO, implement REINFORCE with baseline (~50 lines of PyTorch) as the first RL algorithm. This teaches core policy gradient concepts (log-probability weighting, reward baselines, variance reduction) with minimal complexity. Getting REINFORCE working first provides:

- A known-good baseline to compare GRPO against
- A simpler system to debug when things go wrong
- The pedagogical progression: why does GRPO improve on this?

### Training Flow

1. Load pre-trained model as both the **active policy** and the **reference policy** (frozen copy).
2. For each training step:
   a. Sample a batch of Wordle games (random target words).
   b. For each game, play up to 6 turns. At each turn, the model sees the game state (prior guesses + feedback) and generates a group of candidate guesses via trie-constrained decoding.
   c. Score each candidate with the reward function.
   d. Compute GRPO loss and update the active policy.
3. Log: win rate, average guesses to solve, reward distribution, KL divergence from reference.

### Curriculum Learning (future)
3. **Stage 3:** Full games from Turn 1. Full word list.

This progression is critical — prior art shows even 4B-parameter models fail without curriculum learning.

## Wordle Environment

A stateful game environment following a simple API (not full Gymnasium — too heavyweight for this):

```python
class WordleEnv:
    def reset(self, target_word: str | None = None) -> GameState: ...
    def step(self, guess: str) -> tuple[GameState, float, bool]: ...
```

- `reset()` starts a new game. If no target word given, picks one at random from the word list.
- `step()` accepts a 5-letter guess, returns updated game state, reward, and done flag.
- `GameState` contains: target word (hidden until done), guesses so far, feedback per guess (green/yellow/gray per letter), turn number.

The environment also provides:
- `render()` for human-readable display of the game board.
- `to_tokens()` to serialize the game state as a token sequence for the model.
- The official Wordle word lists (answers + valid guesses).

## Observability and Visualization

Visibility into training is a first-class concern, not an afterthought. Two layers:

### Layer 1: Training Metrics (TensorBoard)

Standard training telemetry logged via PyTorch's `torch.utils.tensorboard.SummaryWriter`. Runs locally, no accounts needed.

**Pre-training metrics:**

| Metric | Type | Why |
|--------|------|-----|
| Train/val loss per step | Scalar | Core learning signal — is the model learning? |
| Learning rate | Scalar | Verify schedule is working |
| Tokens/sec | Scalar | Performance tracking |
| Gradient norm | Scalar | Detect exploding/vanishing gradients |
| Weight distributions per layer | Histogram | Watch for dead neurons, saturation |
| Embedding norms | Histogram | Are embeddings learning? Collapsing? |
| Sample generations | Text | Qualitative — what does the model produce at each checkpoint? |

**RL fine-tuning metrics:**

| Metric | Type | Why |
|--------|------|-----|
| Win rate (rolling) | Scalar | Is the model learning to play? |
| Average guesses to solve | Scalar | Is it getting more efficient? |
| Reward distribution | Histogram | Reward signal shape over time |
| KL divergence from reference | Scalar | Is the policy drifting too far? |
| Valid word rate | Scalar | Is the model guessing real words? |
| Policy loss, entropy | Scalar | GRPO training dynamics |

View with: `tensorboard --logdir runs/`

### Layer 2: Custom Visualizations (mm-viz)

Domain-specific visualizations that TensorBoard can't do. Built with matplotlib/plotly, rendered as HTML reports or interactive notebooks.

**Wordle game replays:**
- At each checkpoint, play N evaluation games and render the full game board (colored tiles, like the real Wordle UI).
- Side-by-side comparison: how the same target word is played at checkpoint 100 vs 1000 vs 5000.
- Serialize replays so they can be viewed later without re-running the model.

**Attention visualizations:**
- Heatmaps showing what the model attends to when making a guess.
- Key question: does it attend to feedback tokens (green/yellow/gray)? Does this emerge during RL?
- Per-layer, per-head attention patterns on a single Wordle turn.

**Strategy evolution:**
- Track the model's first-guess distribution over training. Does it converge on high-information opening words?
- Letter frequency analysis: what letters does the model favor at each stage of training?
- Elimination efficiency: how many candidate words does each guess rule out?

**GRPO step inspector:**
- Drill into a single GRPO training step and see the full algorithm pipeline:
  1. **Prompt/game state** — the Wordle board the model is looking at (guesses so far + feedback).
  2. **Group of completions** — all N candidate guesses the model sampled, with their token-level log-probabilities.
  3. **Reward scoring** — how each completion was scored (breakdown: valid word? green letters? yellow? solved?).
  4. **Advantage computation** — the raw rewards, group mean/std, and the normalized advantages. Which completions are above/below average and by how much.
  5. **Policy update** — the probability shift: for each completion, what was the old probability vs new probability after the gradient step? Which guesses got reinforced, which got suppressed?
  6. **KL penalty** — how much the update was regularized toward the reference policy.
- Render as a single scrollable view for one step, or animate across steps to show how the policy evolves.
- This is the "GRPO explained with your own model's data" view — makes the algorithm concrete.

**Checkpoint comparison dashboard:**
- Pick any two checkpoints, run the same set of evaluation games, and compare results side-by-side.
- Win rate, guess distribution, strategy differences.

### Checkpointing

Checkpoints are the backbone of reproducibility and visualization. Save frequently enough to tell a story.

| What | When | Contents |
|------|------|----------|
| Pre-training checkpoint | Every N steps + end of training | Model weights, optimizer state, step number, config, RNG states |
| RL checkpoint | Every N training steps + end | Policy weights, reference weights, optimizer state, step, config, RNG states |
| Evaluation snapshot | At each checkpoint | Game replays, metrics, attention data for a fixed set of target words |

Checkpoints go in a `runs/` directory (gitignored), organized by experiment name and timestamp. A checkpoint loader can resume training from any checkpoint or load a model for evaluation.

**Fixed evaluation set:** A set of ~50 target words used consistently across all checkpoints. This makes progress visible — you can watch the model's performance on the same words evolve over training.

## Library Boundaries

Reusable pieces extracted into library packages within the repo:

| Library | What It Contains | Used By |
|---------|-----------------|---------|
| `mm-tokenizers` | Character-level tokenizer, BPE tokenizer | Pre-training, RL, evaluation |
| `mm-model` | GPT model definition, config, weight loading/saving | Pre-training, RL, evaluation |
| `mm-training` | Training loop utilities (optimizer setup, LR schedule, checkpointing, metrics logging, TensorBoard integration) | Pre-training, RL |
| `mm-grpo` | GRPO algorithm implementation | RL fine-tuning |
| `mm-wordle` | Wordle environment, word lists, reward function, game state tokenization | RL fine-tuning, evaluation |
| `mm-viz` | Custom visualizations: game replays, attention heatmaps, strategy analysis, checkpoint comparison | Evaluation, analysis |

Each library is a Python package in the repo. The exercise scripts import from them.

## Repo Structure

```
mm/
├── AGENTS.md
├── pyproject.toml              # workspace root (uv workspace)
├── docs/
│   ├── prd.md
│   └── architecture.md
├── libs/
│   ├── mm-tokenizers/
│   │   ├── pyproject.toml
│   │   ├── src/mm_tokenizers/
│   │   └── tests/
│   ├── mm-model/
│   │   ├── pyproject.toml
│   │   ├── src/mm_model/
│   │   └── tests/
│   ├── mm-training/
│   │   ├── pyproject.toml
│   │   ├── src/mm_training/
│   │   └── tests/
│   ├── mm-grpo/
│   │   ├── pyproject.toml
│   │   ├── src/mm_grpo/
│   │   └── tests/
│   ├── mm-wordle/
│   │   ├── pyproject.toml
│   │   ├── src/mm_wordle/
│   │   └── tests/
│   └── mm-viz/
│       ├── pyproject.toml
│       ├── src/mm_viz/
│       └── tests/
├── wordle/
│   ├── README.md
│   ├── pretrain.py             # pre-training script
│   ├── finetune.py             # GRPO RL fine-tuning script
│   ├── evaluate.py             # evaluation and demo script
│   └── configs/
│       ├── small.yaml           # ~5M param config (6L/8H/256d)
│       └── medium.yaml          # ~10M param config (6L/6H/384d)
└── .agents/
    └── ...
```

Uses uv workspaces so libraries can depend on each other and the exercise scripts can import them all.

## Data Management

Raw datasets are downloaded and cached by the HuggingFace `datasets` library (`~/.cache/huggingface/`). No project-local data directory — HF handles caching, deduplication, and versioning across projects.

The data pipeline in each training script:
1. Load dataset from HF Hub (cached after first download).
2. Tokenize and chunk on the fly (or cache the processed version via HF's `.map()` with caching).
3. Feed to DataLoader.

No large files in the repo. Word lists (~13K words) are small enough to bundle in `mm-wordle` as package data.

## Reproducibility

Config-level reproducibility: the same config file should produce very similar (not bit-identical) results. Not targeting deterministic mode — it restricts MPS performance and isn't worth it at this scale.

### What makes a run reproducible

Each training run produces a **run manifest** saved alongside its checkpoints:

| Field | What | Why |
|-------|------|-----|
| Config file | Full model + training hyperparameters | Defines the run |
| Seed | Random seed used for torch, numpy, python | RNG reproducibility |
| Dataset ID + revision | HuggingFace dataset path and commit hash | Pins the exact data version |
| Package versions | torch, datasets, and mm-* library versions | Pins the code |
| Git commit | Repo commit hash at time of run | Pins the training scripts |
| Hardware | Device type, OS, chip | Context for performance |

The manifest is written automatically at the start of each training run by `mm-training`. It's a JSON file in the run directory next to the checkpoints and TensorBoard logs.

### Run directory structure

```
runs/
└── pretrain-small-2026-06-01T14:30:00/
    ├── manifest.json          # everything needed to understand this run
    ├── events.out.tfevents.*  # TensorBoard logs
    ├── checkpoint-1000/       # periodic checkpoint
    │   ├── model.pt
    │   ├── optimizer.pt
    │   └── rng_states.pt
    ├── checkpoint-5000/
    └── eval/                  # evaluation snapshots
        ├── step-1000.json
        └── step-5000.json
```

`runs/` is gitignored. Checkpoints are local artifacts, not committed.

## Hardware Considerations

- **MPS backend:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for any unsupported ops. All core transformer ops (matmul, softmax, LayerNorm, GELU, attention) are MPS-supported.
- **Precision:** fp32. Mixed precision offers negligible benefit at this scale on Apple Silicon.
- **Memory:** Both model configs fit comfortably. ~5M params needs ~200MB, ~10M needs ~400MB for training (params + optimizer + activations). No memory pressure on any Apple Silicon Mac.
- **CPU fallback:** Everything should also work on CPU (slower). Useful for CI or machines without MPS.
- **No `torch.compile()`:** Limited MPS support. Not needed at this scale.
