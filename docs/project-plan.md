# Project Plan: Wordle Exercise

This plan breaks the [architecture](architecture.md) into concrete tasks organized for parallel execution. Each phase gate ensures the next phase has a solid foundation.

See [docs/prd.md](prd.md) for goals. See [docs/architecture.md](architecture.md) for technical design.

## Dependency Graph

```
Phase 0: Repo Setup
    │
    ▼
Phase 1: Foundation Libraries ──────────────────────────────
    │              │              │              │           │
    ▼              ▼              ▼              ▼           ▼
mm-tokenizers  mm-model      mm-wordle      mm-viz       mm-training
(char-level)   (GPT def)     (env + words)  (framework)  (loop utils)
    │              │              │              │           │
    └──────────────┴──────┬───────┘              │           │
                          │                      │           │
                          ▼                      │           │
                   Phase 2: Pre-Training ────────┤           │
                   (data + pretrain.py)          │           │
                          │                      │           │
                          ▼                      │           │
                   Phase 2.5: Pre-Training       │           │
                   Validation & Experiments      │           │
                          │                      │           │
                          │    ┌─────────────────┘           │
                          ▼    ▼                             │
                   Phase 3: RL Fine-Tuning ──────────────────┘
                   (REINFORCE → GRPO, constrained → unconstrained)
                          │
                          ▼
                   Phase 3.5: RL Experiments
                   (reward tuning, curriculum, ablations)
                          │
                          ▼
                   Phase 4: Evaluation & Viz
                   (evaluate.py + dashboards)
                          │
                          ▼
                   Phase 5: BPE Tokenizer
                   (learning exercise, independent)
```

## Phase 0: Repo Setup

**Goal:** Working uv workspace with all packages scaffolded, linting configured, gitignore in place.

One task, done first since everything depends on it.

| # | Task | Deliverable |
|---|------|-------------|
| 0.1 | Initialize uv workspace | Root `pyproject.toml` with workspace config, ruff settings, common dev dependencies (pytest, tensorboard) |
| 0.2 | Scaffold all library packages | `libs/mm-{tokenizers,model,training,wordle,grpo,viz}/` each with `pyproject.toml`, empty `src/` and `tests/` |
| 0.3 | Scaffold wordle exercise | `wordle/` directory with empty scripts and config dir |
| 0.4 | Repo hygiene | `.gitignore` (runs/, checkpoints, __pycache__, .venv), `uv sync` works, `uv run ruff check .` passes |

**Gate:** `uv sync && uv run ruff check . && uv run pytest` all pass (on empty packages).

---

## Phase 1: Foundation Libraries

**Goal:** The core building blocks, each independently testable. All five can be built in parallel.

**Prerequisite:** Before starting parallel work, define interface contracts (type signatures and data shapes) for all cross-library boundaries: model forward/generate signatures, tokenizer encode/decode signatures, reward function signature, step data schema. This prevents integration pain later.

### 1A: mm-tokenizers (character-level)

| # | Task | Deliverable |
|---|------|-------------|
| 1A.1 | Character-level tokenizer | `CharTokenizer` class: encode/decode strings, vocab property, special token IDs (bos, eos, pad, sep, newline), feedback tokens (green, yellow, gray) |
| 1A.2 | Tests | Round-trip encode/decode, special tokens, edge cases (empty string, unknown chars) |

### 1B: mm-model

| # | Task | Deliverable |
|---|------|-------------|
| 1B.1 | Model config | `GPTConfig` dataclass: n_layers, n_heads, embed_dim, vocab_size, context_len, dropout. Factory methods for `small` (~5M, 256 dim) and `medium` (~10M, 384 dim) presets |
| 1B.2 | GPT model | `GPT` nn.Module: token + position embeddings, transformer blocks (pre-norm), LM head. `forward()` returns logits. `generate()` for autoregressive sampling with temperature/top-k |
| 1B.3 | Constrained decoding | `score_words()` method: given a game state and a list of valid words, return a probability distribution over the word list by scoring each word's character sequence |
| 1B.4 | Checkpointing | `save_checkpoint()` / `load_checkpoint()`: model weights, optimizer state, step, config, RNG states (torch, python, numpy). Resume-from-checkpoint support |
| 1B.5 | Tests | Forward pass shape check, generate produces valid token IDs, score_words returns valid distribution, checkpoint save/load round-trip, param count matches expected for small/medium configs |

### 1C: mm-wordle

| # | Task | Deliverable |
|---|------|-------------|
| 1C.1 | Word lists | Bundle the official Wordle answer list (~2,300 words) and valid guess list (~13,000 words) as package data |
| 1C.2 | Game environment | `WordleEnv` with `reset()` / `step()` API. `GameState` dataclass with guesses, feedback (green/yellow/gray per letter), turn number |
| 1C.3 | Reward function | Configurable reward function: invalid word penalty, repeated guess penalty, contradicts-clues penalty, green/yellow bonuses, win bonus, loss penalty. Default values from architecture doc |
| 1C.4 | Game state serialization | `to_tokens()` method: serialize game state as token sequence (letter + feedback interleaved). `render()` for human-readable board display |
| 1C.5 | Tests | Correct feedback for known guess/target pairs, reward scoring (including repeated/contradictory guess penalties), serialization round-trip, render output, word list loading |

### 1D: mm-training

| # | Task | Deliverable |
|---|------|-------------|
| 1D.1 | Device setup | Auto-detect MPS/CUDA/CPU, configure device, set fallback env vars |
| 1D.2 | Optimizer + schedule | `create_optimizer()` with AdamW defaults, `create_scheduler()` with linear warmup + cosine decay |
| 1D.3 | TensorBoard logging | `MetricsLogger` wrapping SummaryWriter: scalar, histogram, text logging. Consistent run directory structure (`runs/{experiment}/{timestamp}/`) |
| 1D.4 | Training loop utilities | Gradient clipping, gradient norm computation, ETA estimation, progress display |
| 1D.5 | Run manifest | `RunManifest` dataclass + `write_manifest()`: captures config, seed, dataset ID + HF revision, package versions, git commit, hardware info. Written as JSON at start of each run |
| 1D.6 | Tests | Optimizer creates without error, scheduler produces expected LR curve, metrics logger writes valid TensorBoard events, manifest round-trips through JSON |

### 1E: mm-viz (framework)

| # | Task | Deliverable |
|---|------|-------------|
| 1E.1 | Data models | Dataclasses for structured training data: `GRPOStepData` (game state, completions, rewards, advantages, prob changes), `GameReplay` (full game with board state at each turn), `EvalSnapshot` (metrics + replays for a checkpoint) |
| 1E.2 | Wordle board renderer | Render a Wordle game board as colored HTML (green/yellow/gray tiles). Single game and side-by-side comparison layouts |
| 1E.3 | Tests | Renderer produces valid HTML, data models serialize/deserialize |

**Gate:** All libraries pass their tests. `uv run pytest` from root runs everything.

---

## Phase 2: Pre-Training

**Goal:** Train a small GPT on character-level English text. Produces a checkpoint for RL fine-tuning.

Depends on: mm-tokenizers, mm-model, mm-training.

| # | Task | Deliverable |
|---|------|-------------|
| 2.1 | Data pipeline | Load TinyStories via HuggingFace `datasets` (cached in `~/.cache/huggingface/`). Load Wordle word lists from `mm-wordle` package data. Character-level tokenization via `.map()` with caching. Train/val split. Chunking into fixed-length blocks. PyTorch Dataset + DataLoader |
| 2.2 | Pre-training script | `wordle/pretrain.py`: load config (small/medium YAML), build model, train with cross-entropy loss. TensorBoard logging of all pre-training metrics from architecture doc. Periodic checkpointing. Resumable. Writes run manifest at start |
| 2.3 | Model configs | `wordle/configs/small.yaml` (~5M, 6L/6H/256d) and `medium.yaml` (~10M, 6L/6H/384d) with model architecture + training hyperparameters |
| 2.4 | Pre-training evaluation | Validation loss, sample text generation at each checkpoint, valid-word-rate metric |
| 2.5 | Run pre-training (small) | Train the small model end-to-end, verify loss decreases, inspect TensorBoard, save final checkpoint |

**Gate:** Small model trains to completion. Loss curve shows learning. Generated text is English-like. Checkpoint loads and generates text.

---

## Phase 2.5: Pre-Training Validation

**Goal:** Evaluate the pre-trained model before RL. Establish baselines. This is where you build intuition for what the model learned.

| # | Task | Deliverable |
|---|------|-------------|
| 2.5.1 | Zero-shot Wordle evaluation | Have the pre-trained model attempt Wordle (both constrained and unconstrained). It will fail — this is the baseline that makes RL improvement visible |
| 2.5.2 | Valid-word-rate analysis | In unconstrained mode, what fraction of generated 5-character sequences are valid English words? This tells you how hard the RL spelling problem will be |
| 2.5.3 | Pre-training ablations (optional) | Try different data mixes (more/fewer word list repetitions), different model sizes. What helps the valid-word rate most? |

**Gate:** Baseline metrics established. You can articulate what the pre-trained model knows and doesn't know.

---

## Phase 3: RL Fine-Tuning

**Goal:** Implement REINFORCE then GRPO. Start with constrained decoding, then try unconstrained.

Depends on: Phase 2 checkpoint, mm-wordle, mm-training, mm-viz data models.

| # | Task | Deliverable |
|---|------|-------------|
| 3.1 | REINFORCE baseline | Implement REINFORCE with baseline (~50 lines) in mm-grpo. Constrained decoding mode. Train on Wordle, verify win rate improves |
| 3.2 | mm-grpo core | GRPO algorithm: group sampling, advantage normalization, clipped surrogate loss, KL penalty against reference policy. Emits `GRPOStepData` for each training step |
| 3.3 | mm-grpo tests | Advantage computation correctness, loss computation with known inputs, KL penalty math, step data emission |
| 3.4 | Fine-tuning script (constrained) | `wordle/finetune.py`: load pre-trained checkpoint as policy + reference, run GRPO training loop against WordleEnv with constrained decoding. TensorBoard logging of RL metrics. Periodic checkpointing with eval snapshots |
| 3.5 | Run constrained GRPO (small) | Train end-to-end, verify win rate improves over REINFORCE baseline, inspect TensorBoard |
| 3.6 | Unconstrained mode | Add unconstrained character-by-character decoding to finetune.py. Add curriculum learning support (staged word lists, mid-game starts) |
| 3.7 | Run unconstrained GRPO (small) | Train with curriculum, compare results to constrained mode |

**Gate:** Constrained GRPO shows clear improvement over both the pre-trained baseline and REINFORCE. Comparison between constrained and unconstrained is documented.

---

## Phase 3.5: RL Experiments

**Goal:** This is where the real learning happens. Iterate on the RL setup, observe failure modes, and fix them.

| # | Task | Deliverable |
|---|------|-------------|
| 3.5.1 | Reward function iteration | Observe reward hacking (model exploiting loopholes). Adjust reward signals. Document what happened and why |
| 3.5.2 | Hyperparameter exploration | Vary group size, KL coefficient, clip epsilon. Compare learning curves. What matters most? |
| 3.5.3 | GRPO ablations | Remove components one at a time (KL penalty, advantage normalization, clipping) and observe the effect. Understand why each piece exists |
| 3.5.4 | Entropy monitoring | Track token entropy and response diversity. Observe and diagnose entropy collapse if it occurs |
| 3.5.5 | Scale up (optional) | Try medium config (~10M) if small plateaus. Compare learning curves |

**Gate:** You can explain why the reward function works (or doesn't), what each GRPO component contributes, and where the model struggles.

---

## Phase 4: Evaluation & Visualization

**Goal:** Full evaluation pipeline and custom visualizations.

Depends on: Phase 3 checkpoints, mm-viz.

| # | Task | Deliverable |
|---|------|-------------|
| 4.1 | Evaluation script | `wordle/evaluate.py`: load any checkpoint, play the fixed evaluation set, report win rate / avg guesses / strategy stats. Optionally run in interactive mode (play Wordle against the model in the terminal) |
| 4.2 | Game replay visualization | Render game replays as HTML. Side-by-side checkpoint comparison for the same target word |
| 4.3 | Attention visualization | Extract and render attention heatmaps for a Wordle turn. Per-layer, per-head views |
| 4.4 | GRPO step inspector | Load a saved `GRPOStepData`, render the full pipeline: game state → completions → rewards → advantages → probability shifts. Scrollable HTML view |
| 4.5 | Strategy evolution | Plot first-guess distribution, letter preferences, elimination efficiency across checkpoints |
| 4.6 | Checkpoint comparison dashboard | Pick two checkpoints, run eval set, render side-by-side comparison report |

**Gate:** All visualizations render. Can inspect any checkpoint. GRPO step inspector tells the story of a single training step.

---

## Phase 5: BPE Tokenizer (Learning Exercise)

**Goal:** Understand subword tokenization by building BPE from scratch.

Independent of everything else — can be done anytime after Phase 0.

| # | Task | Deliverable |
|---|------|-------------|
| 5.1 | BPE training | `BPETokenizer.train(corpus, vocab_size)`: byte-pair encoding merge loop, build merge table |
| 5.2 | BPE encode/decode | Encode text to token IDs using learned merges, decode back. Handle unknown bytes |
| 5.3 | Tests | Train on small corpus, verify vocab size, round-trip encode/decode, compare with tiktoken on same text |

---

## Parallelism Summary

What can run at the same time:

| Phase | Parallel tracks |
|-------|----------------|
| Phase 1 | 1A, 1B, 1C, 1D, 1E — all five libraries simultaneously (after interface contracts defined) |
| Phase 2 | 2.1-2.4 are sequential (data → script → configs → eval), then 2.5 runs it |
| Phase 3 | 3.1 (REINFORCE) can start during Phase 2 if model interface is stable. 3.2-3.3 (GRPO) can overlap |
| Phase 3.5 | Experiments are sequential (each informs the next) but can overlap with Phase 4 |
| Phase 4 | 4.2, 4.3, 4.4, 4.5 can all be built in parallel once data models exist |
| Phase 5 | Fully independent, can run in parallel with anything |

## Estimated Timeline

Rough estimates assuming agents work in parallel where possible. Implementation time and experimentation/training time are separated — ML projects spend more time experimenting than coding.

| Phase | Implementation | Experimentation | Calendar (parallel) |
|-------|---------------|-----------------|---------------------|
| Phase 0 | 1-2 hours | — | 1-2 hours |
| Phase 1 | 8-12 hours total | — | 3-4 hours (5 parallel) |
| Phase 2 | 4-6 hours | 2-4 hours (training runs) | 6-10 hours |
| Phase 2.5 | 1-2 hours | 2-4 hours (evaluations) | 3-6 hours |
| Phase 3 | 6-8 hours | 4-8 hours (REINFORCE + GRPO runs) | 10-16 hours |
| Phase 3.5 | 2-3 hours | 6-12 hours (ablations, tuning) | 8-15 hours |
| Phase 4 | 6-8 hours total | — | 3-4 hours (parallel) |
| Phase 5 | 2-3 hours | — | 2-3 hours |
| **Total** | **~30-44 hours** | **~14-28 hours** | **~36-60 hours** |

The experimentation time is where the learning happens. Don't rush it.
