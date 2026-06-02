# Wordle Game Token Format

This document defines the token format used for Wordle game data across pre-training and RL fine-tuning. The format must be consistent between phases — the model should see the same token patterns during pre-training that it will encounter during RL.

## Vocabulary

| Token | ID | Purpose |
|-------|-----|---------|
| `a`-`z` | 0-25 | Letter characters |
| `[green]` | 26 | Feedback: correct letter, correct position |
| `[yellow]` | 27 | Feedback: correct letter, wrong position |
| `[gray]` | 28 | Feedback: letter not in word |
| `[bos]` | 29 | Beginning of game |
| `[pad]` | 31 | Padding within a batch |
| `[sep]` | 32 | Separator — signals "next guess follows" |

## Pre-Training Data Format

Each training example is a **partial game state** (the prompt) followed by the **next guess** (the target). From a single completed game, we generate one example per turn.

**Example: a 3-turn game where the target is "crane", guesses are "slate", "prone", "crane"**

This game produces 3 training examples:

```
Example 1 (turn 1):
  Prompt: [bos]
  Target: s l a t e

Example 2 (turn 2):
  Prompt: [bos] s l a t e [gray] [gray] [green] [gray] [green] [sep]
  Target: p r o n e

Example 3 (turn 3):
  Prompt: [bos] s l a t e [gray] [gray] [green] [gray] [green] [sep] p r o n e [gray] [green] [gray] [green] [green] [sep]
  Target: c r a n e
```

The prompt is always a valid RL game state. The target is always exactly 5 letter tokens — the next guess.

### Loss Masking

The training loss is computed **only on the 5 target letter tokens** (the guess). Feedback tokens (`[green]`, `[yellow]`, `[gray]`) and `[sep]` tokens appear in the prompt context but the model is NOT trained to predict them.

**Why:** During RL, the model only generates letters. Feedback is injected by the environment. Training the model to predict feedback wastes capacity and distorts the output distribution — after generating 5 letters, the model's logits would favor feedback tokens instead of being ready for the next game state.

The model learns to *read* feedback tokens (they are in the input context and influence attention), but not to *produce* them.

## Batching

Each training example is one partial game state + 5 target letters. Games are **never split across examples**. Multiple examples are batched together, padded to the length of the longest example in the batch using `[pad]`.

## Game State Prompt (RL fine-tuning)

During RL, the model sees a partial game state and generates 5 characters for the next guess. The prompt format is identical to the pre-training prompt.

**Turn 1 — no guesses yet:**

```
[bos]
```

**Turn 2 — after one guess:**

```
[bos] s l a t e [gray] [gray] [green] [gray] [green] [sep]
```

**Turn 3 — after two guesses:**

```
[bos] s l a t e [gray] [gray] [green] [gray] [green] [sep] p r o n e [gray] [green] [gray] [green] [green] [sep]
```

## Format Consistency

The RL prompt is identical to the pre-training prompt. No format mismatch between phases.

| Situation | Pre-training prompt | RL prompt |
|-----------|-------------------|-----------|
| First guess | `[bos]` | `[bos]` |
| After turn 1 | `[bos] ... [green] [sep]` | `[bos] ... [green] [sep]` |
| After turn 5 | `[bos] ... [green] [sep]` | `[bos] ... [green] [sep]` |

## Token Counts

| Component | Tokens |
|-----------|--------|
| A single guess + feedback | 10 (5 letters + 5 feedback) |
| Turn separator | 1 (`[sep]`) |
| Turn 1 prompt | 1 (`[bos]`) |
| Turn 2 prompt | 12 (`[bos]` + 10 + `[sep]`) |
| Turn 6 prompt | 56 (`[bos]` + 5×11 + `[sep]`) |
| Target | 5 (always 5 letters) |
| Max example length (prompt + target) | 61 |

All examples fit well within the 256-token context window. Padding overhead is minimal since max example length is 61 tokens.
