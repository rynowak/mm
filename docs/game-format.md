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

Each training example is a **partial game state** (the prompt) followed by the **next guess** (the target). A single completed game produces multiple training examples — one per turn.

**Example: a 3-turn game where the target is "crane"**

This game produces 3 training examples:

```
Example 1 (turn 1):
  Input:  [bos] c r a n
  Target: c r a n e [green] [green] [green] [green] [green]
  (prompt is [bos], target is the first guess + feedback)

Example 2 (turn 2, if guess 1 was "slate"):
  Input:  [bos] s l a t e [gray] [gray] [green] [gray] [green] [sep] p r o n
  Target: r o n e [gray] [green] [gray] [green] [green] [sep] c r a n e
  (prompt is game state after turn 1 + [sep], target continues the sequence)

Example 3 (turn 3):
  Input:  [bos] s l a t e [gray] [gray] [green] [gray] [green] [sep] p r o n e [gray] [green] [gray] [green] [green] [sep] c r a n
  Target: n e [green] [green] [green] [green] [green]
```

The model learns next-token prediction. Given the game state, it learns to:
- After `[bos]`: generate the first letter of a guess
- After `[sep]`: generate the first letter of the next guess
- After guess letters: predict the feedback tokens
- Use feedback context to inform the next guess

## Batching

Each training example is one partial game — a self-contained sequence. Games are **never split across examples**. Multiple examples are batched together, padded to the length of the longest example in the batch using `[pad]`.

This matches RL exactly: during fine-tuning, the model sees a complete game state and generates the next word. Pre-training uses the same framing.

## Game State Prompt (RL fine-tuning)

During RL, the model sees a partial game state and generates 5 characters for the next guess. The prompt format is identical to the pre-training input.

**Turn 1 — no guesses yet:**

```
[bos]
```

The model generates 5 characters. This is what the model saw at the start of every pre-training example.

**Turn 2 — after one guess:**

```
[bos] s l a t e [gray] [gray] [green] [gray] [green] [sep]
```

`[sep]` signals "generate the next guess." The model learned during pre-training that letters follow `[sep]`.

**Turn 3 — after two guesses:**

```
[bos] s l a t e [gray] [gray] [green] [gray] [green] [sep] p r o n e [gray] [green] [gray] [green] [green] [sep]
```

## Format Consistency

The RL prompt is always identical to what the model saw during pre-training. There is no format mismatch between phases.

| Situation | Pre-training input | RL prompt |
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
| Max example length (turn 6 + guess) | 61 |

All examples fit well within the 256-token context window. Padding overhead is minimal since max example length is 61 tokens.
