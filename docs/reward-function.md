# Reward Function Design

The reward for a guess is its information gain relative to the expected information gain, with a special case for solving the puzzle.

```
reward = actual_info_gain - expected_info_gain
```

## Inputs

- **guess**: The 5-letter word that was guessed.
- **feedback**: The green/yellow/gray feedback for this guess.
- **candidates_before**: The remaining candidate answer words before this guess.

## Actual Information Gain

`candidates_after` is computed by filtering `candidates_before` to only words consistent with the feedback. The actual information gain is:

```
actual = log2(len(candidates_before) / len(candidates_after))
```

## Expected Information Gain

For each candidate target in `candidates_before`, compute the feedback pattern that guess would produce. Group candidates by pattern and compute the weighted average info gain:

```
expected = Σ (count / N) * log2(N / count)
```

where `count` is the number of candidates producing each unique feedback pattern, and `N = len(candidates_before)`.

## Reward

```
reward = actual - expected
```

- **Positive reward**: the guess performed better than expected.
- **Zero reward**: the guess performed exactly as expected.
- **Negative reward**: the guess performed worse than expected.

## Special Case: Solving

When all 5 letters are green, the guess solved the puzzle. This gets a fixed bonus reward regardless of the candidate count.

**Why this is needed:** Information gain cannot distinguish solving from not solving when `candidates_before` is small. With 1 candidate remaining, both a correct guess and an incorrect guess leave `candidates_after = 1` — the candidate list doesn't change either way. Without a solve bonus, the model has no incentive to guess the known answer. It would waste turns guessing random words with zero reward.

The solve bonus is:

```
solved_bonus = log2(N_answers)   # ~11.2 bits for 2315 answers
```

This equals the maximum possible single-turn info gain (guessing correctly from the full list on turn 1), so it scales consistently with the information gain metric.

## Examples (turn 1, 2315 candidates, target = "crane")

| Guess | Actual | Expected | Reward | Why |
|-------|--------|----------|--------|-----|
| slate | 6.4 bits | 5.9 bits | +0.5 | Good guess, slightly above average feedback |
| arose | 6.9 bits | 5.8 bits | +1.2 | Great guess, good feedback |
| crane | 11.2 bits | 5.7 bits | +5.4 | Solved — maximum info gain |
| fuzzy | 0.8 bits | 2.3 bits | -1.5 | Bad guess, barely eliminates anything |

## Examples (late turn, 1 candidate remaining)

| Guess | Reward | Why |
|-------|--------|-----|
| correct word | +11.2 (bonus) | Solved the puzzle |
| wrong word | 0.0 | No info gained, no penalty |

## Candidate Tracking

`candidates_before` starts as all 2,315 answer words on turn 1. After each turn, it is filtered based on the chosen guess's feedback. The filtered list becomes `candidates_before` for the next turn.

In GRPO, the group of 4 candidates per turn are all scored against the same `candidates_before`. Each gets different feedback (same target, different guess), so each gets a different reward. The best-scoring candidate is played to advance the game.

## Performance

Expected info gain is computed by counting feedback patterns, not by filtering per target. This is O(N) where N = len(candidates_before). ~3-6ms per guess on turn 1, faster on later turns with fewer candidates.
