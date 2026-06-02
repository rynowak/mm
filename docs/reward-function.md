# Reward Function Design

The reward for a guess is its information gain relative to the expected information gain:

```
reward = actual_info_gain - expected_info_gain
```

This measures decision quality, not outcome luck.

## Inputs

- **guess**: The 5-letter word that was guessed.
- **feedback**: The green/yellow/gray feedback for this guess.
- **candidates_before**: The remaining candidate answer words before this guess.

## Actual Information Gain

`candidates_after` is computed by filtering `candidates_before` to only words consistent with the feedback. The actual information gain is:

```
actual = log2(len(candidates_before) / len(candidates_after))
```

This depends on both the guess quality AND the target word (luck).

## Expected Information Gain

For each candidate target in `candidates_before`, compute the feedback pattern that guess would produce. Group candidates by pattern and compute the weighted average info gain:

```
expected = Σ (count / N) * log2(N / count)
```

where `count` is the number of candidates producing each unique feedback pattern, and `N = len(candidates_before)`.

This measures the intrinsic quality of the guess — how well it splits the candidate pool on average, regardless of what the target happens to be.

## Reward

```
reward = actual - expected
```

- **Positive reward**: the guess performed better than expected. Either the guess was inherently good, or it got lucky feedback, or both.
- **Zero reward**: the guess performed exactly as expected.
- **Negative reward**: the guess performed worse than expected. A bad guess that doesn't split candidates well, or an unlucky feedback outcome.

## Examples (turn 1, 2315 candidates, target = "crane")

| Guess | Actual | Expected | Reward | Why |
|-------|--------|----------|--------|-----|
| slate | 6.4 bits | 5.9 bits | +0.5 | Good guess, slightly above average feedback |
| arose | 6.9 bits | 5.8 bits | +1.2 | Great guess, good feedback |
| crane | 11.2 bits | 5.7 bits | +5.4 | Solved it — massive info gain |
| fuzzy | 0.8 bits | 2.3 bits | -1.5 | Bad guess, barely eliminates anything |
| mummy | 0.8 bits | 2.5 bits | -1.7 | Repeated letters waste information |

## Candidate Tracking

`candidates_before` starts as all 2,315 answer words on turn 1. After each turn, it is filtered based on the chosen guess's feedback. The filtered list becomes `candidates_before` for the next turn.

In GRPO, the group of 4 candidates per turn are all scored against the same `candidates_before`. Each gets different feedback (same target, different guess), so each gets a different reward.

## Performance

Expected info gain is computed by counting feedback patterns, not by filtering per target. This is O(N) where N = len(candidates_before). ~3-6ms per guess on turn 1, faster on later turns with fewer candidates.

## Why This Design

The normalization by expected value means:

- **No special cases needed.** Solving, failing, repeating guesses, contradicting clues — all are naturally handled by information gain. Solving gives the maximum possible info gain. Repeating a guess gives 0 bits. A guess that ignores clues gets low info gain.
- **No perverse incentives.** The old priority-ordered reward function penalized exploratory guesses that reused gray letters (which is optimal early-game strategy). It also rewarded longer games with higher total reward. Pure information gain avoids both.
- **Decision quality over outcome.** Two guesses with the same expected info gain but different actual outcomes get different raw scores — but the normalization separates the decision quality from the luck. GRPO learns which guesses are inherently good, not which ones happened to get lucky feedback.
