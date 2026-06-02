# Reward Function Design

The reward for a guess is its information gain in bits:

```
reward = log2(candidates_before / candidates_after)
```

## Inputs

- **guess**: The 5-letter word that was guessed.
- **feedback**: The green/yellow/gray feedback for this guess.
- **candidates_before**: The remaining candidate answer words before this guess.

## Computation

`candidates_after` is computed by filtering `candidates_before` to only words consistent with the feedback from this guess. The reward is how many bits of information the guess provided.

| Scenario | Before | After | Bits |
|----------|--------|-------|------|
| Good turn 1 guess | 2315 | 100 | 4.5 |
| Average turn 1 guess | 2315 | 300 | 2.9 |
| Bad turn 1 guess | 2315 | 1000 | 1.2 |
| Solved (any turn) | N | 1 | log2(N) |
| Good late guess | 50 | 3 | 4.1 |

Solving the puzzle reduces candidates to 1, which naturally produces the highest information gain for that turn. No special solved/failed bonuses are needed.

## Candidate Tracking

`candidates_before` starts as all 2,315 answer words on turn 1. After each turn, it is filtered to only words consistent with the chosen guess's feedback. This filtered list becomes `candidates_before` for the next turn.

In GRPO, the group of 4 candidates per turn are all scored against the same `candidates_before`. Each candidate gets different feedback (same target, different guess), so each gets a different information gain. The best-scoring candidate is played to advance the game.

## Why This Design

No priority-ordered early returns. No special cases for solved, failed, repeated, or contradicting guesses. Every guess is scored by the same metric: how much did it narrow down the possibilities?

This avoids:
- Penalizing exploratory guesses that reuse gray letters (which is optimal strategy early in the game)
- Rewarding solved games with a flat bonus that doesn't account for turn number
- Creating perverse incentives where longer games accumulate more total reward
