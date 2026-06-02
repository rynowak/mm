# Reward Function Design

The reward function has two modes, matching the two curriculum phases.

## Phase 1: Expected Information Gain (Deterministic)

```
reward = expected_info_gain(guess, candidates_before)
```

The expected info gain is computed by simulating the guess against every candidate in the pool, grouping by feedback pattern, and computing the weighted average:

```
expected = Σ (count / N) * log2(N / count)
```

This is a deterministic measure of guess quality. It does not depend on the target word. "raise" always scores 5.88, "marry" always scores 4.22, "fuzzy" always scores 2.31.

**Why deterministic:** The old reward (`actual - expected`) depended on the target word. A bad guess could score positive if it got lucky feedback. GRPO reinforced lucky bad guesses as often as it penalized them, so the model didn't learn to avoid them. With deterministic expected info gain, GRPO consistently reinforces high-info words and suppresses low-info words.

**No solve bonus.** Phase 1 trains turns 1-2 only. Solving on turn 1-2 means guessing the answer, which is not optimal information gathering. A word that happens to be the answer doesn't necessarily split the candidate space well.

## Phase 2: Expected Information Gain + Solve Bonus

```
reward = expected_info_gain(guess, candidates_before)
if solved:
    reward += SOLVED_BONUS  # ~11.2 bits = log2(2315)
```

Same deterministic base reward as Phase 1. When the model solves the puzzle (all 5 feedback letters green), it gets the solve bonus on top.

**Why the solve bonus is needed in Phase 2:** Information gain cannot distinguish solving from not solving when `candidates_before` is small. With 1 candidate remaining, both a correct guess and an incorrect guess leave `candidates_after = 1`. Without the bonus, the model has no incentive to guess the known answer.

The bonus equals the maximum possible single-turn info gain (~11.2 bits), so it scales consistently with the base reward.

## Expected Info Gain Computation

For each candidate target in `candidates_before`, compute the feedback pattern that the guess would produce. Group candidates by pattern. The expected info gain is the weighted average of `log2(N / count)` across all patterns.

| Guess | Expected Info Gain | Why |
|-------|-------------------|-----|
| raise | 5.88 bits | High letter diversity, common letters |
| slate | 5.86 bits | Similar quality to raise |
| crane | 5.74 bits | Good but slightly less diverse |
| marry | 4.22 bits | Repeated R wastes a slot |
| fuzzy | 2.31 bits | Uncommon letters, low coverage |

These values are **precomputed** for the full 2,315-word answer list at startup (~6s one-time cost). Turn 1 reward lookups are instant.

## Candidate Tracking

`candidates_before` starts as all 2,315 answer words on turn 1. After each turn, it is filtered based on the chosen guess's feedback. The filtered list becomes `candidates_before` for the next turn.

In GRPO, the group of candidates per turn are all scored against the same `candidates_before`. Each gets a deterministic expected info gain score. The best-scoring candidate is played to advance the game.

## Summary

| | Phase 1 | Phase 2 |
|---|---------|---------|
| Turns | 1-2 | 3-6 |
| Base reward | Expected info gain | Expected info gain |
| Solve bonus | No | Yes (+11.2 bits) |
| Depends on target | No | Only for solve bonus |
| Objective | Learn to gather information | Learn to solve the puzzle |
