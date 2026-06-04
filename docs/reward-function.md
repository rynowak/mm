# Reward Function Design

## Strategy Phases

Optimal Wordle play has three distinct phases:

1. **Opener (turn 1-2):** Play high-coverage words that eliminate large portions of
   the candidate space. Words like "slate" or "crane" that test common letters.

2. **Midgame (3+ candidates):** Play "discovery" words that distinguish between
   remaining candidates. These words may NOT be candidates themselves — the goal
   is elimination, not solving.

3. **Endgame (1-2 candidates):** Guess a candidate. Discovery is no longer useful.

## Reward Formula

```
reward = normalized_info_gain + endgame_bonus + solve_bonus
```

### Normalized Info Gain

```
max_possible = log2(n_candidates)
normalized = info_gain / max_possible    # 0 to 1
reward_ig = normalized * INFO_GAIN_SCALE
```

Info gain is normalized by the maximum possible info gain at that game state.
This puts all turns on the same scale — a 50% efficient opener and a 90%
efficient midgame play produce comparable reward magnitudes.

`INFO_GAIN_SCALE` is set high enough that optimal information gathering always
dominates over lucky outcomes during opener and midgame.

When candidates = 1, max_possible = 0 and info gain is 0. The endgame bonus
takes over.

### Endgame Bonus

```
if n_candidates <= 2 and guess in candidates:
    reward += ENDGAME_BONUS
```

Only activates when candidates are 1-2. Rewards guessing from the candidate
set when discovery is no longer useful.

### Solve Bonus

```
if solved and n_candidates <= 2:
    reward += SOLVED_BONUS
```

Only given when solving IS the right strategy (endgame). Solving with 10
candidates is luck, not skill — no bonus. Solving with 1-2 candidates is
the intended play and gets rewarded.

## Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| INFO_GAIN_SCALE | 10.0 | Dominates reward during opener and midgame |
| ENDGAME_BONUS | 3.0 | Comparable to a good normalized info gain |
| SOLVED_BONUS | 5.0 | Only in endgame, dominates when solving is optimal |

## Reward by Phase

| Phase | Candidates | Reward components |
|-------|-----------|-------------------|
| Opener | 2315 | normalized_ig * 10 (max ~5.3) |
| Midgame | 3-100 | normalized_ig * 10 (max ~10) |
| Endgame | 1-2 | endgame_bonus (3) + solve_bonus (5) = 8 |

In the midgame, a discovery word that achieves 100% of max info gain scores 10.
A lucky solve with lower info gain scores less. Info gain drives the policy
toward optimal play, not gambling.

In the endgame, the combined bonus (8) exceeds typical midgame rewards, so the
model learns to shift from discovery to solving when candidates are low.
