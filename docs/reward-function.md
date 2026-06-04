# Reward Function Design

## Modes

The reward function has two modes controlled by the `composite` flag.

### Non-composite (`composite=False`) — Phase 1

Used for Phase 1 RL (turns 1-2, learning openers).

```
reward = expected_info_gain(guess, candidates_before)
```

Returns the deterministic expected information gain in bits. Does not
depend on the target word. Higher means the guess splits the candidate
space more effectively.

Invalid words receive `INVALID_WORD_PENALTY` (-10.0).

### Composite (`composite=True`) — Phase 2

Used for Phase 2 RL (turns 3-6, learning mid/endgame strategy).

```
if guess is not a valid word:
    reward = INVALID_WORD_PENALTY

elif candidates > 2:
    reward = (expected_ig / log2(candidates)) * INFO_GAIN_SCALE

elif candidates <= 2 and guess in candidates:
    reward = (expected_ig / log2(candidates)) * INFO_GAIN_SCALE
           + ENDGAME_BONUS
           + SOLVED_BONUS (if all green)

elif candidates <= 2 and guess not in candidates:
    reward = (expected_ig / log2(candidates)) * INFO_GAIN_SCALE
```

When candidates = 1, expected info gain is 0 and log2(1) = 0, so the
normalized info gain term is 0. The endgame bonus and solve bonus are
the only signal.

The info gain used is **expected** info gain — a deterministic measure
of guess quality computed by simulating feedback against every candidate.
It does not depend on the actual target word.

## Strategy Phases

1. **Opener (turn 1-2):** Play high-coverage words (≥5.5 bits expected
   info gain). Threshold: top ~50 words out of 2315.

2. **Midgame (3+ candidates):** Play discovery words that distinguish
   between remaining candidates. These may NOT be candidates themselves.
   Normalized info gain rewards any word that splits well.

3. **Endgame (1-2 candidates):** Guess a candidate. Discovery is no
   longer useful. Endgame bonus rewards guessing from the candidate set.

## Invalid Word Penalty

Both modes penalize invalid words (words not in the Wordle valid word
list of ~12,972 words). The penalty is -10.0, worse than any valid
word's reward.

## Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| INFO_GAIN_SCALE | 10.0 | Dominates reward during opener and midgame |
| ENDGAME_BONUS | 3.0 | Signals "guess from the candidate set" |
| SOLVED_BONUS | 5.0 | Only in endgame, rewards solving |
| INVALID_WORD_PENALTY | -10.0 | Worse than any valid word's reward |

## Return Value

`compute_reward` returns `(reward, actual_info_gain, expected_info_gain)`.

- `reward`: the training signal (per-turn, never summed across turns)
- `actual_info_gain`: how many bits were actually gained (depends on
  target word, used for logging only)
- `expected_info_gain`: the deterministic expected info gain (used as
  reward in non-composite mode, normalized in composite mode)

## Reward by Phase

| Phase | Candidates | Reward components |
|-------|-----------|-------------------|
| Opener | 2315 | normalized_ig * 10 (max ~5.3) |
| Midgame | 3-100 | normalized_ig * 10 (max ~10) |
| Endgame | 1-2 | endgame_bonus (3) + solve_bonus (5) = 8 |
| Invalid word | any | -10 |
