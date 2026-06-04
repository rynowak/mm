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
    best_ig = max expected info gain across all answer words for this candidate set
    normalized = expected_ig / best_ig    # 0 to 1
    reward = normalized * INFO_GAIN_SCALE

elif candidates <= 2 and guess in candidates:
    reward = ENDGAME_BONUS + SOLVED_BONUS (if all green)

elif candidates <= 2 and guess not in candidates:
    reward = 0.0
```

The key change from previous design: normalization is against the **best
available word**, not the theoretical maximum `log2(candidates)`. This
means:

- A word that achieves the best possible info gain for this game state
  scores 1.0 (normalized) → 10.0 (scaled).
- A word that achieves half the best possible scores 0.5 → 5.0.
- The reward measures **how close to optimal the play was**, not just
  how much information was gained.

Previously, normalization against the theoretical `log2(candidates)`
meant mediocre plays could score high if the candidate set was easy
to split, and optimal plays could score low if the theoretical ceiling
was unreachable. The model was rewarded for "decent" plays rather than
"the best available play."

When candidates = 1, expected info gain is 0 for all words. The endgame
bonus and solve bonus are the only signal.

The info gain used is **expected** info gain — a deterministic measure
of guess quality computed by simulating feedback against every candidate.
It does not depend on the actual target word.

## Computing Best Available Info Gain

For a given candidate set, the best available info gain is the maximum
`expected_info_gain(word, candidates)` across all answer words. This
requires evaluating ~2,315 words per turn.

For turn 1 (full 2,315 candidate set), this is precomputed at startup.
The best opener is ~5.88 bits (raise).

For later turns with smaller candidate sets, the computation is fast
because `expected_info_gain` iterates over the candidate list (which
shrinks each turn). With 10 candidates, evaluating 2,315 words × 10
candidates = 23,150 feedback pattern computations — negligible.

## Strategy Phases

1. **Opener (turn 1-2):** Play high-coverage words. The best openers
   score 1.0 normalized (≥5.5 bits). Threshold: top ~50 words.

2. **Midgame (3+ candidates):** Play discovery words that distinguish
   between remaining candidates. The reward measures how close to the
   best available play this guess is.

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
| Opener | 2315 | normalized_ig * 10 (optimal = 10.0) |
| Midgame | 3-100 | normalized_ig * 10 (optimal = 10.0) |
| Endgame | 1-2 | endgame_bonus (3) + solve_bonus (5) = 8 |
| Invalid word | any | -10 |
