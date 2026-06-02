# Reward Function Design

This document describes the reward function used to train the Wordle model via GRPO. The reward function scores a single guess given the current game state.

## Inputs

- **state**: The game state after the guess has been applied. Contains all prior guesses, their feedback, whether the game is solved/failed.
- **guess**: The 5-letter word that was guessed.
- **feedback**: The green/yellow/gray feedback for this guess (computed by the environment from the guess and the hidden target word).
- **valid_words**: The set of all valid Wordle words.
- **candidates_before**: The list of remaining candidate answer words before this guess. On turn 1 this is all 2,315 answer words. Each subsequent turn, it is filtered to only words consistent with all prior feedback.

## Scoring Logic

The reward is computed by checking conditions in priority order. The first matching condition returns immediately — later conditions are not evaluated.

### Priority 1: Invalid word → -1.0

The guess is not in the valid word list. With trie-constrained decoding this never happens.

### Priority 2: Repeated guess → -0.5

The guess was already made earlier in this game.

### Priority 3: Solved → +10.0

The guess matches the target word. All 5 letters are green. This is the highest possible reward.

### Priority 4: Failed → -0.5

This was the 6th guess and it did not solve the puzzle. The game is lost.

### Priority 5: Contradicts clues → -0.3

The guess uses a letter that was previously gray (known absent from the target), or places a letter in a position that contradicts a previous green or yellow clue.

This check applies unconditionally regardless of turn number.

### Priority 6: Score based on feedback + information gain

If none of the above conditions triggered, the reward is the sum of:

**Green/yellow letter bonuses:**
- +0.2 per green letter (correct letter, correct position)
- +0.1 per yellow letter (correct letter, wrong position)

**Information gain in bits:**

`candidates_after` is computed by filtering `candidates_before` to only words consistent with this guess's feedback. The information gain is:

```
info_bits = log2(len(candidates_before) / len(candidates_after))
```

This measures how much the guess narrowed down the remaining possibilities. Examples:

| Scenario | candidates_before | candidates_after | info_bits |
|----------|------------------|-----------------|-----------|
| Turn 1, good guess | 2315 | 100 | 4.5 bits |
| Turn 1, average guess | 2315 | 300 | 2.9 bits |
| Turn 1, bad guess | 2315 | 1000 | 1.2 bits |
| Turn 3, good guess | 50 | 3 | 4.1 bits |
| Turn 3, bad guess | 50 | 25 | 1.0 bits |

The information gain reward has weight 1.0 (configurable via `elimination_weight`).

**No new information:**

If the total of green/yellow bonuses + information gain is exactly 0.0, the reward is 0.0 (configurable via `no_new_info`).

## Candidate Tracking

The `candidates_before` list is maintained across turns within a game:

1. Turn 1: `candidates_before` = all 2,315 answer words
2. After the chosen guess is played, `candidates_before` is filtered using `filter_candidates(candidates_before, chosen_guess, feedback)` to remove words inconsistent with the feedback
3. Turn 2: `candidates_before` = the filtered list from step 2
4. Repeat

In GRPO, the group of 4 candidate guesses on each turn are all scored against the same `candidates_before`. Each candidate gets different feedback (same target, different guess), so each gets a different information gain score. Only the best-scoring candidate is actually played to advance the game, and `candidates_before` is filtered based on that candidate's feedback.

## Reward Ranges

| Situation | Typical reward |
|-----------|---------------|
| Solved | +10.0 |
| Turn 1, good info guess | +4 to +7 |
| Turn 1, average guess | +2 to +4 |
| Turn 1, bad guess | +0.5 to +1.5 |
| Late turn, good guess | +3 to +5 |
| Failed (6th miss) | -0.5 |
| Repeated guess | -0.5 |
| Contradicts clues | -0.3 |

## Configuration

All values are configurable via `RewardConfig`:

```python
@dataclass
class RewardConfig:
    invalid_word: float = -1.0
    repeated_guess: float = -0.5
    contradicts_clues: float = -0.3
    no_new_info: float = 0.0
    green_letter: float = 0.2
    yellow_letter: float = 0.1
    elimination_weight: float = 1.0
    solved: float = 10.0
    failed: float = -0.5
```
