# Training Results

## Pre-training

- **Model:** 5M params (6L/8H/256d)
- **Data:** 20,000 game transcripts (mixed skill levels), per-turn examples with loss masking
- **Steps:** 10,000
- **Val loss:** 1.30
- **Valid-word-rate:** 55%
- **Eval win rate (constrained):** 30%
- **Pretrain eval:** All 5 criteria passed

The pre-trained model learned the game format, generates diverse valid words, and plays at 30% win rate from game transcript imitation alone.

## Phase 1: Information Gathering (Turns 1-2)

- **Reward:** Expected information gain (deterministic, no solve bonus)
- **Group size:** 8
- **Max turns:** 2
- **KL beta:** 0.2
- **Learning rate:** 1e-5
- **PPO epochs:** 2

### Result at ~1000 steps

- **Turn 1:** Model converged to "SLATE" (5.9 bits expected info gain, near-optimal). Top 5 words by expected info gain are raise (5.88), slate (5.86), crate (5.83), irate (5.83), trace (5.83).
- **Turn 2:** Diverse words selected based on turn-1 feedback. CREED, BARON, CRONY, RAINY — different words for different feedback patterns.
- **Reward mean:** ~10 (two turns of ~5 bits each)
- **Eval win rate:** Peaked at 30% (step 800), settled at 24%. Expected — Phase 1 doesn't optimize for solving.
- **KL divergence:** Climbed from 0.02 to 1.4. Model drifted from pre-trained distribution, collapsing to SLATE for turn 1.

### Key observations

1. The deterministic reward (expected info gain) worked. The model consistently moved toward high-info words and away from low-info words like "MARRY" or "FUZZY".
2. Convergence was fast — good openings emerged by step ~200, near-optimal by step ~500.
3. The model learned to use turn-1 feedback for turn-2 word selection.
4. Previous RL attempts with actual-minus-expected reward failed because lucky bad guesses got reinforced as often as they got penalized.

## Phase 2: Solving (Turns 3-6)

Not yet run. Will start from Phase 1 checkpoint with solve bonus enabled.
