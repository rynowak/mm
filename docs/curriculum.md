# Curriculum: Two-Phase RL Training

The model already plays at 30% win rate from pre-training alone. Full-game GRPO degrades this because the reward signal is too noisy across 6 turns with 2,315 candidates. This curriculum splits RL into two phases, each with a focused objective.

## Problem

Full-game RL fails because:

1. **Turns 1-2 and turns 3-6 require different skills.** Early turns are about information gathering (which letters exist, where). Late turns are about solving (guessing the answer from a small candidate pool).
2. **The reward signal is dominated by turn 1.** With 2,315 candidates, turn 1 info gain ranges from 0-11 bits. By turn 3, there are often <10 candidates and the info gain range collapses. GRPO only learns from turn 1.
3. **The solve bonus comes too late.** The model needs 3-6 turns of good play before it gets the +11.2 solve bonus. The credit assignment across that many turns is too noisy for a 5M param model with group_size=4.

## Phase 1: Information Gathering (Turns 1-2)

**Objective:** Learn to pick opening words that maximize information gain.

**Setup:**
- Play only 2 turns per game. No solving pressure.
- Reward: `actual_info_gain - expected_info_gain` on each turn.
- Candidate tracking: starts at 2,315, filtered after turn 1.
- After 2 turns, the game ends regardless of whether the puzzle is solved.

**What the model learns:**
- Which first-guess words split the candidate space most effectively.
- How to use turn-1 feedback to pick a good turn-2 word.
- No need to learn solving — just information discovery.

**Success criteria:**
- Average info gain on turn 1 exceeds the pre-trained model's average.
- Average remaining candidates after 2 turns decreases over training.
- The model's turn-1 word distribution shifts toward high-entropy words.

## Phase 2: Solving (Turns 3-6)

**Objective:** Learn to solve the puzzle from a narrowed candidate pool.

**Setup:**
- Games start at turn 3 with 2 guesses already played.
- The first 2 guesses come from the Phase 1 model (frozen) playing against the target word.
- Reward: `actual_info_gain - expected_info_gain` + solve bonus.
- The candidate pool at the start of Phase 2 is typically 5-50 words (after 2 good guesses).
- The model has 4 remaining turns to solve.

**What the model learns:**
- How to use feedback from prior guesses to narrow down the answer.
- When to "go for the kill" (guess a likely answer) vs explore more.
- The solve bonus (+11.2 bits) provides strong signal because the candidate pool is small and solving is achievable.

**Success criteria:**
- Win rate on the Phase 2 eval set improves over training.
- Average guesses to solve decreases.
- The model solves when <5 candidates remain (this is the current failure mode).

## Why This Works

**Phase 1 is learnable.** With 2,315 candidates and only 2 turns, the reward signal is strong and consistent. Every guess gets meaningful info gain. The model can learn opening strategy without the noise of mid-game play.

**Phase 2 is learnable.** Starting from 5-50 candidates, the model can actually solve within 4 turns. The solve bonus is achievable and provides a clear gradient signal. The model doesn't need to learn openings — those are provided by Phase 1.

**Credit assignment is clean.** Phase 1: 2 turns of info gain, no ambiguity about what's being rewarded. Phase 2: solve within 4 turns from a small pool, clear connection between guess quality and outcome.

## Data Flow

```
Pre-trained model
    │
    ▼
Phase 1 RL (turns 1-2, info gain only)
    │
    ▼
Phase 1 model (frozen, used for opening play)
    │
    ▼
Phase 2 RL (turns 3-6, info gain + solve bonus)
    │
    ▼
Final model
```

At inference time, the final model plays all 6 turns. Phase 1 trained its opening instincts, Phase 2 trained its closing instincts. The pre-trained foundation connects them — the model saw full games during pre-training.

## Evaluation

After each phase, run the full eval (all 6 turns, same 50-game eval set):
- Phase 1 model should have higher turn-1 info gain but similar or slightly better win rate than pre-trained.
- Phase 2 model should have significantly higher win rate than Phase 1.
- Compare: pre-trained (30%) → Phase 1 → Phase 2.

## Configuration

Phase 1 YAML adds:
```yaml
rl:
  curriculum_phase: 1
  max_turns: 2
```

Phase 2 YAML adds:
```yaml
rl:
  curriculum_phase: 2
  opening_checkpoint: path/to/phase1/model.pt
```
