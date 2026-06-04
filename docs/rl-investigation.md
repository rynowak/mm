# RL Training Investigation

Goal: 100% solve rate with semi-optimal turn count. Current best: 47.5%.

## Methodology Bug

The eval code had inconsistent decoding temperatures. `evaluate.py` used temp=0.1
(near-greedy), `finetune.py:evaluate_games` used temp=1.0 (stochastic). The reported
"pretrained 30%, RL 24%" numbers came from different temperatures and were never
comparable. This is a code bug that produced a false narrative of RL degradation.

All results below use consistent methodology: temp=0.1, 200 words (seed 99), 
`torch.manual_seed(42)` reset per model for identical random draws.

## Experiment 1: Win Rates (temp=0.1, 200 games)

| Model | Wins | Win Rate | Avg Guesses | vs Pretrained |
|-------|------|----------|-------------|---------------|
| pretrained | 65/200 | 32.5% | 5.32 | — |
| phase1 (turns 1-2 GRPO) | 54/200 | 27.0% | 5.46 | -5.5pp |
| phase2 (turns 3-6 GRPO) | 76/200 | 38.0% | 5.35 | +5.5pp |
| phase2 + phase1 opener | 95/200 | 47.5% | 5.13 | +15.0pp |
| pretrained + phase1 opener | 68/200 | 34.0% | 5.33 | +1.5pp |

**RL is working.** Phase 2 beats pretrained by 5.5pp solo. Phase 2 + opener is 47.5%.
Phase 2 + opener beats pretrained + same opener (47.5% vs 34.0%), proving Phase 2
learned better turns 3-6 play specifically.

Phase 1 solo is worse than pretrained (27.0% vs 32.5%) because it collapsed first-guess
diversity and has high repetition rate. But it produces the best opener for Phase 2.

## Experiment 2: Game-by-Game Comparison

| Comparison | Both Win | Both Lose | Regressions | Improvements | Net |
|-----------|----------|-----------|-------------|--------------|-----|
| pretrained vs phase1 | 39 | 120 | 26 | 15 | -11 |
| pretrained vs phase2 | 45 | 104 | 20 | 31 | +11 |
| pretrained vs phase2+opener | 49 | 89 | 16 | 46 | **+30** |

Phase 2+opener: 46 new wins vs 16 regressions. Both-lose drops from 120 to 89 — 
31 previously-unsolvable games now solved.

## Experiment 3: Feedback Consistency

Does the model guess a word consistent with all prior feedback? (i.e., the guess
is in the filtered candidate set.)

| Turn | Pretrained | Phase 1 | Phase 2 | Phase 2+Opener |
|------|-----------|---------|---------|----------------|
| 1 | 100% | 100% | 100% | 100% |
| 2 | 56% | 65% | 62% | 64% |
| 3 | 33% | 20% | 37% | 39% |
| 4 | 20% | 16% | 24% | 25% |
| 5 | 13% | 9% | 17% | 19% |
| 6 | 7% | 7% | 14% | 12% |

Phase 2 improved consistency at every turn from 3 onward compared to pretrained.
At turn 6: 14% vs 7% (doubled). But even the best model ignores feedback 75%+ of
the time by turn 4.

A 100% solve rate requires near-100% consistency by turn 3-4. We're at 25-39%.
This is the primary gap.

## Experiment 4: Per-Turn Quality

### Pretrained
| Turn | N | Exp IG | Cands Before → After | Avg Greens |
|------|---|--------|---------------------|------------|
| 1 | 200 | 5.46 | 2315 → 86 | 0.6 |
| 2 | 200 | 3.42 | 86 → 8 | 1.3 |
| 3 | 191 | 1.40 | 9 → 3 | 2.0 |
| 4 | 173 | 0.61 | 3 → 2 | 2.1 |
| 5 | 156 | 0.22 | 2 → 1 | 1.9 |
| 6 | 143 | 0.14 | 1 → 1 | 1.7 |

### Phase 2 + Opener
| Turn | N | Exp IG | Cands Before → After | Avg Greens |
|------|---|--------|---------------------|------------|
| 1 | 200 | 5.86 | 2315 → 71 | 0.6 |
| 2 | 200 | 3.63 | 71 → 5 | 1.6 |
| 3 | 193 | 1.18 | 5 → 2 | 2.3 |
| 4 | 171 | 0.40 | 2 → 1 | 2.6 |
| 5 | 143 | 0.16 | 1 → 1 | 2.5 |
| 6 | 119 | 0.05 | 1 → 1 | 2.4 |

Phase 2+opener narrows to ~2 candidates by turn 3, ~1 by turn 4. But then it
can't close — 143 games reach turn 5 and 119 reach turn 6 with 1 candidate
remaining. The model has narrowed the answer but doesn't guess it.

## Experiment 5: First Guess Analysis

| Model | Unique Guesses | Avg Info Gain | Top 3 |
|-------|---------------|---------------|-------|
| pretrained | 15 | 5.46 bits | spare(55), stare(53), stone(26) |
| phase1 | 1 | 5.86 bits | slate(200) |
| phase2 | 15 | 5.25 bits | spine(107), store(47), spare(15) |

Phase 1 learned "always play slate" — optimal for info gain but causes repetition
problems (slate appears again in later turns). Pretrained near-greedy already
concentrates on a few good openers.

## Experiment 6: Guess Repetition

| Model | Games with Repeats | Rate |
|-------|--------------------|------|
| pretrained | 43/200 | 21.5% |
| phase1 | 93/200 | 46.5% |
| phase2 | 47/200 | 23.5% |
| phase2+opener | 36/200 | 18.0% |

Phase 1 repeats guesses in nearly half its games — the model re-guesses "slate"
or other learned words in later turns. Phase 2+opener has the lowest repetition.

Repeated guesses waste turns and guarantee a loss. At 18% this costs ~36 games
that could potentially be won.

## Where the 52.5% of losses come from

Of 200 games, 105 are losses under the best configuration (phase2+opener). 
Breaking down by what goes wrong:

1. **Consistency failure (primary).** By turn 4, the model has ~1-2 candidates
   remaining but only guesses from that set 25% of the time. The other 75% of
   guesses waste turns on words that can't be the answer.

2. **Can't close with 1 candidate.** 143 games reach turn 5, 119 reach turn 6,
   almost all with 1 candidate. The model doesn't know what the remaining word
   is — it sees feedback tokens but can't deduce the answer.

3. **Repetition.** 18% of games have repeated guesses, wasting turns.

## What we know

1. RL training improved win rate from 32.5% to 47.5% (with two-model pipeline).
2. Phase 2 specifically learned better turns 3-6 play (+13.5pp over pretrained with same opener).
3. Phase 2 improved feedback consistency (doubled at turn 6, +4-7pp at turns 3-5).
4. The model narrows candidates effectively (1-2 by turn 4) but can't guess the remaining word.
5. Repetition wastes turns in 18-47% of games depending on model.

## What we don't know

1. Whether more Phase 2 training would continue improving consistency.
2. Whether a larger model would learn feedback usage faster or better.
3. Whether explicit feedback encoding (giving the model constraint state as input) would help.
4. Whether a repetition penalty in decoding or reward would recover the 18% repeated-guess losses.
5. Whether training on full games (no curriculum split) with the composite reward would work.
6. Whether supervised fine-tuning on correct late-game decisions (before RL) would bootstrap
   consistency high enough for RL to finish the job.

## Raw Data

| File | Contents |
|------|----------|
| `runs/investigation/r3_games.json` | Full game-level data for all 4 configurations |
| `runs/investigation/r3_analysis.json` | Aggregated analysis (consistency, per-turn, comparisons) |
| `runs/investigation/r2_results.json` | Temperature comparison experiment |
| `wordle/investigate_rl.py` | R1 investigation script |
| `wordle/investigate_rl_r2.py` | R2 temperature experiment |
| `wordle/investigate_rl_r3.py` | R3 detailed near-greedy analysis |
