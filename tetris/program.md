# Tetris RL — Autoresearch Program

## Objective

Maximize `val_avg_lines` — the average number of lines cleared per game over 100 evaluation games.

## Setup

- **Environment:** 10x20 Tetris grid with placement-based actions (rotation, column). 40 discrete actions.
- **State:** 207-dimensional vector (200 grid cells + 7-dim piece one-hot).
- **Metric:** Average lines cleared per game. Higher is better.
- **Time budget:** 5 minutes of training per experiment.
- **Baseline:** Random agent scores ~0-1 lines per game.

## What you can modify

Everything in `train.py` is fair game:

- **Network architecture:** depth, width, activation functions, CNN layers, attention, etc.
- **Algorithm:** DQN, double DQN, dueling DQN, PPO, A2C, REINFORCE, etc.
- **Reward function:** line clear bonuses, height penalties, hole penalties, bumpiness, etc.
- **State representation:** add engineered features (heights, holes, bumpiness), normalize inputs, etc.
- **Hyperparameters:** learning rate, batch size, gamma, epsilon schedule, replay size, etc.
- **Training tricks:** prioritized replay, learning rate scheduling, gradient clipping, etc.

## What you should NOT modify

- The `mm_tetris` library (environment, pieces, game rules). Treat it as a fixed API.
- The evaluation protocol (100 games, seeds 10000-10099, greedy policy).
- The 5-minute training budget.
- The output format: `val_avg_lines: X.XXXX` on the last line.

## Research directions to explore

1. **Reward shaping:** The current reward is sparse (mostly from line clears). Adding dense shaping signals (penalize holes, reward flat surfaces, penalize height) could help the agent learn faster.

2. **State features:** The raw grid is high-dimensional. Adding computed features like column heights, hole counts, bumpiness, and wells could help the network focus on what matters.

3. **Architecture:** A CNN might capture spatial patterns in the grid better than an MLP. Or a hybrid approach with both grid features and computed features.

4. **Algorithm improvements:** Double DQN reduces overestimation. Dueling networks separate state value from action advantage. Prioritized replay focuses on surprising transitions.

5. **Exploration:** Epsilon-greedy is simple but might not explore efficiently. Try Boltzmann exploration, noisy networks, or intrinsic motivation.
