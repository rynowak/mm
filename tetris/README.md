# Tetris RL — Autoresearch

Train an RL agent to play Tetris using the [autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI coding agent iterates on `train.py` in a tight loop, keeping changes that improve the metric and reverting ones that don't.

## Environment

Standard 10x20 Tetris with placement-based actions — the agent picks a (rotation, column) pair and the piece drops to the lowest valid position. This simplifies the action space to 40 discrete actions compared to frame-by-frame key presses.

## Files

| File | Role |
|------|------|
| `prepare.py` | Run once — validates env, prints random baseline |
| `train.py` | **The autoresearch target** — agent edits this |
| `program.md` | Research instructions for the autoresearch agent |

## Quick start

```bash
# Validate environment and see random baseline
uv run python tetris/prepare.py

# Run one training iteration (5 minutes)
uv run python tetris/train.py
```

## Metric

`val_avg_lines` — average lines cleared per game over 100 evaluation games with a greedy policy. Random agent scores ~0-1; a well-trained agent should do significantly better.
