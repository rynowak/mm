# Wordle Exercise

Train a small GPT to play Wordle: pre-train on text, then RL fine-tune with REINFORCE and GRPO.

See [docs/prd.md](../docs/prd.md) for goals and [docs/architecture.md](../docs/architecture.md) for design.

## Scripts

- `pretrain.py` — Pre-train a character-level GPT on TinyStories + word lists
- `finetune.py` — RL fine-tune with REINFORCE/GRPO against the Wordle environment
- `evaluate.py` — Evaluate checkpoints and generate visualizations

## Configs

- `configs/small.yaml` — ~5M param model (6L/6H/256d)
- `configs/medium.yaml` — ~10M param model (6L/6H/384d)
