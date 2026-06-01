# PRD: mm — Model Training Playground

## Purpose

mm is a hands-on learning repo for model training. The goal is to build practical skills in pre-training and reinforcement learning by working through concrete exercises with small models that train on consumer hardware (laptop, Mac Studio).

This is not a framework or library. It's a collection of self-contained, well-documented training exercises that each produce a working model. Think of it as a lab notebook — each exercise stands alone, teaches a specific technique, and produces a measurable result.

## Workflow

Each exercise follows a phased delivery process:

1. **PRD** — What we want to achieve and why (this document for the overall repo; per-exercise PRDs as needed)
2. **Architecture/Design** — How we'll build it, what components, what tradeoffs
3. **Project Plan** — Concrete tasks, broken down for parallel execution
4. **Implementation** — Agents work in parallel on independent tasks
5. **Working code** — Runnable training scripts with reproducible results

## Hardware Constraints

- **Primary:** MacBook Pro (Apple Silicon) and Mac Studio
- **Implication:** Models must be small enough to train in reasonable time on MPS or CPU. Think 1M–50M parameters, not billions.
- **GPU cloud optional:** Exercises should work locally first. Cloud GPU is a stretch goal, not a requirement.

## Exercise 1: Wordle

Train a small language model to play Wordle competently. This exercise covers the full pipeline: pre-training a base model on text, then using reinforcement learning to teach it a specific skill.

### What is Wordle?

A word-guessing game. The player has 6 attempts to guess a hidden 5-letter word. After each guess, the game reveals:
- **Green:** correct letter in the correct position
- **Yellow:** correct letter in the wrong position
- **Gray:** letter not in the word

### Phase 1: Pre-train a small language model

Train a small GPT-style transformer from scratch on text data. The model should learn basic English language structure — vocabulary, letter patterns, word structure. This gives the model a foundation before RL fine-tuning.

**Success criteria:**
- Model trains to completion on a laptop/Mac Studio in reasonable time (hours, not days).
- Model generates coherent (if simple) English text.
- Training is fully reproducible (seeded, logged).
- Training loss curves and basic evaluation metrics are captured.

### Phase 2: RL fine-tuning (REINFORCE → GRPO)

Fine-tune the pre-trained model to play Wordle using reinforcement learning. Start with REINFORCE (simplest policy gradient) to learn the core concepts, then implement GRPO to understand what it adds. Explore both constrained decoding (model picks from valid word list) and unconstrained (model generates character-by-character).

**Success criteria:**
- A Wordle game environment that the model can interact with programmatically.
- The model plays Wordle competently — solves most standard Wordle puzzles within 6 guesses.
- Clear improvement visible between the base model, REINFORCE, and GRPO.
- Training metrics show learning progress (win rate, average guesses over time).
- Comparison between constrained and unconstrained decoding is documented.

### What "competent" means

The model should:
- Make valid 5-letter English word guesses (not random letter sequences).
- Use feedback from previous guesses (narrow down based on green/yellow/gray signals).
- Solve the majority of puzzles within the 6-guess limit.

We are not targeting optimal play or beating human averages. The goal is a model that clearly learned the task through RL.

## Non-Goals

- **Production deployment.** No serving infrastructure, APIs, or user-facing applications (for now).
- **Large-scale training.** No multi-node distributed training. Everything fits on one machine.
- **Novel research.** We're implementing known techniques to learn them, not inventing new ones.
- **Framework building for its own sake.** We will extract reusable pieces into libraries when they deepen understanding (tokenizers, data loaders, training loops, environments). But the goal is learning through building, not shipping a polished framework.

## Future Direction

The repo is structured so that new exercises can be added over time. Each exercise is a self-contained directory.

Planned areas to explore after the Wordle exercise:

- **Off-policy RL.** The Wordle exercise uses on-policy GRPO (sample, use once, discard). A follow-up exercise would implement off-policy methods: replay buffers, importance sampling corrections, and off-policy algorithms (e.g. SAC, off-policy PPO variants, DAPO). This teaches sample efficiency and the on-policy/off-policy tradeoff.
- **Simpler RL baselines.** Tabular Q-learning or other traditional RL approaches for comparison with the LM-based approach.
- **New exercises and domains** beyond Wordle — not planned yet.
