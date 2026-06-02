# Pre-Training Evaluation Criteria

Before moving to RL fine-tuning, the pre-trained model must pass these criteria. The purpose of pre-training is to give the model a foundation that RL can build on — a diverse distribution over valid Wordle words and an understanding of the game transcript format.

## Positive Criteria (must pass)

### 1. Generates valid Wordle words from `[bos]`

The turn-1 RL prompt is `[bos]`. The model must be able to generate valid 5-letter words from this prompt.

**Test:** Generate 100 five-character sequences from `[bos]` using trie-constrained decoding. At least 30% should be words from the answer list.

**Why:** If the model can't produce valid words from the start-of-game prompt, RL has no useful behavior to reinforce.

### 2. Diverse word generation

The model must not collapse to a single word or a handful of words.

**Test:** Generate 100 words from `[bos]`. At least 20 unique words must appear. No single word accounts for more than 10% of generations.

**Why:** GRPO needs diversity in the group to compute meaningful advantages. If all group samples are the same word, advantages are zero and the model learns nothing.

### 3. Understands the game transcript format

The model must have learned that feedback tokens follow guess letters, and that letters follow `[sep]`.

**Test:** Given the prompt `[bos] s l a t e`, check what the model predicts next. The top predictions should be feedback tokens (`[green]`, `[yellow]`, `[gray]`), not letters.

**Test:** Given the prompt `[bos] s l a t e [gray] [gray] [green] [gray] [green] [sep]`, check what the model predicts next. The top predictions should be letters (a-z), not feedback tokens or special tokens.

**Why:** The model needs to understand the structure — letters come in groups of 5, followed by 5 feedback tokens, then `[sep]`, then more letters. Without this, the model won't produce coherent game play.

## Negative Criteria (must NOT happen)

### 4. No degenerate output from game prompts

Given a mid-game prompt with feedback tokens, the model should not produce gibberish or repeat the same letter.

**Test:** Given 10 different mid-game prompts (varying feedback patterns), generate words using trie-constrained decoding. At least 80% should be valid 5-letter words.

### 5. No single-word dominance

**Test:** Generate 100 words from 10 different game state prompts (1000 total). No single word accounts for more than 5% of all generations across all prompts.

**Why:** If the model always guesses the same word regardless of game state, it hasn't learned to use feedback.

### 6. Loss has plateaued

**Test:** The validation loss has stopped improving (less than 1% change over the last 20% of training steps).

**Why:** Continuing pre-training past convergence wastes time without benefit. The model has extracted what it can from the data.

## Running the Evaluation

These checks should run automatically at the end of pre-training before declaring the model ready for RL. If any positive criterion fails, the model needs more or different training data. If any negative criterion triggers, the model has a degenerate distribution that RL cannot fix.
