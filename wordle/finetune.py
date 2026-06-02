"""RL fine-tune a pre-trained GPT to play Wordle using REINFORCE or GRPO.

Usage:
    uv run python wordle/finetune.py --config wordle/configs/finetune.yaml \
        --checkpoint runs/pretrain-small-.../checkpoint-final/model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import time
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from config import FinetuneConfig
from mm_grpo import (
    MovingAverageBaseline,
    build_step_data,
    compute_group_advantages,
    grpo_loss,
    reinforce_loss,
    sequence_log_probs,
)
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from mm_tokenizers import CharTokenizer
from mm_training import (
    MetricsLogger,
    RunManifest,
    clip_grad_norm,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from mm_viz import EvalSnapshot, GameReplay, GRPOStepData
from mm_wordle import (
    RewardConfig,
    WordleEnv,
    WordTrie,
    all_valid_words,
    compute_reward,
    game_state_to_tokens,
    load_answers,
)
from torch import Tensor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RL fine-tune a pre-trained GPT to play Wordle")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint (.pt file)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pretrained_model(checkpoint_path: pathlib.Path, device: torch.device) -> GPT:
    """Load a pre-trained GPT model from a checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    config_dict = checkpoint["config"]
    model_config = GPTConfig(**config_dict)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


def create_reference_model(model: GPT) -> GPT:
    """Create a frozen copy of the model to serve as the reference policy."""
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    return ref_model


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------


def build_reward_config(config: FinetuneConfig) -> RewardConfig:
    """Build a RewardConfig from the parsed config."""
    rc = config.reward
    return RewardConfig(
        invalid_word=rc.invalid_word,
        repeated_guess=rc.repeated_guess,
        contradicts_clues=rc.contradicts_clues,
        no_new_info=rc.no_new_info,
        green_letter=rc.green_letter,
        yellow_letter=rc.yellow_letter,
        solved=rc.solved,
        failed=rc.failed,
    )


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------


def sample_constrained(
    model: GPT,
    game_state_ids: Tensor,
    trie: WordTrie,
    tokenizer: CharTokenizer,
    device: torch.device,
    n_samples: int = 1,
    temperature: float = 1.0,
) -> list[tuple[str, Tensor]]:
    """Sample word(s) using trie-constrained autoregressive decoding.

    At each of the 5 character positions, mask the model's output logits
    to only allow characters that continue a valid word in the trie.
    5 forward passes per sample regardless of word list size.
    """
    was_training = model.training
    model.eval()

    results: list[tuple[str, Tensor]] = []
    for _ in range(n_samples):
        idx = game_state_ids.unsqueeze(0).to(device)  # (1, prompt_len)
        prefix = ""

        for _pos in range(5):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Mask to only valid next characters from the trie
            valid_chars = trie.valid_next_chars(prefix)
            if not valid_chars:
                break

            mask = torch.full_like(logits, float("-inf"))
            for ch in valid_chars:
                token_ids = tokenizer.encode(ch)
                if token_ids:
                    mask[0, token_ids[0]] = 0.0
            logits = logits + mask

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            try:
                ch = tokenizer.decode([next_token.item()])
                prefix += ch
            except ValueError:
                prefix += "?"

        word_ids = idx[0, -5:]
        word = prefix.ljust(5, "a")[:5]
        results.append((word, word_ids))

    if was_training:
        model.train()

    return results


# ---------------------------------------------------------------------------
# Unconstrained decoding
# ---------------------------------------------------------------------------


def sample_unconstrained(
    model: GPT,
    game_state_ids: Tensor,
    device: torch.device,
    tokenizer: CharTokenizer,
    n_samples: int = 1,
    temperature: float = 0.8,
) -> list[tuple[str, Tensor]]:
    """Generate word(s) autoregressively (5 characters each).

    Args:
        model: The GPT model.
        game_state_ids: (prompt_len,) token IDs for current game state.
        device: Device to run on.
        tokenizer: Tokenizer for decoding.
        n_samples: Number of words to generate.
        temperature: Sampling temperature.

    Returns:
        List of (word, word_token_ids) tuples.
    """
    was_training = model.training
    model.eval()

    results: list[tuple[str, Tensor]] = []
    prompt = game_state_ids.unsqueeze(0).to(device)  # (1, prompt_len)

    for _ in range(n_samples):
        output = model.generate(prompt, max_new_tokens=5, temperature=temperature)
        # Extract the 5 generated tokens
        word_ids = output[0, -5:]  # (5,)
        try:
            word = tokenizer.decode(word_ids.tolist())
        except ValueError:
            word = "?????"  # Invalid tokens, will get invalid_word penalty
        results.append((word, word_ids))

    if was_training:
        model.train()

    return results


# ---------------------------------------------------------------------------
# Compute log probs for a guess under a model
# ---------------------------------------------------------------------------


def compute_guess_log_probs(
    model: GPT,
    game_state_ids: Tensor,
    word_ids: Tensor,
) -> Tensor:
    """Compute per-token log probs of a guess given a game state.

    Args:
        model: The GPT model.
        game_state_ids: (prompt_len,) token IDs.
        word_ids: (5,) token IDs for the guess.

    Returns:
        (5,) log probabilities for each character of the guess.
    """
    prompt_len = game_state_ids.shape[0]

    # Build full sequence: (1, prompt_len + 5)
    full_seq = torch.cat([game_state_ids, word_ids]).unsqueeze(0)

    logits, _ = model(full_seq)  # (1, total_len, vocab_size)

    # Extract logits predicting the word characters
    completion_logits = logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]  # (1, 5, vocab)

    # Compute per-token log probs
    lp = sequence_log_probs(completion_logits, word_ids.unsqueeze(0))  # (1, 5)
    return lp.squeeze(0)  # (5,)


# ---------------------------------------------------------------------------
# Play a single game
# ---------------------------------------------------------------------------


def play_game_reinforce(
    model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    reward_config: RewardConfig,
    valid_words: set[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
) -> tuple[list[Tensor], list[float], GameReplay]:
    """Play a complete Wordle game, collecting log probs and rewards per turn.

    Returns:
        (turn_log_probs, turn_rewards, replay)
        - turn_log_probs: list of (5,) log prob tensors, one per turn
        - turn_rewards: list of scalar rewards, one per turn
        - replay: GameReplay for visualization
    """
    state = env.reset(target_word)
    turn_log_probs: list[Tensor] = []
    turn_rewards: list[float] = []
    replay_guesses: list[str] = []
    replay_feedback: list[list[str]] = []

    while not state.solved and not state.failed:
        # Encode game state to tokens
        state_tokens = game_state_to_tokens(state)
        state_ids = torch.tensor(tokenizer.encode("".join(state_tokens)), dtype=torch.long, device=device)

        # Sample a guess
        if constrained:
            samples = sample_constrained(model, state_ids, trie, tokenizer, device)
        else:
            samples = sample_unconstrained(model, state_ids, device, tokenizer)

        guess, word_ids = samples[0]

        # Compute log probs under current policy (with gradients)
        lp = compute_guess_log_probs(model, state_ids, word_ids)
        turn_log_probs.append(lp)

        # Play the guess
        new_state, _done = env.step(state, guess)

        # Get feedback for reward computation
        feedback = new_state.guesses[-1].feedback if new_state.guesses else []

        reward = compute_reward(new_state, guess, feedback, valid_words, reward_config)
        turn_rewards.append(reward)

        # Record for replay
        replay_guesses.append(guess)
        replay_feedback.append([fb.value for fb in feedback])

        state = new_state

    replay = GameReplay(
        target=target_word,
        guesses=replay_guesses,
        feedback=replay_feedback,
        solved=state.solved,
        turns=state.turn,
    )

    return turn_log_probs, turn_rewards, replay


@dataclass
class TurnExperience:
    """Collected experience from one turn of a GRPO game."""

    state_ids: Tensor
    word_ids_batch: Tensor
    rewards_tensor: Tensor
    ref_log_probs: Tensor
    old_log_probs: Tensor


def collect_game_experience(
    model: GPT,
    ref_model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    reward_config: RewardConfig,
    valid_words: set[str],
    trie: WordTrie,
    device: torch.device,
    group_size: int,
    constrained: bool,
) -> tuple[list[TurnExperience], GameReplay, float]:
    """Play a Wordle game and collect experience for GRPO optimization.

    Sampling and reward computation happen here (no gradients needed).
    Returns the collected experience, replay, and total reward.
    """
    state = env.reset(target_word)
    experiences: list[TurnExperience] = []
    replay_guesses: list[str] = []
    replay_feedback: list[list[str]] = []
    total_reward = 0.0

    while not state.solved and not state.failed:
        state_tokens = game_state_to_tokens(state)
        state_ids = torch.tensor(tokenizer.encode("".join(state_tokens)), dtype=torch.long, device=device)

        if constrained:
            samples = sample_constrained(model, state_ids, trie, tokenizer, device, n_samples=group_size)
        else:
            samples = sample_unconstrained(model, state_ids, device, tokenizer, n_samples=group_size)

        guesses = [s[0] for s in samples]
        word_ids_list = [s[1] for s in samples]
        word_ids_batch = torch.stack(word_ids_list)

        rewards: list[float] = []
        for guess in guesses:
            sim_state, _ = env.step(state, guess)
            fb = sim_state.guesses[-1].feedback if sim_state.guesses else []
            r = compute_reward(sim_state, guess, fb, valid_words, reward_config)
            rewards.append(r)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

        prompt_expanded = state_ids.unsqueeze(0).expand(group_size, -1)
        full_sequences = torch.cat([prompt_expanded, word_ids_batch], dim=-1)
        prompt_len = state_ids.shape[0]

        with torch.no_grad():
            old_logits, _ = model(full_sequences)
            old_completion_logits = old_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
            old_lp = sequence_log_probs(old_completion_logits, word_ids_batch)

            ref_logits, _ = ref_model(full_sequences)
            ref_completion_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
            ref_lp = sequence_log_probs(ref_completion_logits, word_ids_batch)

        experiences.append(
            TurnExperience(
                state_ids=state_ids,
                word_ids_batch=word_ids_batch,
                rewards_tensor=rewards_tensor,
                ref_log_probs=ref_lp,
                old_log_probs=old_lp,
            )
        )

        best_idx = rewards_tensor.argmax().item()
        chosen_guess = guesses[best_idx]
        total_reward += rewards[best_idx]

        new_state, _done = env.step(state, chosen_guess)
        feedback = new_state.guesses[-1].feedback if new_state.guesses else []
        replay_guesses.append(chosen_guess)
        replay_feedback.append([fb.value for fb in feedback])
        state = new_state

    replay = GameReplay(
        target=target_word,
        guesses=replay_guesses,
        feedback=replay_feedback,
        solved=state.solved,
        turns=state.turn,
    )
    return experiences, replay, total_reward


def compute_grpo_loss(
    model: GPT,
    experiences: list[TurnExperience],
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute GRPO loss from collected experience.

    This is called AFTER sampling. Because the model weights may have been
    updated since sampling (via multiple optimization steps on the same batch),
    current_log_probs can differ from old_log_probs, making clipping active.
    """
    total_loss = torch.tensor(0.0, device=experiences[0].state_ids.device)
    metrics_accum: dict[str, list[float]] = {
        "policy_loss": [],
        "kl_div": [],
        "entropy": [],
        "clip_fraction": [],
    }

    for exp in experiences:
        prompt_expanded = exp.state_ids.unsqueeze(0).expand(exp.word_ids_batch.shape[0], -1)
        full_sequences = torch.cat([prompt_expanded, exp.word_ids_batch], dim=-1)
        prompt_len = exp.state_ids.shape[0]

        logits, _ = model(full_sequences)
        completion_logits = logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        current_log_probs = sequence_log_probs(completion_logits, exp.word_ids_batch)

        turn_loss, turn_metrics = grpo_loss(
            log_probs=current_log_probs,
            old_log_probs=exp.old_log_probs,
            rewards=exp.rewards_tensor,
            ref_log_probs=exp.ref_log_probs,
            clip_epsilon=clip_epsilon,
            beta=kl_beta,
        )
        total_loss = total_loss + turn_loss
        for key in metrics_accum:
            if key in turn_metrics:
                metrics_accum[key].append(turn_metrics[key])

    if experiences:
        total_loss = total_loss / len(experiences)

    aggregated = {k: sum(v) / max(len(v), 1) for k, v in metrics_accum.items()}
    return total_loss, aggregated


# ---------------------------------------------------------------------------
# GRPO step data collection (for visualization)
# ---------------------------------------------------------------------------


def collect_grpo_step_data(
    model: GPT,
    ref_model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    reward_config: RewardConfig,
    valid_words: set[str],
    trie: WordTrie,
    device: torch.device,
    group_size: int,
    step: int,
) -> GRPOStepData | None:
    """Collect a GRPOStepData snapshot for one game turn (first turn only)."""
    state = env.reset(target_word)
    state_tokens = game_state_to_tokens(state)
    state_text = "".join(state_tokens)
    state_ids = torch.tensor(tokenizer.encode(state_text), dtype=torch.long, device=device)

    # Generate group_size candidates (constrained)
    samples = sample_constrained(
        model,
        state_ids,
        trie,
        tokenizer,
        device,
        n_samples=group_size,
    )

    guesses = [s[0] for s in samples]
    word_ids_list = [s[1] for s in samples]
    word_ids_batch = torch.stack(word_ids_list)

    # Score each guess
    rewards: list[float] = []
    reward_breakdowns: list[dict[str, float]] = []
    for guess in guesses:
        sim_state, _ = env.step(state, guess)
        fb = sim_state.guesses[-1].feedback if sim_state.guesses else []
        r = compute_reward(sim_state, guess, fb, valid_words, reward_config)
        rewards.append(r)
        # Build a simple breakdown
        reward_breakdowns.append({"total": r})

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = compute_group_advantages(rewards_tensor)

    # Compute log probs
    prompt_len = state_ids.shape[0]
    prompt_expanded = state_ids.unsqueeze(0).expand(group_size, -1)
    full_sequences = torch.cat([prompt_expanded, word_ids_batch], dim=-1)

    with torch.no_grad():
        logits_new, _ = model(full_sequences)
        completion_logits_new = logits_new[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        new_log_probs = sequence_log_probs(completion_logits_new, word_ids_batch)

        ref_logits, _ = ref_model(full_sequences)
        ref_completion_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
        ref_log_probs_t = sequence_log_probs(ref_completion_logits, word_ids_batch)

    # Compute KL
    kl_per_token = new_log_probs - ref_log_probs_t
    kl_div = kl_per_token.sum(dim=-1).mean().item()

    # Build completion data
    completion_texts = guesses
    completion_token_lists = [list(g) for g in guesses]
    log_probs_per_completion = new_log_probs.tolist()

    # Old probs and new probs (sequence-level)
    old_probs = ref_log_probs_t.sum(dim=-1).exp().tolist()
    new_probs = new_log_probs.sum(dim=-1).exp().tolist()

    return build_step_data(
        step=step,
        game_state_tokens=state_tokens,
        game_state_text=state_text,
        completion_texts=completion_texts,
        completion_token_lists=completion_token_lists,
        log_probs_per_completion=log_probs_per_completion,
        rewards=rewards,
        reward_breakdowns=reward_breakdowns,
        advantages=advantages.tolist(),
        group_mean=rewards_tensor.mean().item(),
        group_std=rewards_tensor.std().item(),
        old_probs=old_probs,
        new_probs=new_probs,
        kl_divergence=kl_div,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_games(
    model: GPT,
    env: WordleEnv,
    eval_words: list[str],
    tokenizer: CharTokenizer,
    reward_config: RewardConfig,
    valid_words: set[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
) -> tuple[float, float, list[GameReplay]]:
    """Play a set of evaluation games and return metrics.

    Returns:
        (win_rate, avg_guesses, replays)
    """
    was_training = model.training
    model.eval()

    wins = 0
    total_guesses = 0
    replays: list[GameReplay] = []

    for target in eval_words:
        state = env.reset(target)
        guess_list: list[str] = []
        feedback_list: list[list[str]] = []

        while not state.solved and not state.failed:
            state_tokens = game_state_to_tokens(state)
            state_ids = torch.tensor(
                tokenizer.encode("".join(state_tokens)),
                dtype=torch.long,
                device=device,
            )

            if constrained:
                samples = sample_constrained(
                    model,
                    state_ids,
                    trie,
                    tokenizer,
                    device,
                )
            else:
                samples = sample_unconstrained(model, state_ids, device, tokenizer)

            guess, _word_ids = samples[0]
            new_state, _done = env.step(state, guess)

            fb = new_state.guesses[-1].feedback if new_state.guesses else []

            guess_list.append(guess)
            feedback_list.append([f.value for f in fb])
            state = new_state

        if state.solved:
            wins += 1
        total_guesses += state.turn

        replays.append(
            GameReplay(
                target=target,
                guesses=guess_list,
                feedback=feedback_list,
                solved=state.solved,
                turns=state.turn,
            )
        )

    n_games = len(eval_words)
    win_rate = wins / max(n_games, 1)
    avg_guesses = total_guesses / max(n_games, 1)

    if was_training:
        model.train()

    return win_rate, avg_guesses, replays


# ---------------------------------------------------------------------------
# Precompute valid word token IDs
# ---------------------------------------------------------------------------


def build_word_trie(action_space: str) -> WordTrie:
    """Build a trie for constrained decoding."""
    words = load_answers() if action_space == "answers" else sorted(all_valid_words())
    return WordTrie.from_words(words)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: FinetuneConfig, checkpoint_path: str) -> None:
    """Run the RL fine-tuning loop."""
    rl_cfg = config.rl
    seed = rl_cfg.seed
    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()

    # Load pre-trained model
    ckpt_path = pathlib.Path(checkpoint_path)
    print(f"Loading pre-trained model from {ckpt_path}")
    model = load_pretrained_model(ckpt_path, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # Create reference model (frozen copy for KL penalty)
    ref_model = create_reference_model(model)
    print("Created frozen reference model")

    # Algorithm and decoding mode
    algorithm = rl_cfg.algorithm
    constrained = rl_cfg.decoding == "constrained"
    print(f"Algorithm: {algorithm}, Decoding: {rl_cfg.decoding}")

    # Optimizer and scheduler
    optimizer = create_optimizer(
        model,
        lr=rl_cfg.learning_rate,
        weight_decay=rl_cfg.weight_decay,
    )
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=rl_cfg.warmup_steps,
        total_steps=rl_cfg.max_steps,
    )

    # Reward config
    reward_config = build_reward_config(config)

    # Game environment
    env = WordleEnv()
    valid_words = all_valid_words()
    answers = load_answers()

    # Build trie for constrained decoding
    print("Building word trie for constrained decoding...")
    word_trie = build_word_trie(rl_cfg.action_space)
    print(f"  Action space: {rl_cfg.action_space}")

    # Fixed evaluation set
    max_eval_games = rl_cfg.max_eval_games
    eval_words = random.sample(answers, min(max_eval_games, len(answers)))
    print(f"Fixed evaluation set: {len(eval_words)} words")

    # Metrics logger
    logger = MetricsLogger(experiment=f"finetune-{algorithm}")
    print(f"Logging to {logger.log_dir}")

    # Save evaluation words for reproducibility
    eval_words_path = logger.log_dir / "eval_words.json"
    eval_words_path.write_text(json.dumps(eval_words, indent=2))

    # Run manifest
    manifest = RunManifest.capture(
        experiment=f"finetune-{algorithm}",
        config=config.model_dump(),
        seed=seed,
        dataset_id="wordle-answers",
    )
    manifest.save(logger.log_dir / "manifest.json")

    # Training state
    max_steps = rl_cfg.max_steps
    batch_size = rl_cfg.batch_size
    grad_clip = rl_cfg.grad_clip
    eval_interval = rl_cfg.eval_interval
    checkpoint_interval = rl_cfg.checkpoint_interval
    group_size = rl_cfg.group_size
    clip_epsilon = rl_cfg.clip_epsilon
    kl_beta = rl_cfg.kl_beta
    baseline_momentum = rl_cfg.baseline_momentum

    # Rolling metrics
    recent_wins: deque[bool] = deque(maxlen=100)
    recent_guesses: deque[int] = deque(maxlen=100)
    recent_valid: deque[bool] = deque(maxlen=100)

    # REINFORCE baseline
    baseline = MovingAverageBaseline(momentum=baseline_momentum)

    model.train()
    t_start = time.time()

    print(f"\nStarting RL fine-tuning from step 1 to {max_steps}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Decoding: {'constrained' if constrained else 'unconstrained'}")
    print(f"  Batch size: {batch_size}")
    if algorithm == "grpo":
        print(f"  Group size: {group_size}")
    print()

    for step in range(1, max_steps + 1):
        # Sample batch_size random target words
        batch_targets = random.choices(answers, k=batch_size)

        if algorithm == "reinforce":
            # --- REINFORCE training step ---
            all_log_probs: list[Tensor] = []
            all_rewards: list[float] = []

            for target in batch_targets:
                turn_lps, turn_rewards, replay = play_game_reinforce(
                    model=model,
                    env=env,
                    target_word=target,
                    tokenizer=tokenizer,
                    reward_config=reward_config,
                    valid_words=valid_words,
                    trie=word_trie,
                    device=device,
                    constrained=constrained,
                )

                # Total reward for the game
                total_reward = sum(turn_rewards)

                # Concatenate all turn log probs: (total_chars,)
                game_log_probs = torch.cat(turn_lps) if turn_lps else torch.zeros(1, device=device)

                all_log_probs.append(game_log_probs)
                all_rewards.append(total_reward)

                # Track rolling metrics
                recent_wins.append(replay.solved)
                recent_guesses.append(replay.turns)
                for g in replay.guesses:
                    recent_valid.append(g in valid_words)

            # Pad log probs to same length for batching
            max_len = max(lp.shape[0] for lp in all_log_probs)
            padded_lps = torch.zeros(batch_size, max_len, device=device)
            for i, lp in enumerate(all_log_probs):
                padded_lps[i, : lp.shape[0]] = lp

            rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=device)

            # Update baseline
            baseline.update(rewards_tensor.mean().item())
            baseline_val = torch.tensor(baseline.get(), dtype=torch.float32, device=device)

            # Compute REINFORCE loss
            loss = reinforce_loss(
                log_probs=padded_lps,
                rewards=rewards_tensor,
                baseline=baseline_val,
            )

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(model, grad_clip)
            optimizer.step()
            scheduler.step()

            # Logging
            lr = optimizer.param_groups[0]["lr"]
            reward_mean = rewards_tensor.mean().item()

            logger.log_scalar("train/loss", loss.item(), step)
            logger.log_scalar("train/reward_mean", reward_mean, step)
            logger.log_scalar("train/grad_norm", grad_norm, step)
            logger.log_scalar("train/lr", lr, step)

        else:
            # --- GRPO training step ---
            # Phase 1: Collect experience (sampling, no gradients needed)
            all_experiences: list[list[TurnExperience]] = []
            batch_rewards: list[float] = []

            for target in batch_targets:
                experiences, replay, game_reward = collect_game_experience(
                    model=model,
                    ref_model=ref_model,
                    env=env,
                    target_word=target,
                    tokenizer=tokenizer,
                    reward_config=reward_config,
                    valid_words=valid_words,
                    trie=word_trie,
                    device=device,
                    group_size=group_size,
                    constrained=constrained,
                )
                all_experiences.append(experiences)
                batch_rewards.append(game_reward)

                recent_wins.append(replay.solved)
                recent_guesses.append(replay.turns)
                for g in replay.guesses:
                    recent_valid.append(g in valid_words)

            # Phase 2: Multiple optimization epochs on the same batch (PPO-style)
            # old_log_probs are frozen from sampling; current_log_probs diverge
            # with each gradient step, making clipping active.
            ppo_epochs = rl_cfg.ppo_epochs
            last_metrics: dict[str, float] = {}
            last_loss = 0.0
            last_grad_norm = 0.0

            for _epoch in range(ppo_epochs):
                epoch_loss = torch.tensor(0.0, device=device)
                epoch_metrics: dict[str, list[float]] = {
                    "policy_loss": [],
                    "kl_div": [],
                    "entropy": [],
                    "clip_fraction": [],
                }

                for game_experiences in all_experiences:
                    game_loss, game_metrics = compute_grpo_loss(
                        model,
                        game_experiences,
                        clip_epsilon,
                        kl_beta,
                    )
                    epoch_loss = epoch_loss + game_loss
                    for key in epoch_metrics:
                        if key in game_metrics:
                            epoch_metrics[key].append(game_metrics[key])

                epoch_loss = epoch_loss / batch_size

                optimizer.zero_grad()
                epoch_loss.backward()
                last_grad_norm = clip_grad_norm(model, grad_clip)
                optimizer.step()

                last_loss = epoch_loss.item()
                last_metrics = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}

            scheduler.step()

            # Logging (from last epoch)
            lr = optimizer.param_groups[0]["lr"]
            reward_mean = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0

            logger.log_scalar("train/loss", last_loss, step)
            logger.log_scalar("train/reward_mean", reward_mean, step)
            logger.log_scalar("train/grad_norm", last_grad_norm, step)
            logger.log_scalar("train/lr", lr, step)

            for key, value in last_metrics.items():
                logger.log_scalar(f"train/{key}", value, step)

        # Rolling metrics
        if recent_wins:
            win_rate = sum(recent_wins) / len(recent_wins)
            logger.log_scalar("train/win_rate", win_rate, step)
        if recent_guesses:
            avg_guesses = sum(recent_guesses) / len(recent_guesses)
            logger.log_scalar("train/avg_guesses", avg_guesses, step)
        if recent_valid:
            valid_word_rate = sum(recent_valid) / len(recent_valid)
            logger.log_scalar("train/valid_word_rate", valid_word_rate, step)

        # Print progress
        elapsed = time.time() - t_start
        if step % 10 == 0 or step == 1:
            wr = sum(recent_wins) / max(len(recent_wins), 1)
            ag = sum(recent_guesses) / max(len(recent_guesses), 1)
            print(
                f"step {step:>5d}/{max_steps} | "
                f"loss {last_loss if algorithm == 'grpo' else loss.item():.4f} | "
                f"win_rate {wr:.2%} | "
                f"avg_guesses {ag:.1f} | "
                f"lr {lr:.2e} | "
                f"elapsed {elapsed:.0f}s"
            )

        # Evaluation
        if step % eval_interval == 0:
            print(f"\n  [eval] Running {len(eval_words)} evaluation games...")
            eval_win_rate, eval_avg_guesses, eval_replays = evaluate_games(
                model=model,
                env=env,
                eval_words=eval_words,
                tokenizer=tokenizer,
                reward_config=reward_config,
                valid_words=valid_words,
                trie=word_trie,
                device=device,
                constrained=constrained,
            )
            logger.log_scalar("eval/win_rate", eval_win_rate, step)
            logger.log_scalar("eval/avg_guesses", eval_avg_guesses, step)
            print(f"  [eval] win_rate: {eval_win_rate:.2%}, avg_guesses: {eval_avg_guesses:.1f}")

            # Save a few replays
            replay_dir = logger.log_dir / f"eval-{step}"
            replay_dir.mkdir(parents=True, exist_ok=True)
            for i, replay in enumerate(eval_replays[:5]):
                replay.save(replay_dir / f"replay-{i}.json")

            # Save eval snapshot
            snapshot = EvalSnapshot(
                step=step,
                checkpoint_path=str(ckpt_path),
                win_rate=eval_win_rate,
                avg_guesses=eval_avg_guesses,
                replays=eval_replays[:5],
            )
            snapshot.save(replay_dir / "snapshot.json")

        # Step data emission (GRPO visualization, every 100 steps)
        if algorithm == "grpo" and step % 100 == 0:
            viz_target = random.choice(answers)
            step_data = collect_grpo_step_data(
                model=model,
                ref_model=ref_model,
                env=env,
                target_word=viz_target,
                tokenizer=tokenizer,
                reward_config=reward_config,
                valid_words=valid_words,
                trie=word_trie,
                device=device,
                group_size=group_size,
                step=step,
            )
            if step_data is not None:
                step_data_dir = logger.log_dir / "step_data"
                step_data_dir.mkdir(parents=True, exist_ok=True)
                step_data.save(step_data_dir / f"step-{step}.json")

        # Checkpoint
        if step % checkpoint_interval == 0:
            ckpt_dir = logger.log_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir / "model.pt",
                model,
                optimizer,
                step,
                model.config,
            )
            # Save reference model weights alongside
            torch.save(
                ref_model.state_dict(),
                ckpt_dir / "ref_model.pt",
            )
            print(f"  [checkpoint] saved to {ckpt_dir}")

    # Final checkpoint
    ckpt_dir = logger.log_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        ckpt_dir / "model.pt",
        model,
        optimizer,
        step,
        model.config,
    )
    torch.save(ref_model.state_dict(), ckpt_dir / "ref_model.pt")
    print(f"\nTraining complete at step {step}")
    print(f"Final checkpoint: {ckpt_dir}")

    # Final evaluation
    eval_win_rate, eval_avg_guesses, _ = evaluate_games(
        model=model,
        env=env,
        eval_words=eval_words,
        tokenizer=tokenizer,
        reward_config=reward_config,
        valid_words=valid_words,
        trie=word_trie,
        device=device,
        constrained=constrained,
    )
    print(f"Final eval win_rate: {eval_win_rate:.2%}")
    print(f"Final eval avg_guesses: {eval_avg_guesses:.1f}")

    logger.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)
    train(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
