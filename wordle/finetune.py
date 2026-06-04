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
import signal
import time
from collections import deque

import torch
from config import FinetuneConfig
from decoding import compute_guess_log_probs, sample_constrained, sample_unconstrained
from grpo_train import (
    collect_grpo_step_data,
)
from mm_grpo import MovingAverageBaseline
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from mm_tokenizers import CharTokenizer
from mm_training import (
    MetricsLogger,
    RunManifest,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from mm_viz import EvalSnapshot, GameReplay
from mm_wordle import (
    WordleEnv,
    WordTrie,
    compute_reward,
    game_state_to_prompt,
    load_answers,
)
from mm_wordle.solver import filter_candidates
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to fine-tuning checkpoint directory to resume from",
    )
    parser.add_argument(
        "--opener",
        type=str,
        default=None,
        help="Path to Phase 1 opener model checkpoint (Phase 2 only)",
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


def play_game_reinforce(
    model: GPT,
    env: WordleEnv,
    target_word: str,
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
) -> tuple[list[Tensor], list[float], GameReplay]:
    """Play a complete Wordle game, collecting log probs and rewards per turn."""
    state = env.reset(target_word)
    turn_log_probs: list[Tensor] = []
    turn_rewards: list[float] = []
    replay_guesses: list[str] = []
    replay_feedback: list[list[str]] = []
    candidates = list(answers)

    while not state.solved and not state.failed:
        state_tokens = game_state_to_prompt(state)
        state_ids = torch.tensor(tokenizer.encode("".join(state_tokens)), dtype=torch.long, device=device)

        if constrained:
            samples = sample_constrained(model, state_ids, trie, tokenizer, device)
        else:
            samples = sample_unconstrained(model, state_ids, device, tokenizer)

        guess, word_ids = samples[0]

        lp = compute_guess_log_probs(model, state_ids, word_ids)
        turn_log_probs.append(lp)

        new_state, _done = env.step(state, guess)
        feedback = new_state.guesses[-1].feedback if new_state.guesses else []

        reward, _, _ = compute_reward(guess, feedback, candidates)
        turn_rewards.append(reward)

        candidates = filter_candidates(candidates, guess, feedback)

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


@torch.no_grad()
def evaluate_games(
    model: GPT,
    env: WordleEnv,
    eval_words: list[str],
    tokenizer: CharTokenizer,
    answers: list[str],
    trie: WordTrie,
    device: torch.device,
    constrained: bool,
    temperature: float = 0.1,
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
            state_tokens = game_state_to_prompt(state)
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
                    temperature=temperature,
                )
            else:
                samples = sample_unconstrained(model, state_ids, device, tokenizer, temperature=temperature)

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
    from mm_wordle import all_valid_words

    words = load_answers() if action_space == "answers" else sorted(all_valid_words())
    return WordTrie.from_words(words)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _resolve_resume_dir(resume_path: str) -> pathlib.Path:
    p = pathlib.Path(resume_path)
    return p if p.is_dir() else p.parent


def train(
    config: FinetuneConfig,
    checkpoint_path: str,
    resume_path: str | None = None,
    opener_checkpoint: str | None = None,
) -> None:
    """Run the RL fine-tuning loop."""
    rl_cfg = config.rl
    seed = rl_cfg.seed
    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()

    # Algorithm and decoding mode
    algorithm = rl_cfg.algorithm
    constrained = rl_cfg.decoding == "constrained"

    # Load model — either from resume checkpoint or pre-trained checkpoint
    start_step = 0
    if resume_path is not None:
        resume_dir = pathlib.Path(resume_path)
        ckpt_file = resume_dir / "model.pt" if resume_dir.is_dir() else resume_dir
        print(f"Resuming from {ckpt_file}")
        model = load_pretrained_model(ckpt_file, device)
    else:
        ckpt_path = pathlib.Path(checkpoint_path)
        print(f"Loading pre-trained model from {ckpt_path}")
        model = load_pretrained_model(ckpt_path, device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # Create or load reference model
    if resume_path is not None:
        resume_dir = _resolve_resume_dir(resume_path)
        ref_path = resume_dir / "ref_model.pt"
        if ref_path.exists():
            ref_model = copy.deepcopy(model)
            ref_model.load_state_dict(torch.load(ref_path, map_location=device, weights_only=True))
            for param in ref_model.parameters():
                param.requires_grad = False
            ref_model.eval()
            print("Loaded frozen reference model from checkpoint")
        else:
            ref_model = create_reference_model(model)
            print("Created frozen reference model (no ref checkpoint found)")
    else:
        ref_model = create_reference_model(model)
        print("Created frozen reference model")

    # Load opener model for Phase 2
    opener_model = None
    if rl_cfg.curriculum_phase == 2 and opener_checkpoint:
        opener_path = pathlib.Path(opener_checkpoint)
        print(f"Loading opener model from {opener_path}")
        opener_model = load_pretrained_model(opener_path, device)
        opener_model.eval()
        for param in opener_model.parameters():
            param.requires_grad = False
        print("Opener model loaded (frozen)")

    print(f"Algorithm: {algorithm}, Decoding: {rl_cfg.decoding}")
    if rl_cfg.curriculum_phase > 0:
        print(f"Curriculum phase: {rl_cfg.curriculum_phase}, max_turns: {rl_cfg.max_turns}")

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

    # Restore optimizer, step, and RNG state on resume
    if resume_path is not None:
        resume_dir = _resolve_resume_dir(resume_path)
        ckpt_file = resume_dir / "model.pt"
        checkpoint = load_checkpoint(ckpt_file, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        rng = checkpoint.get("rng_states", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu())
        if "random" in rng:
            random.setstate(rng["random"])
        if "numpy" in rng:
            import numpy as np

            np.random.set_state(rng["numpy"])
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed at step {start_step}")

    # Reward config
    # Game environment
    env = WordleEnv()
    answers = load_answers()

    # Build trie for constrained decoding
    print("Building word trie for constrained decoding...")
    word_trie = build_word_trie(rl_cfg.action_space)
    char_to_id = {chr(ord("a") + i): i for i in range(26)}
    word_trie.build_gpu_masks(tokenizer.vocab_size, char_to_id, device)
    print(f"  Action space: {rl_cfg.action_space}")

    # Precompute expected info gain for turn 1
    from mm_wordle.reward import precompute_expected_info_gain

    print("Precomputing expected info gain for all answer words...")
    precompute_expected_info_gain()
    print("  Done")

    # Fixed evaluation set — reuse from previous run on resume
    max_eval_games = rl_cfg.max_eval_games
    if resume_path is not None:
        resume_dir = _resolve_resume_dir(resume_path)
        prev_eval_path = resume_dir.parent / "eval_words.json"
        if prev_eval_path.exists():
            eval_words = json.loads(prev_eval_path.read_text())
            print(f"Loaded evaluation set from {prev_eval_path}: {len(eval_words)} words")
        else:
            eval_words = random.sample(answers, min(max_eval_games, len(answers)))
            print(f"Fixed evaluation set: {len(eval_words)} words")
    else:
        eval_words = random.sample(answers, min(max_eval_games, len(answers)))
        print(f"Fixed evaluation set: {len(eval_words)} words")

    # Metrics logger — reuse directory on resume
    if resume_path is not None:
        resume_dir = _resolve_resume_dir(resume_path)
        logger = MetricsLogger(experiment=f"finetune-{algorithm}", run_dir=resume_dir.parent)
    else:
        logger = MetricsLogger(experiment=f"finetune-{algorithm}")
    print(f"Logging to {logger.log_dir}")

    # Save evaluation words for reproducibility
    eval_words_path = logger.log_dir / "eval_words.json"
    if not eval_words_path.exists():
        eval_words_path.write_text(json.dumps(eval_words, indent=2))

    # Run manifest (only on fresh runs)
    manifest_path = logger.log_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = RunManifest.capture(
            experiment=f"finetune-{algorithm}",
            config=config.model_dump(),
            seed=seed,
            dataset_id="wordle-answers",
        )
        manifest.save(manifest_path)

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
    recent_info_gain: deque[float] = deque(maxlen=200)
    recent_candidates_remaining: deque[int] = deque(maxlen=100)

    # REINFORCE baseline
    baseline = MovingAverageBaseline(momentum=baseline_momentum)

    model.train()
    t_start = time.time()

    print(f"\nStarting RL fine-tuning from step {start_step + 1} to {max_steps}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Decoding: {'constrained' if constrained else 'unconstrained'}")
    print(f"  Batch size: {batch_size}")
    if algorithm == "grpo":
        print(f"  Group size: {group_size}")
    print()

    from rl_steps import grpo_step, reinforce_step

    for step in range(start_step + 1, max_steps + 1):
        batch_targets = random.choices(answers, k=batch_size)

        if algorithm == "reinforce":
            step_metrics = reinforce_step(
                model=model,
                env=env,
                batch_targets=batch_targets,
                tokenizer=tokenizer,
                answers=answers,
                trie=word_trie,
                device=device,
                constrained=constrained,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                baseline=baseline,
                logger=logger,
                step=step,
            )
            last_loss = step_metrics["loss"]
            lr = step_metrics["lr"]

        else:
            step_metrics, batch_replays, batch_rewards, batch_turn_details = grpo_step(
                model=model,
                ref_model=ref_model,
                opener_model=opener_model,
                env=env,
                batch_targets=batch_targets,
                tokenizer=tokenizer,
                answers=answers,
                trie=word_trie,
                device=device,
                constrained=constrained,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=grad_clip,
                group_size=group_size,
                clip_epsilon=clip_epsilon,
                kl_beta=kl_beta,
                ppo_epochs=rl_cfg.ppo_epochs,
                curriculum_phase=rl_cfg.curriculum_phase,
                max_turns=rl_cfg.max_turns,
                logger=logger,
                log_dir=logger.log_dir,
                step=step,
            )
            if not step_metrics:
                continue

            last_loss = step_metrics["loss"]
            lr = step_metrics["lr"]

            for replay in batch_replays:
                recent_wins.append(replay.solved)
                recent_guesses.append(replay.turns)
                for g in replay.guesses:
                    recent_valid.append(g in set(answers))

            for td_list in batch_turn_details:
                for td in td_list:
                    cd = next((gd for gd in td["group"] if gd["guess"] == td["chosen"]), None)
                    if cd:
                        recent_info_gain.append(cd["actual"])
                if td_list:
                    last_td = td_list[-1]
                    last_cd = next((gd for gd in last_td["group"] if gd["guess"] == last_td["chosen"]), None)
                    if last_cd:
                        n_before = last_td["candidates"]
                        actual = last_cd["actual"]
                        n_after = max(int(n_before / (2**actual)), 1) if actual > 0 else n_before
                        recent_candidates_remaining.append(n_after)

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
        if recent_info_gain:
            logger.log_scalar("train/avg_info_gain", sum(recent_info_gain) / len(recent_info_gain), step)
        if recent_candidates_remaining:
            avg_cands = sum(recent_candidates_remaining) / len(recent_candidates_remaining)
            logger.log_scalar("train/avg_candidates_left", avg_cands, step)

        # Print progress
        elapsed = time.time() - t_start
        if step % 10 == 0 or step == 1:
            wr = sum(recent_wins) / max(len(recent_wins), 1)
            ag = sum(recent_guesses) / max(len(recent_guesses), 1)
            avg_ig = sum(recent_info_gain) / max(len(recent_info_gain), 1)
            avg_cr = sum(recent_candidates_remaining) / max(len(recent_candidates_remaining), 1)

            if rl_cfg.curriculum_phase == 1:
                print(
                    f"step {step:>5d}/{max_steps} | "
                    f"loss {last_loss:.4f} | "
                    f"info_gain {avg_ig:.2f} | "
                    f"candidates_left {avg_cr:.0f} | "
                    f"lr {lr:.2e} | "
                    f"elapsed {elapsed:.0f}s"
                )
            else:
                print(
                    f"step {step:>5d}/{max_steps} | "
                    f"loss {last_loss:.4f} | "
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
                answers=answers,
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
        if algorithm == "grpo":
            viz_target = random.choice(answers)
            step_data = collect_grpo_step_data(
                model=model,
                ref_model=ref_model,
                env=env,
                target_word=viz_target,
                tokenizer=tokenizer,
                answers=answers,
                trie=word_trie,
                device=device,
                group_size=group_size,
                step=step,
                constrained=constrained,
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
        answers=answers,
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
    train(config, checkpoint_path=args.checkpoint, resume_path=args.resume, opener_checkpoint=args.opener)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
