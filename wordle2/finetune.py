"""V2 RL fine-tuning with constraint-state encoding.

Usage:
    uv run python wordle2/finetune.py --config wordle2/configs/finetune-phase1.yaml \
        --checkpoint runs/pretrain-v2/.../model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mm_grpo import compute_group_advantages
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from mm_training import (
    MetricsLogger,
    RunManifest,
    clip_grad_norm,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from mm_wordle import WordleEnv, compute_reward, load_answers
from mm_wordle.reward import INVALID_WORD_PENALTY
from mm_wordle.solver import filter_candidates
from torch import Tensor

from wordle.config import FinetuneConfig
from wordle2.tokenizer import ConstraintTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 RL fine-tune")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--opener", type=str, default=None)
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> GPT:
    ckpt = load_checkpoint(path, device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def sample_guess(
    model: GPT,
    prompt_ids: Tensor,
    device: torch.device,
    letter_mask: Tensor,
    temperature: float = 1.0,
) -> tuple[str, Tensor]:
    """Generate one 5-letter guess. Returns (word, token_ids)."""
    si = prompt_ids.unsqueeze(0)
    generated: list[int] = []
    for _ in range(5):
        logits, _, _ = model(si)
        next_logits = logits[0, -1, :] / temperature
        next_logits = next_logits + letter_mask
        probs = F.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        si = torch.cat([si, torch.tensor([[next_id]], device=device)], dim=1)

    tok = ConstraintTokenizer()
    word = tok.decode_letters(generated)
    return word, torch.tensor(generated, dtype=torch.long, device=device)


def compute_guess_log_probs(
    model: GPT,
    prompt_ids: Tensor,
    word_ids: Tensor,
    letter_mask: Tensor,
) -> Tensor:
    """Compute log prob of a 5-letter guess given prompt. Returns scalar."""
    full_seq = torch.cat([prompt_ids, word_ids]).unsqueeze(0)
    logits, _, _ = model(full_seq)
    prompt_len = prompt_ids.shape[0]
    gen_logits = logits[0, prompt_len - 1 : prompt_len + 4, :]
    gen_logits = gen_logits + letter_mask.unsqueeze(0)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = log_probs.gather(1, word_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


@torch.no_grad()
def evaluate_games(
    model: GPT,
    tokenizer: ConstraintTokenizer,
    eval_words: list[str],
    device: torch.device,
    letter_mask: Tensor,
    temperature: float = 0.1,
) -> tuple[float, float]:
    model.eval()
    env = WordleEnv()
    wins = 0
    total_turns = 0

    for target in eval_words:
        state = env.reset(target_word=target)
        while not state.solved and not state.failed:
            prompt_ids = torch.tensor(tokenizer.encode_game_state(state), dtype=torch.long, device=device)
            guess, _ = sample_guess(model, prompt_ids, device, letter_mask, temperature)
            if len(guess) != 5:
                guess = "zzzzz"
            state, _ = env.step(state, guess)

        if state.solved:
            wins += 1
        total_turns += state.turn

    model.train()
    n = len(eval_words)
    return wins / max(n, 1), total_turns / max(n, 1)


def train(
    config: FinetuneConfig,
    checkpoint_path: str,
    opener_checkpoint: str | None = None,
) -> None:
    rl_cfg = config.rl
    seed_everything(rl_cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = ConstraintTokenizer()
    answers = load_answers()
    env = WordleEnv()

    letter_mask = torch.full((tokenizer.vocab_size,), float("-inf"), device=device)
    for lid in tokenizer.letter_ids:
        letter_mask[lid] = 0.0

    model = load_model(Path(checkpoint_path), device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    opener_model = None
    if rl_cfg.curriculum_phase == 2 and opener_checkpoint:
        opener_model = load_model(Path(opener_checkpoint), device)
        opener_model.eval()
        for p in opener_model.parameters():
            p.requires_grad = False
        print("Loaded opener model")

    optimizer = create_optimizer(model, lr=rl_cfg.learning_rate, weight_decay=rl_cfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=rl_cfg.warmup_steps, total_steps=rl_cfg.max_steps)

    from mm_wordle.reward import precompute_expected_info_gain

    print("Precomputing expected info gain...")
    precompute_expected_info_gain()

    eval_words = random.sample(answers, min(rl_cfg.max_eval_games, len(answers)))

    logger = MetricsLogger(experiment="finetune-v2")
    print(f"Logging to {logger.log_dir}")

    manifest = RunManifest.capture(
        experiment="finetune-v2",
        config=config.model_dump(),
        seed=rl_cfg.seed,
        dataset_id="wordle-answers",
    )
    manifest.save(logger.log_dir / "manifest.json")
    (logger.log_dir / "eval_words.json").write_text(json.dumps(eval_words))

    max_steps = rl_cfg.max_steps
    model.train()
    t_start = time.time()

    print(f"\nStarting V2 RL: {max_steps} steps, phase {rl_cfg.curriculum_phase}")
    print(f"  Group size: {rl_cfg.group_size}, Batch size: {rl_cfg.batch_size}\n")

    for step in range(1, max_steps + 1):
        batch_targets = random.choices(answers, k=rl_cfg.batch_size)

        all_prompt_ids: list[Tensor] = []
        all_word_ids: list[Tensor] = []
        all_rewards: list[Tensor] = []
        all_old_lp: list[Tensor] = []
        all_ref_lp: list[Tensor] = []
        batch_replays: list[dict] = []

        for target in batch_targets:
            state = env.reset(target_word=target)
            candidates = list(answers)
            replay_guesses: list[str] = []
            replay_feedback: list[list[str]] = []
            replay_rewards: list[float] = []

            if rl_cfg.curriculum_phase == 2 and opener_model is not None:
                for _ in range(2):
                    if state.solved or state.failed:
                        break
                    pi = torch.tensor(tokenizer.encode_game_state(state), dtype=torch.long, device=device)
                    guess, _ = sample_guess(opener_model, pi, device, letter_mask, 0.1)
                    if len(guess) != 5:
                        guess = "zzzzz"
                    state, _ = env.step(state, guess)
                    fb = state.guesses[-1].feedback
                    composite = rl_cfg.curriculum_phase != 1
                    r, _, _ = compute_reward(guess, fb, candidates, composite=composite)
                    replay_rewards.append(r)
                    candidates = filter_candidates(candidates, guess, fb)
                    replay_guesses.append(guess)
                    replay_feedback.append([f.value for f in fb])

                if state.solved:
                    batch_replays.append(
                        {
                            "target": target,
                            "guesses": replay_guesses,
                            "feedback": replay_feedback,
                            "solved": True,
                            "turns": state.turn,
                            "turn_rewards": replay_rewards,
                        }
                    )
                    continue

            turns_played = 0
            while not state.solved and not state.failed and turns_played < rl_cfg.max_turns:
                pi = torch.tensor(tokenizer.encode_game_state(state), dtype=torch.long, device=device)

                group_guesses: list[str] = []
                group_word_ids: list[Tensor] = []
                for _ in range(rl_cfg.group_size):
                    g, wids = sample_guess(model, pi, device, letter_mask)
                    group_guesses.append(g)
                    group_word_ids.append(wids)

                rewards: list[float] = []
                composite = rl_cfg.curriculum_phase != 1
                for g in group_guesses:
                    if len(g) != 5:
                        rewards.append(INVALID_WORD_PENALTY)
                        continue
                    sim_state, _ = env.step(state, g)
                    fb = sim_state.guesses[-1].feedback
                    r, _, _ = compute_reward(g, fb, candidates, composite=composite)
                    rewards.append(r)

                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                word_ids_batch = torch.stack(group_word_ids)

                with torch.no_grad():
                    old_lps = torch.stack(
                        [compute_guess_log_probs(model, pi, wid, letter_mask) for wid in group_word_ids]
                    )
                    ref_lps = torch.stack(
                        [compute_guess_log_probs(ref_model, pi, wid, letter_mask) for wid in group_word_ids]
                    )

                all_prompt_ids.append(pi)
                all_word_ids.append(word_ids_batch)
                all_rewards.append(rewards_t)
                all_old_lp.append(old_lps)
                all_ref_lp.append(ref_lps)

                best_idx = int(rewards_t.argmax().item())
                chosen = group_guesses[best_idx]
                replay_rewards.append(rewards[best_idx])

                if len(chosen) == 5:
                    new_state, _ = env.step(state, chosen)
                    fb = new_state.guesses[-1].feedback
                    replay_guesses.append(chosen)
                    replay_feedback.append([f.value for f in fb])
                    candidates = filter_candidates(candidates, chosen, fb)
                    state = new_state
                turns_played += 1

            batch_replays.append(
                {
                    "target": target,
                    "guesses": replay_guesses,
                    "feedback": replay_feedback,
                    "solved": state.solved,
                    "turns": state.turn,
                    "turn_rewards": replay_rewards,
                }
            )

        if not all_prompt_ids:
            continue

        n_turns = len(all_prompt_ids)
        last_loss = 0.0
        last_kl = 0.0
        last_clip = 0.0

        for _epoch in range(rl_cfg.ppo_epochs):
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_kl = 0.0
            epoch_clip = 0.0
            for i in range(n_turns):
                current_lps = torch.stack(
                    [compute_guess_log_probs(model, all_prompt_ids[i], wid, letter_mask) for wid in all_word_ids[i]]
                )

                advantages = compute_group_advantages(all_rewards[i])
                ratio = (current_lps - all_old_lp[i]).exp()
                clipped = ratio.clamp(1 - rl_cfg.clip_epsilon, 1 + rl_cfg.clip_epsilon)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                kl = (current_lps - all_ref_lp[i]).exp() - 1 - (current_lps - all_ref_lp[i])
                epoch_loss = epoch_loss + policy_loss + rl_cfg.kl_beta * kl.mean()
                epoch_kl += kl.mean().item()
                epoch_clip += ((ratio - 1).abs() > rl_cfg.clip_epsilon).float().mean().item()

            epoch_loss = epoch_loss / n_turns
            optimizer.zero_grad()
            epoch_loss.backward()
            clip_grad_norm(model, rl_cfg.grad_clip)
            optimizer.step()
            last_loss = epoch_loss.item()
            last_kl = epoch_kl / n_turns
            last_clip = epoch_clip / n_turns

        scheduler.step()

        live_dir = logger.log_dir / "live"
        live_dir.mkdir(exist_ok=True)
        live_data = {
            "step": step,
            "loss": last_loss,
            "kl_div": last_kl,
            "clip_fraction": last_clip,
            "games": batch_replays,
        }
        (live_dir / "latest.json").write_text(json.dumps(live_data))
        with open(live_dir / "history.jsonl", "a") as hf:
            hf.write(json.dumps({"step": step, "loss": last_loss, "kl_div": last_kl}) + "\n")

        elapsed = time.time() - t_start
        if step % 10 == 0 or step == 1:
            print(f"step {step:>5d}/{max_steps} | loss {last_loss:.4f} | elapsed {elapsed:.0f}s")

        if step % rl_cfg.eval_interval == 0:
            print(f"\n  [eval] Running {len(eval_words)} games...")
            eval_wr, eval_ag = evaluate_games(model, tokenizer, eval_words, device, letter_mask)
            logger.log_scalar("eval/win_rate", eval_wr, step)
            print(f"  [eval] win_rate: {eval_wr:.1%}, avg_guesses: {eval_ag:.1f}\n")

        if step % rl_cfg.checkpoint_interval == 0:
            ckpt_dir = logger.log_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
            print(f"  [checkpoint] saved to {ckpt_dir}")

    ckpt_dir = logger.log_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_dir / "model.pt", model, optimizer, step, model.config)
    print(f"\nTraining complete at step {step}")

    eval_wr, eval_ag = evaluate_games(model, tokenizer, eval_words, device, letter_mask)
    print(f"Final eval: win_rate={eval_wr:.1%}, avg_guesses={eval_ag:.1f}")
    logger.close()


def main() -> None:
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)
    train(config, checkpoint_path=args.checkpoint, opener_checkpoint=args.opener)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
