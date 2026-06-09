"""V3 Phase 3: two-phase GRPO RL (§5.4).

Phase 1 trains openers (turns 1-2) on raw expected info gain. Phase 2 freezes a
phase-1 opener for turns 1-2 and trains the policy on turns 3-6 with the composite
reward (normalized info gain + endgame/solve bonuses). Targets are sampled only
from the train split (hold-out is never an answer); the candidate pool is the full
universe (hold-out words stay reachable). Rollout uses the batched KV-cache group
decode (§5.7-B).

Usage:
    uv run python -m wordle3.finetune --config wordle3/configs/finetune-phase1.yaml \
        --checkpoint runs/sft-v3/<ts>/checkpoint-8000/model.pt
    uv run python -m wordle3.finetune --config wordle3/configs/finetune-phase2.yaml \
        --checkpoint runs/sft-v3/<ts>/checkpoint-8000/model.pt --opener runs/finetune-v3/<ts>/checkpoint-XXXX/model.pt
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
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
from mm_wordle import ConstraintTokenizer, GoldenSolver, PatternMatrix, WordleEnv, load_full_word_set
from mm_wordle.solver import filter_candidates

from wordle3.config import FinetuneConfig
from wordle3.metrics import build_letter_mask
from wordle3.reward import compute_reward_v3
from wordle3.rollout import group_log_probs, sample_group
from wordle3.splits import Split, load_split
from wordle3.steplog import MetricReporter
from wordle3.trainutil import write_live


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 two-phase GRPO RL")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="SFT checkpoint to fine-tune")
    parser.add_argument("--opener", type=str, default=None, help="Phase-1 checkpoint (phase 2 only)")
    parser.add_argument("--cache-dir", type=str, default="runs/cache")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _load_model(path: str, device: torch.device) -> GPT:
    ckpt = load_checkpoint(Path(path), device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def _record_move(
    g_list: list[str], fb_list: list[list[str]], r_list: list[float], guess: str, fb: list, reward: float
) -> None:
    g_list.append(guess)
    fb_list.append([f.value for f in fb])
    r_list.append(round(reward, 2))


def _advance(pm: PatternMatrix, env: WordleEnv, guess: str, target: str, candidate_idx: np.ndarray) -> np.ndarray:
    """Filter the candidate set after playing ``guess`` (matrix path for valid words)."""
    gi = pm.guess_index.get(guess)
    if gi is not None:
        return pm.consistent_idx(guess, pm.pattern_id(guess, target), candidate_idx)
    fb = env.compute_feedback(guess, target)
    kept = set(filter_candidates([pm.targets[i] for i in candidate_idx], guess, fb))
    return np.array([i for i in candidate_idx if pm.targets[i] in kept], dtype=np.int64)


def train(
    config: FinetuneConfig,
    *,
    checkpoint: str,
    opener_checkpoint: str | None = None,
    words: list[str] | None = None,
    pattern_matrix: PatternMatrix | None = None,
    split: Split | None = None,
    run_dir: Path | None = None,
    device: torch.device | None = None,
    cache_dir: str = "runs/cache",
) -> Path:
    """Run two-phase GRPO RL. Returns the run directory."""
    tcfg = config.training
    seed_everything(tcfg.seed)
    device = device or get_device()
    tokenizer = ConstraintTokenizer()
    words = words if words is not None else load_full_word_set()
    split = split if split is not None else load_split()
    if pattern_matrix is None:
        print("Loading/building pattern matrix...")
        pattern_matrix = PatternMatrix.load_or_build(words, cache_dir)
    print(f"Device: {device}")
    composite = tcfg.curriculum_phase != 1

    letter_mask = build_letter_mask(tokenizer, device)
    top_probes = GoldenSolver(pattern_matrix, probe_top_k=tcfg.probe_top_k).top_probes

    model = _load_model(checkpoint, device)
    model.eval()  # no dropout during RL so old/current log-probs match at ratio=1
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    opener_model = None
    if composite and opener_checkpoint:
        opener_model = _load_model(opener_checkpoint, device)
        opener_model.eval()
        for p in opener_model.parameters():
            p.requires_grad = False
        print(f"Opener model: {opener_checkpoint}")

    optimizer = create_optimizer(model, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=tcfg.warmup_steps, total_steps=tcfg.max_steps)

    if run_dir is None:
        logger = MetricsLogger(experiment="finetune-v3")
        run_dir = logger.log_dir
    else:
        logger = None
    print(f"Run dir: {run_dir}  (phase {tcfg.curriculum_phase})")
    RunManifest.capture(
        experiment="finetune-v3", config=config.model_dump(), seed=tcfg.seed, dataset_id="wordle-v3-rl"
    ).save(run_dir / "manifest.json")

    reporter = MetricReporter(logger, run_dir, model, tokenizer, pattern_matrix, device, tcfg, split)
    env = WordleEnv()
    n_words = len(pattern_matrix.targets)
    t0 = time.time()

    for step in range(1, tcfg.max_steps + 1):
        targets = [random.choice(split.train_answers) for _ in range(tcfg.batch_size)]
        turns: list[dict] = []  # collected experience for the PPO update
        replays: list[dict] = []  # the actual games played this step (for the live dashboard)
        chosen_valid = chosen_total = wins = 0
        chosen_ig_sum = 0.0
        chosen_ig_n = 0

        for target in targets:
            state = env.reset(target_word=target)
            candidate_idx = np.arange(n_words)
            g_list: list[str] = []
            fb_list: list[list[str]] = []
            r_list: list[float] = []

            if composite and opener_model is not None:
                for _ in range(2):
                    if state.solved or state.failed:
                        break
                    prompt = tokenizer.encode_game_state(state)
                    gw, _ = sample_group(opener_model, prompt, 1, device, letter_mask, tokenizer, temperature=0.1)
                    guess = gw[0] if len(gw[0]) == 5 else "zzzzz"
                    observed = pattern_matrix.pattern_id(guess, target) if guess in pattern_matrix.guess_index else 0
                    si = np.union1d(candidate_idx, top_probes)
                    best_ig = pattern_matrix.best_expected_info_gain(candidate_idx, search_idx=si)
                    r, _ = compute_reward_v3(
                        pattern_matrix, guess, candidate_idx, observed, composite=True, best_info_gain=best_ig
                    )
                    state, _ = env.step(state, guess)
                    _record_move(g_list, fb_list, r_list, guess, state.guesses[-1].feedback, r)
                    candidate_idx = _advance(pattern_matrix, env, guess, target, candidate_idx)

            played = 0
            while not state.solved and not state.failed and played < tcfg.max_turns:
                prompt = tokenizer.encode_game_state(state)
                gw, gen = sample_group(model, prompt, tcfg.group_size, device, letter_mask, tokenizer)
                # Denominator depends only on the candidate set — compute once for the whole group.
                best_ig = None
                if composite and len(candidate_idx) > 1:
                    si = np.union1d(candidate_idx, top_probes)
                    best_ig = pattern_matrix.best_expected_info_gain(candidate_idx, search_idx=si)

                rewards: list[float] = []
                expecteds: list[float] = []
                for w in gw:
                    observed = pattern_matrix.pattern_id(w, target) if w in pattern_matrix.guess_index else 0
                    r, e = compute_reward_v3(
                        pattern_matrix, w, candidate_idx, observed, composite=composite, best_info_gain=best_ig
                    )
                    rewards.append(r)
                    expecteds.append(e)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

                with torch.no_grad():
                    old_lp = group_log_probs(model, prompt, gen, device, letter_mask)
                    ref_lp = group_log_probs(ref_model, prompt, gen, device, letter_mask)
                turns.append({"prompt": prompt, "gen": gen, "rewards": rewards_t, "old": old_lp, "ref": ref_lp})

                best = int(rewards_t.argmax().item())
                chosen = gw[best] if len(gw[best]) == 5 else "zzzzz"
                chosen_total += 1
                chosen_valid += int(chosen in pattern_matrix.guess_index)
                if len(candidate_idx) > 1:
                    chosen_ig_sum += expecteds[best]
                    chosen_ig_n += 1
                state, _ = env.step(state, chosen)
                _record_move(g_list, fb_list, r_list, chosen, state.guesses[-1].feedback, rewards[best])
                candidate_idx = _advance(pattern_matrix, env, chosen, target, candidate_idx)
                played += 1

            wins += int(state.solved)
            replays.append(
                {
                    "target": target,
                    "guesses": g_list,
                    "feedback": fb_list,
                    "solved": state.solved,
                    "turns": state.turn,
                    "turn_rewards": r_list,
                }
            )

        if not turns:
            continue

        last_loss = last_kl = last_clip = 0.0
        for _epoch in range(tcfg.ppo_epochs):
            optimizer.zero_grad()
            total = torch.zeros((), device=device)
            kl_sum = clip_sum = 0.0
            for t in turns:
                cur = group_log_probs(model, t["prompt"], t["gen"], device, letter_mask)
                adv = compute_group_advantages(t["rewards"])
                ratio = (cur - t["old"]).exp()
                clipped = ratio.clamp(1 - tcfg.clip_epsilon, 1 + tcfg.clip_epsilon)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                kl = ((cur - t["ref"]).exp() - 1 - (cur - t["ref"])).mean()
                total = total + policy_loss + tcfg.kl_beta * kl
                kl_sum += kl.item()
                clip_sum += ((ratio - 1).abs() > tcfg.clip_epsilon).float().mean().item()
            loss = total / len(turns)
            loss.backward()
            clip_grad_norm(model, tcfg.grad_clip)
            optimizer.step()
            last_loss, last_kl, last_clip = loss.item(), kl_sum / len(turns), clip_sum / len(turns)
        scheduler.step()

        # Live dashboard: the actual games just played, with their composite rewards.
        row = {
            "step": step,
            "loss": last_loss,
            "kl_div": last_kl,
            "clip_fraction": last_clip,
            "valid_word_rate": chosen_valid / max(chosen_total, 1),
            "info_gain": chosen_ig_sum / chosen_ig_n if chosen_ig_n else 0.0,
            "win_rate": wins / len(replays),
        }
        write_live(run_dir, row, replays)
        if logger is not None:
            for name, val in row.items():
                if name != "step":
                    logger.log_scalar(f"train/{name}", val, step)
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:>5d}/{tcfg.max_steps} | loss {last_loss:.4f} | win {row['win_rate']:.0%} | {el:.0f}s")

        reporter.maybe_eval(step)

        if step % tcfg.checkpoint_interval == 0 or step == tcfg.max_steps:
            save_checkpoint(run_dir / f"checkpoint-{step}" / "model.pt", model, optimizer, step, model.config)

    if logger is not None:
        logger.close()
    print(f"Done. Run dir: {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    config = FinetuneConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    train(config, checkpoint=args.checkpoint, opener_checkpoint=args.opener, cache_dir=args.cache_dir)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
