"""Tetris RL training script for autoresearch.

This is the file the autoresearch agent edits. It trains a DQN agent to play
Tetris for a fixed time budget, then evaluates and reports the metric.

Metric: val_avg_lines (average lines cleared per game over evaluation games).
"""

import json
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mm_tetris
import torch
import torch.nn as nn
import torch.nn.functional as F
from mm_training import get_device, seed_everything

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
TRAIN_SECONDS = 300  # 5-minute training budget
EVAL_GAMES = 100
LR = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 300_000
REPLAY_CAPACITY = 50_000
TARGET_UPDATE_FREQ = 200
MAX_STEPS_PER_GAME = 500

EventCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Reward function (autoresearch can modify this)
# ---------------------------------------------------------------------------
def compute_reward(old: mm_tetris.TetrisState, new: mm_tetris.TetrisState) -> float:
    lines = new.lines_cleared - old.lines_cleared
    reward = 0.0

    if lines == 1:
        reward += 1.0
    elif lines == 2:
        reward += 3.0
    elif lines == 3:
        reward += 5.0
    elif lines >= 4:
        reward += 8.0

    if new.game_over:
        reward -= 2.0

    reward += 0.01

    new_holes = new.count_holes()
    old_holes = old.count_holes()
    reward -= 0.1 * max(0, new_holes - old_holes)

    reward -= 0.01 * new.max_height()

    return reward


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, mm_tetris.NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAPACITY) -> None:
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def push(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FEATURE_SIZE = 10 + 10 + 7 + 4  # heights, height_diffs, piece_onehot, aggregate stats


def encode_state(state: mm_tetris.TetrisState, device: torch.device) -> torch.Tensor:
    heights = state.column_heights()
    max_h = max(heights) if heights else 0
    holes = state.count_holes()
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    normed_heights = [h / mm_tetris.GRID_HEIGHT for h in heights]
    height_diffs = [(heights[i] - heights[i + 1]) / mm_tetris.GRID_HEIGHT for i in range(len(heights) - 1)] + [0.0]
    piece_onehot = [0.0] * mm_tetris.NUM_PIECE_TYPES
    piece_onehot[mm_tetris.PIECE_NAMES.index(state.current_piece)] = 1.0
    total_height = sum(heights) / (mm_tetris.GRID_HEIGHT * mm_tetris.GRID_WIDTH)
    aggregate = [max_h / mm_tetris.GRID_HEIGHT, holes / 40.0, bumpiness / 40.0, total_height]
    features = normed_heights + height_diffs + piece_onehot + aggregate
    return torch.tensor(features, dtype=torch.float32, device=device)


def select_action(
    policy_net: DQN,
    state: mm_tetris.TetrisState,
    device: torch.device,
    epsilon: float,
) -> int:
    mask = mm_tetris.valid_action_mask(state)
    valid_actions = [i for i, v in enumerate(mask) if v]
    if not valid_actions:
        return 0

    if random.random() < epsilon:
        return random.choice(valid_actions)

    with torch.no_grad():
        q_values = policy_net(encode_state(state, device))
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        q_values[~mask_tensor] = float("-inf")
        return int(q_values.argmax().item())


def evaluate(policy_net: DQN, device: torch.device, num_games: int = EVAL_GAMES) -> tuple[float, float]:
    """Play games greedily and return (avg_lines, avg_score)."""
    total_lines = 0
    total_score = 0

    for game_idx in range(num_games):
        state = mm_tetris.reset(seed=10000 + game_idx)
        steps = 0
        while not state.game_over and steps < MAX_STEPS_PER_GAME:
            action = select_action(policy_net, state, device, epsilon=0.0)
            state = mm_tetris.step(state, action)
            steps += 1
        total_lines += state.lines_cleared
        total_score += state.score

    return total_lines / num_games, total_score / num_games


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
_LIVE_DIR = Path(__file__).parent / ".live"


def _write_live(name: str, data: dict[str, Any]) -> None:
    _LIVE_DIR.mkdir(exist_ok=True)
    _LIVE_DIR.joinpath(name).write_text(json.dumps(data))


def _append_live(name: str, data: dict[str, Any]) -> None:
    _LIVE_DIR.mkdir(exist_ok=True)
    with _LIVE_DIR.joinpath(name).open("a") as f:
        f.write(json.dumps(data) + "\n")


def train(on_event: EventCallback | None = None) -> None:
    _LIVE_DIR.mkdir(exist_ok=True)
    for f in _LIVE_DIR.iterdir():
        f.unlink()

    last_board_write = 0.0

    def emit(event: dict[str, Any]) -> None:
        nonlocal last_board_write
        if on_event is not None:
            on_event(event)
        if event["type"] == "step":
            now = time.monotonic()
            if now - last_board_write > 0.2:
                _write_live("board.json", event)
                last_board_write = now
        elif event["type"] == "episode_end":
            _append_live("metrics.jsonl", event)

    seed_everything(SEED)
    device = torch.device("cpu")
    print(f"Device: {device}")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer()

    epsilon = EPSILON_START
    game_seed = 0
    state = mm_tetris.reset(seed=game_seed)
    game_steps = 0

    emit({"type": "game_start", "game_id": game_seed, "piece": state.current_piece})

    episodes = 0
    total_steps = 0
    last_loss: float | None = None
    start_time = time.time()

    while time.time() - start_time < TRAIN_SECONDS:
        state_tensor = encode_state(state, device)
        action = select_action(policy_net, state, device, epsilon)
        new_state = mm_tetris.step(state, action)
        reward = compute_reward(state, new_state)
        next_tensor = encode_state(new_state, device)

        rotation, column = divmod(action, mm_tetris.GRID_WIDTH)
        emit(
            {
                "type": "step",
                "game_id": game_seed,
                "step": game_steps,
                "grid": new_state.grid,
                "piece": new_state.current_piece,
                "action": action,
                "rotation": rotation,
                "column": column,
                "lines_cleared": new_state.lines_cleared,
                "score": new_state.score,
                "game_over": new_state.game_over,
            }
        )

        replay.push(Experience(state_tensor, action, reward, next_tensor, new_state.game_over))

        total_steps += 1
        game_steps += 1

        if new_state.game_over or game_steps >= MAX_STEPS_PER_GAME:
            emit(
                {
                    "type": "episode_end",
                    "game_id": game_seed,
                    "episode": episodes,
                    "lines": new_state.lines_cleared,
                    "score": new_state.score,
                    "pieces": new_state.pieces_placed,
                    "total_steps": total_steps,
                    "epsilon": epsilon,
                    "loss": last_loss,
                    "elapsed": time.time() - start_time,
                }
            )
            episodes += 1
            game_seed += 1
            state = mm_tetris.reset(seed=game_seed)
            game_steps = 0
            emit({"type": "game_start", "game_id": game_seed, "piece": state.current_piece})
        else:
            state = new_state

        epsilon = max(EPSILON_END, EPSILON_START - total_steps * (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS)

        if len(replay) < BATCH_SIZE:
            continue

        batch = replay.sample(BATCH_SIZE)
        states = torch.stack([e.state for e in batch])
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=device)
        next_states = torch.stack([e.next_state for e in batch])
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=device)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = policy_net(next_states).argmax(1)
            next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target_q)
        last_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()

        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    elapsed = time.time() - start_time
    print(f"Training: {episodes} episodes, {total_steps} steps in {elapsed:.1f}s")

    emit({"type": "eval_start"})
    avg_lines, avg_score = evaluate(policy_net, device)
    print(f"Eval ({EVAL_GAMES} games): avg_lines={avg_lines:.2f}, avg_score={avg_score:.1f}")
    print(f"val_avg_lines: {avg_lines:.4f}")
    emit(
        {
            "type": "training_complete",
            "val_avg_lines": avg_lines,
            "val_avg_score": avg_score,
            "episodes": episodes,
            "total_steps": total_steps,
            "elapsed": elapsed,
        }
    )


if __name__ == "__main__":
    train()
