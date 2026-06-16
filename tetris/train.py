"""Tetris RL training script for autoresearch.

1-step lookahead heuristic with learned weights: for each valid action,
simulate the placement and score the resulting board using a linear
combination of features. Weights are optimized via CMA-ES-style
evolutionary search.

Metric: val_avg_lines (average lines cleared per game over evaluation games).
"""

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mm_tetris
from mm_training import seed_everything

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
TRAIN_SECONDS = 300
EVAL_GAMES = 100
MAX_STEPS_PER_GAME = 500
POPULATION_SIZE = 20
ELITE_COUNT = 5
NUM_FEATURES = 4  # aggregate_height, holes, bumpiness, lines_cleared
NOISE_STD = 0.3
EVAL_GAMES_PER_CANDIDATE = 5

EventCallback = Callable[[dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Board scoring
# ---------------------------------------------------------------------------
def score_board(state: mm_tetris.TetrisState, lines_just_cleared: int, weights: list[float]) -> float:
    heights = state.column_heights()
    agg_height = sum(heights)
    holes = state.count_holes()
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    return weights[0] * agg_height + weights[1] * holes + weights[2] * bumpiness + weights[3] * lines_just_cleared


def select_action_heuristic(state: mm_tetris.TetrisState, weights: list[float]) -> int:
    mask = mm_tetris.valid_action_mask(state)
    best_action = -1
    best_score = float("-inf")

    for action in range(mm_tetris.NUM_ACTIONS):
        if not mask[action]:
            continue
        next_state = mm_tetris.step(state, action)
        lines = next_state.lines_cleared - state.lines_cleared
        s = score_board(next_state, lines, weights)
        if s > best_score:
            best_score = s
            best_action = action

    return best_action if best_action >= 0 else 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_weights(weights: list[float], num_games: int, start_seed: int) -> float:
    total_lines = 0
    for i in range(num_games):
        state = mm_tetris.reset(seed=start_seed + i)
        steps = 0
        while not state.game_over and steps < MAX_STEPS_PER_GAME:
            action = select_action_heuristic(state, weights)
            state = mm_tetris.step(state, action)
            steps += 1
        total_lines += state.lines_cleared
    return total_lines / num_games


def evaluate(weights: list[float]) -> tuple[float, float]:
    total_lines = 0
    total_score = 0
    for game_idx in range(EVAL_GAMES):
        state = mm_tetris.reset(seed=10000 + game_idx)
        steps = 0
        while not state.game_over and steps < MAX_STEPS_PER_GAME:
            action = select_action_heuristic(state, weights)
            state = mm_tetris.step(state, action)
            steps += 1
        total_lines += state.lines_cleared
        total_score += state.score
    return total_lines / EVAL_GAMES, total_score / EVAL_GAMES


# ---------------------------------------------------------------------------
# Training — evolutionary weight optimization
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

    seed_everything(SEED)
    rng = random.Random(SEED)
    print("Device: cpu (heuristic search, no neural network)")

    # Known good starting point for Tetris heuristics
    best_weights = [-0.5, -0.7, -0.2, 1.0]
    best_fitness = evaluate_weights(best_weights, EVAL_GAMES_PER_CANDIDATE, start_seed=50000)

    generation = 0
    eval_seed = 60000
    start_time = time.time()

    while time.time() - start_time < TRAIN_SECONDS:
        candidates: list[tuple[list[float], float]] = []

        for _ in range(POPULATION_SIZE):
            child = [w + rng.gauss(0, NOISE_STD) for w in best_weights]
            fitness = evaluate_weights(child, EVAL_GAMES_PER_CANDIDATE, start_seed=eval_seed)
            candidates.append((child, fitness))
            eval_seed += EVAL_GAMES_PER_CANDIDATE

        candidates.sort(key=lambda x: x[1], reverse=True)
        elite = candidates[:ELITE_COUNT]

        new_weights = [0.0] * NUM_FEATURES
        for w, _ in elite:
            for i in range(NUM_FEATURES):
                new_weights[i] += w[i] / ELITE_COUNT

        new_fitness = evaluate_weights(new_weights, EVAL_GAMES_PER_CANDIDATE, start_seed=eval_seed)
        eval_seed += EVAL_GAMES_PER_CANDIDATE

        if new_fitness >= best_fitness:
            best_weights = new_weights
            best_fitness = new_fitness

        elite_avg = sum(f for _, f in elite) / len(elite)
        pop_avg = sum(f for _, f in candidates) / len(candidates)

        _append_live(
            "metrics.jsonl",
            {
                "type": "episode_end",
                "game_id": eval_seed,
                "episode": generation,
                "lines": int(best_fitness),
                "score": int(elite_avg * 100),
                "pieces": 0,
                "total_steps": eval_seed,
                "epsilon": best_fitness,
                "loss": pop_avg,
                "elapsed": time.time() - start_time,
            },
        )

        generation += 1

    elapsed = time.time() - start_time
    print(f"Search: {generation} generations in {elapsed:.1f}s")
    print(f"Best weights: {[f'{w:.3f}' for w in best_weights]}")

    avg_lines, avg_score = evaluate(best_weights)
    print(f"Eval ({EVAL_GAMES} games): avg_lines={avg_lines:.2f}, avg_score={avg_score:.1f}")
    print(f"val_avg_lines: {avg_lines:.4f}")


if __name__ == "__main__":
    train()
