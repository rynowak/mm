"""mm-viz: Training visualizations for game replays and GRPO step inspection."""

from mm_viz.attention import (
    extract_attention_weights,
    render_attention_html,
    render_wordle_attention_html,
)
from mm_viz.board import render_comparison_html, render_game_html, render_games_report
from mm_viz.data import CompletionData, EvalSnapshot, GameReplay, GRPOStepData
from mm_viz.grpo_inspector import (
    render_grpo_step_from_file,
    render_grpo_step_html,
    render_grpo_trajectory_from_dir,
    render_grpo_trajectory_html,
)
from mm_viz.replay_viewer import render_checkpoint_comparison, render_progress_report

__all__ = [
    "CompletionData",
    "EvalSnapshot",
    "GRPOStepData",
    "GameReplay",
    "extract_attention_weights",
    "render_attention_html",
    "render_checkpoint_comparison",
    "render_comparison_html",
    "render_game_html",
    "render_games_report",
    "render_grpo_step_from_file",
    "render_grpo_step_html",
    "render_grpo_trajectory_from_dir",
    "render_grpo_trajectory_html",
    "render_progress_report",
    "render_wordle_attention_html",
]
