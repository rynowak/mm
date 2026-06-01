"""mm-viz: Training visualizations for game replays and GRPO step inspection."""

from mm_viz.board import render_comparison_html, render_game_html, render_games_report
from mm_viz.data import CompletionData, EvalSnapshot, GameReplay, GRPOStepData

__all__ = [
    "CompletionData",
    "EvalSnapshot",
    "GRPOStepData",
    "GameReplay",
    "render_comparison_html",
    "render_game_html",
    "render_games_report",
]
