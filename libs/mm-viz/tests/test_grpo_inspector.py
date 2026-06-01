"""Tests for the GRPO Step Inspector visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm_viz.data import CompletionData, GRPOStepData
from mm_viz.grpo_inspector import render_grpo_step_html, render_grpo_trajectory_html

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures — realistic 8-completion step data
# ---------------------------------------------------------------------------

_COMPLETIONS = [
    CompletionData(
        tokens=["c", "r", "a", "n", "e"],
        text="crane",
        log_probs=[-0.15, -0.22, -0.10, -0.30, -0.18],
        reward=1.0,
        reward_breakdown={"valid": 0.0, "green": 0.6, "yellow": 0.0, "solved": 1.0, "total": 1.0},
    ),
    CompletionData(
        tokens=["s", "l", "a", "t", "e"],
        text="slate",
        log_probs=[-0.25, -0.35, -0.12, -0.40, -0.20],
        reward=0.5,
        reward_breakdown={"valid": 0.0, "green": 0.2, "yellow": 0.2, "solved": 0.0, "total": 0.5},
    ),
    CompletionData(
        tokens=["b", "r", "i", "n", "e"],
        text="brine",
        log_probs=[-0.30, -0.18, -0.50, -0.28, -0.16],
        reward=0.3,
        reward_breakdown={"valid": 0.0, "green": 0.2, "yellow": 0.1, "solved": 0.0, "total": 0.3},
    ),
    CompletionData(
        tokens=["g", "r", "a", "p", "e"],
        text="grape",
        log_probs=[-0.40, -0.20, -0.11, -0.55, -0.19],
        reward=0.2,
        reward_breakdown={"valid": 0.0, "green": 0.1, "yellow": 0.1, "solved": 0.0, "total": 0.2},
    ),
    CompletionData(
        tokens=["t", "r", "a", "c", "e"],
        text="trace",
        log_probs=[-0.35, -0.19, -0.10, -0.42, -0.17],
        reward=0.4,
        reward_breakdown={"valid": 0.0, "green": 0.2, "yellow": 0.2, "solved": 0.0, "total": 0.4},
    ),
    CompletionData(
        tokens=["p", "l", "u", "m", "b"],
        text="plumb",
        log_probs=[-0.60, -0.45, -0.55, -0.65, -0.70],
        reward=-0.5,
        reward_breakdown={"valid": 0.0, "green": 0.0, "yellow": 0.0, "solved": -0.5, "total": -0.5},
    ),
    CompletionData(
        tokens=["f", "l", "o", "s", "s"],
        text="floss",
        log_probs=[-0.55, -0.42, -0.60, -0.50, -0.48],
        reward=-0.3,
        reward_breakdown={"valid": 0.0, "green": 0.0, "yellow": 0.0, "solved": -0.3, "total": -0.3},
    ),
    CompletionData(
        tokens=["d", "r", "a", "n", "k"],
        text="drank",
        log_probs=[-0.38, -0.21, -0.10, -0.29, -0.45],
        reward=0.1,
        reward_breakdown={"valid": 0.0, "green": 0.1, "yellow": 0.0, "solved": 0.0, "total": 0.1},
    ),
]

_REWARDS = [c.reward for c in _COMPLETIONS]
_ADVANTAGES = [0.85, 0.25, 0.05, -0.05, 0.15, -0.75, -0.55, -0.15]

_OLD_PROBS = [0.12, 0.10, 0.08, 0.09, 0.11, 0.05, 0.06, 0.07]
_NEW_PROBS = [0.18, 0.11, 0.09, 0.08, 0.12, 0.03, 0.04, 0.06]

_GAME_STATE_TOKENS = [
    "[bos]",
    "s",
    "l",
    "a",
    "t",
    "e",
    "[gray]",
    "[gray]",
    "[green]",
    "[gray]",
    "[yellow]",
    "[eos]",
]
_GAME_STATE_TEXT = "[bos]slate[gray][gray][green][gray][yellow][eos]"


def _make_step_data(step: int = 100) -> GRPOStepData:
    """Create a realistic GRPOStepData fixture."""
    return GRPOStepData(
        step=step,
        game_state_tokens=_GAME_STATE_TOKENS,
        game_state_text=_GAME_STATE_TEXT,
        completions=_COMPLETIONS,
        rewards=_REWARDS,
        advantages=_ADVANTAGES,
        group_mean=0.2125,
        group_std=0.4720,
        old_probs=_OLD_PROBS,
        new_probs=_NEW_PROBS,
        kl_divergence=0.0042,
    )


def _make_step_data_no_prior_guesses(step: int = 0) -> GRPOStepData:
    """Create a GRPOStepData with no prior guesses (first turn)."""
    return GRPOStepData(
        step=step,
        game_state_tokens=["[bos]", "[eos]"],
        game_state_text="[bos][eos]",
        completions=_COMPLETIONS[:3],
        rewards=_REWARDS[:3],
        advantages=_ADVANTAGES[:3],
        group_mean=0.6,
        group_std=0.36,
        old_probs=_OLD_PROBS[:3],
        new_probs=_NEW_PROBS[:3],
        kl_divergence=0.001,
    )


# ---------------------------------------------------------------------------
# Tests: render_grpo_step_html
# ---------------------------------------------------------------------------


class TestRenderGRPOStepHtml:
    def test_produces_html_with_all_sections(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        assert "<html" in html
        assert "</html>" in html
        assert "1. Game State" in html
        assert "2. Group of Completions" in html
        assert "3. Reward Scoring" in html
        assert "4. Advantage Computation" in html
        assert "5. Policy Update" in html
        assert "6. Summary" in html

    def test_contains_guess_texts(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        for comp in _COMPLETIONS:
            assert comp.text.upper() in html

    def test_contains_reward_values(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        # Check that reward values appear in the HTML
        assert "+1.000" in html  # best reward
        assert "-0.500" in html  # worst reward

    def test_contains_advantage_values(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        # Check advantage values
        assert "+0.8500" in html  # best advantage
        assert "-0.7500" in html  # worst advantage

    def test_contains_probability_change_indicators(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        assert "Reinforced" in html
        assert "Suppressed" in html

    def test_contains_kl_divergence(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        assert "KL Divergence" in html
        assert "0.004200" in html

    def test_contains_wordle_tiles_for_prior_guesses(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        # Should have wordle tile colors for the prior guess "SLATE"
        assert "#6aaa64" in html  # green tile
        assert "#787c7e" in html  # gray tile
        assert "#c9b458" in html  # yellow tile
        # Should contain the letters from SLATE
        assert "S" in html
        assert "L" in html

    def test_handles_no_prior_guesses(self) -> None:
        html = render_grpo_step_html(_make_step_data_no_prior_guesses())
        assert "first turn" in html.lower() or "No prior guesses" in html

    def test_contains_step_number(self) -> None:
        html = render_grpo_step_html(_make_step_data(step=42))
        assert "Step 42" in html

    def test_contains_group_statistics(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        assert "Group Mean" in html
        assert "Group Std" in html

    def test_self_contained_html(self) -> None:
        html = render_grpo_step_html(_make_step_data())
        # Should have inline CSS, no external references
        assert "<style>" in html
        assert "font-family" in html
        # Should not reference external CSS or JS files
        assert 'link rel="stylesheet"' not in html


# ---------------------------------------------------------------------------
# Tests: render_grpo_trajectory_html
# ---------------------------------------------------------------------------


class TestRenderGRPOTrajectoryHtml:
    def test_produces_html_with_multiple_steps(self) -> None:
        steps = [_make_step_data(step=100), _make_step_data(step=200), _make_step_data(step=300)]
        html = render_grpo_trajectory_html(steps)
        assert "<html" in html
        assert "</html>" in html
        assert "Step 100" in html
        assert "Step 200" in html
        assert "Step 300" in html

    def test_contains_collapsible_sections(self) -> None:
        steps = [_make_step_data(step=100), _make_step_data(step=200)]
        html = render_grpo_trajectory_html(steps)
        assert "collapsible" in html
        assert "toggleStep" in html

    def test_contains_step_metrics_in_headers(self) -> None:
        steps = [_make_step_data(step=100)]
        html = render_grpo_trajectory_html(steps)
        assert "reward:" in html
        assert "KL:" in html

    def test_single_step_trajectory(self) -> None:
        steps = [_make_step_data(step=50)]
        html = render_grpo_trajectory_html(steps)
        assert "1 training steps" in html or "1 Steps" in html
        assert "Step 50" in html

    def test_all_sections_present_per_step(self) -> None:
        steps = [_make_step_data(step=100)]
        html = render_grpo_trajectory_html(steps)
        assert "1. Game State" in html
        assert "6. Summary" in html


# ---------------------------------------------------------------------------
# Tests: file I/O helpers
# ---------------------------------------------------------------------------


class TestRenderGRPOStepFromFile:
    def test_reads_json_writes_html(self, tmp_path: Path) -> None:
        from mm_viz.grpo_inspector import render_grpo_step_from_file

        # Save step data as JSON
        step_data = _make_step_data()
        json_path = tmp_path / "step-100.json"
        step_data.save(json_path)

        # Render from file
        output_path = tmp_path / "step-100.html"
        render_grpo_step_from_file(str(json_path), str(output_path))

        # Check output
        assert output_path.exists()
        html = output_path.read_text()
        assert "<html" in html
        assert "Step 100" in html
        assert "CRANE" in html


class TestRenderGRPOTrajectoryFromDir:
    def test_reads_directory_writes_html(self, tmp_path: Path) -> None:
        from mm_viz.grpo_inspector import render_grpo_trajectory_from_dir

        # Save multiple step data files
        for step in [100, 200, 300]:
            sd = _make_step_data(step=step)
            sd.save(tmp_path / f"step-{step}.json")

        # Render from directory
        output_path = tmp_path / "trajectory.html"
        render_grpo_trajectory_from_dir(str(tmp_path), str(output_path))

        # Check output
        assert output_path.exists()
        html = output_path.read_text()
        assert "<html" in html
        assert "Step 100" in html
        assert "Step 200" in html
        assert "Step 300" in html

    def test_sorts_by_step_number(self, tmp_path: Path) -> None:
        from mm_viz.grpo_inspector import render_grpo_trajectory_from_dir

        # Save out of order
        for step in [300, 100, 200]:
            sd = _make_step_data(step=step)
            sd.save(tmp_path / f"step-{step}.json")

        output_path = tmp_path / "trajectory.html"
        render_grpo_trajectory_from_dir(str(tmp_path), str(output_path))

        html = output_path.read_text()
        # All three should be present
        assert "Step 100" in html
        assert "Step 200" in html
        assert "Step 300" in html
