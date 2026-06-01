"""Checkpoint replay viewer: compare game replays across training checkpoints."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from mm_viz.board import render_comparison_html, render_game_html

if TYPE_CHECKING:
    from mm_viz.data import EvalSnapshot


def render_checkpoint_comparison(snapshots: list[EvalSnapshot], output_path: str) -> None:
    """Generate an HTML page comparing game replays across checkpoints.

    For each target word that appears in multiple snapshots, render the games
    side-by-side so you can see how the model's strategy evolved.

    Layout:
    - Header with checkpoint step numbers and win rates
    - For each shared target word: side-by-side game boards with labels
    - Summary stats table at the bottom
    """
    # Index replays by target word across snapshots
    target_to_replays: dict[str, list[tuple[int, EvalSnapshot, int]]] = defaultdict(list)
    for snap_idx, snap in enumerate(snapshots):
        for replay_idx, replay in enumerate(snap.replays):
            target_to_replays[replay.target].append((snap_idx, snap, replay_idx))

    # Find targets that appear in multiple snapshots
    shared_targets = {
        target: entries
        for target, entries in target_to_replays.items()
        if len({snap_idx for snap_idx, _, _ in entries}) > 1
    }

    # Build header row
    header_cells = "".join(
        f"<th>Step {snap.step}<br>Win: {snap.win_rate:.1%}<br>Avg: {snap.avg_guesses:.2f}</th>" for snap in snapshots
    )
    header_html = f"""
    <table style="border-collapse:collapse;margin-bottom:24px;">
    <tr style="background:#f0f0f0;">
        <th style="padding:8px;">Checkpoint</th>
        {header_cells}
    </tr>
    </table>
    """

    # Build comparison sections for shared targets
    sections: list[str] = []
    for target in sorted(shared_targets):
        entries = shared_targets[target]
        labels: list[str] = []
        replays = []
        for _snap_idx, snap, replay_idx in entries:
            labels.append(f"Step {snap.step}")
            replays.append(snap.replays[replay_idx])

        comparison = render_comparison_html(replays, labels)
        status = " / ".join(f"{'solved' if r.solved else 'failed'} in {r.turns}" for r in replays)
        sections.append(
            f'<div style="margin:16px 0;padding:12px;border:1px solid #ddd;border-radius:8px;">'
            f'<h3 style="margin:0 0 8px 0;">Target: {target.upper()} ({status})</h3>'
            f"{comparison}</div>"
        )

    # Build summary stats table
    stats_rows = "".join(
        f"<tr><td style='padding:4px 12px;'>{snap.step}</td>"
        f"<td style='padding:4px 12px;'>{snap.checkpoint_path}</td>"
        f"<td style='padding:4px 12px;'>{snap.win_rate:.1%}</td>"
        f"<td style='padding:4px 12px;'>{snap.avg_guesses:.2f}</td>"
        f"<td style='padding:4px 12px;'>{len(snap.replays)}</td></tr>"
        for snap in snapshots
    )
    stats_html = f"""
    <h2>Summary</h2>
    <table style="border-collapse:collapse;border:1px solid #ccc;">
    <tr style="background:#f0f0f0;">
        <th style="padding:4px 12px;">Step</th>
        <th style="padding:4px 12px;">Checkpoint</th>
        <th style="padding:4px 12px;">Win Rate</th>
        <th style="padding:4px 12px;">Avg Guesses</th>
        <th style="padding:4px 12px;">Games</th>
    </tr>
    {stats_rows}
    </table>
    """

    # Games only in one snapshot
    unique_sections: list[str] = []
    unique_targets = {target: entries for target, entries in target_to_replays.items() if target not in shared_targets}
    if unique_targets:
        unique_sections.append("<h2>Games Unique to One Checkpoint</h2>")
        for target in sorted(unique_targets):
            entries = unique_targets[target]
            snap_idx, snap, replay_idx = entries[0]
            replay = snap.replays[replay_idx]
            board = render_game_html(replay)
            unique_sections.append(
                f'<div style="margin:8px 0;">'
                f"<strong>{target.upper()}</strong> (Step {snap.step}, "
                f"{'solved' if replay.solved else 'failed'})"
                f"{board}</div>"
            )

    no_shared = ""
    if not shared_targets:
        no_shared = '<p style="color:#666;">No shared target words found across checkpoints.</p>'

    page = f"""<html>
<head><title>Checkpoint Comparison</title></head>
<body style="font-family:sans-serif;padding:24px;">
<h1>Checkpoint Comparison</h1>
{header_html}
<h2>Side-by-Side Comparisons</h2>
{no_shared}
{"".join(sections)}
{"".join(unique_sections)}
{stats_html}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)


def render_progress_report(snapshots: list[EvalSnapshot], output_path: str) -> None:
    """Generate an HTML page showing training progress over time.

    - Win rate over training steps (bar chart in HTML)
    - Average guesses over steps
    - Sample game replays at key checkpoints (first, middle, last)
    """
    if not snapshots:
        page = """<html>
<head><title>Training Progress</title></head>
<body style="font-family:sans-serif;padding:24px;">
<h1>Training Progress</h1>
<p>No snapshots provided.</p>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(page)
        return

    sorted_snaps = sorted(snapshots, key=lambda s: s.step)

    # Win rate bar chart
    max_bar = 300  # max bar width in pixels
    win_bars = "".join(
        f'<div style="display:flex;align-items:center;margin:4px 0;">'
        f'<span style="width:80px;text-align:right;margin-right:8px;">Step {s.step}</span>'
        f'<div style="width:{int(s.win_rate * max_bar)}px;height:20px;'
        f'background:#6aaa64;border-radius:3px;"></div>'
        f'<span style="margin-left:8px;">{s.win_rate:.1%}</span>'
        f"</div>"
        for s in sorted_snaps
    )

    # Avg guesses bar chart
    max_guesses = max(s.avg_guesses for s in sorted_snaps) if sorted_snaps else 1.0
    guess_bars = "".join(
        f'<div style="display:flex;align-items:center;margin:4px 0;">'
        f'<span style="width:80px;text-align:right;margin-right:8px;">Step {s.step}</span>'
        f'<div style="width:{int((s.avg_guesses / max_guesses) * max_bar)}px;height:20px;'
        f'background:#c9b458;border-radius:3px;"></div>'
        f'<span style="margin-left:8px;">{s.avg_guesses:.2f}</span>'
        f"</div>"
        for s in sorted_snaps
    )

    # Pick key checkpoints for sample games: first, middle, last
    key_indices: list[int] = [0]
    if len(sorted_snaps) > 2:
        key_indices.append(len(sorted_snaps) // 2)
    if len(sorted_snaps) > 1:
        key_indices.append(len(sorted_snaps) - 1)

    sample_sections: list[str] = []
    for idx in key_indices:
        snap = sorted_snaps[idx]
        # Show up to 3 sample games per checkpoint
        sample_replays = snap.replays[:3]
        games_html = "".join(
            f'<div style="margin:8px 0;">'
            f"<strong>{r.target.upper()}</strong> "
            f"({'solved' if r.solved else 'failed'})"
            f"{render_game_html(r)}</div>"
            for r in sample_replays
        )
        sample_sections.append(
            f'<div style="margin:16px 0;padding:12px;border:1px solid #ddd;border-radius:8px;">'
            f"<h3>Step {snap.step} (Win rate: {snap.win_rate:.1%})</h3>"
            f"{games_html}</div>"
        )

    page = f"""<html>
<head><title>Training Progress</title></head>
<body style="font-family:sans-serif;padding:24px;">
<h1>Training Progress</h1>

<h2>Win Rate Over Training</h2>
<div style="margin:16px 0;">
{win_bars}
</div>

<h2>Average Guesses Over Training</h2>
<div style="margin:16px 0;">
{guess_bars}
</div>

<h2>Sample Games at Key Checkpoints</h2>
{"".join(sample_sections)}

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)
