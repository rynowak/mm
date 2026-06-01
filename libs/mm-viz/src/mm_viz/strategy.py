"""Strategy evolution analysis and visualization.

Analyzes how a model's Wordle strategy changes across training checkpoints,
and renders the results as a self-contained HTML page with inline SVG charts.
"""

from __future__ import annotations

import html as html_mod
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mm_viz.data import EvalSnapshot


def analyze_strategy(snapshots: list[EvalSnapshot]) -> dict:
    """Analyze strategy metrics across checkpoints.

    Returns dict with:
    - ``'steps'``: list of step numbers
    - ``'win_rates'``: list of win rates
    - ``'avg_guesses'``: list of average guesses (winners only)
    - ``'first_guess_diversity'``: list of unique first guess counts
    - ``'first_guess_distribution'``: dict mapping step -> {word: count}
    - ``'letter_frequency'``: dict mapping step -> {letter: count} across all first guesses
    - ``'guess_distribution'``: dict mapping step -> {1: n, 2: n, ..., 6: n, 'X': n}
    """
    steps: list[int] = []
    win_rates: list[float] = []
    avg_guesses: list[float] = []
    first_guess_diversity: list[int] = []
    first_guess_distribution: dict[int, dict[str, int]] = {}
    letter_frequency: dict[int, dict[str, int]] = {}
    guess_distribution: dict[int, dict[str | int, int]] = {}

    for snap in snapshots:
        step = snap.step
        steps.append(step)
        win_rates.append(snap.win_rate)
        avg_guesses.append(snap.avg_guesses)

        # First guess analysis
        first_guesses: list[str] = []
        for replay in snap.replays:
            if replay.guesses:
                first_guesses.append(replay.guesses[0])

        first_counts = Counter(first_guesses)
        first_guess_distribution[step] = dict(first_counts)
        first_guess_diversity.append(len(first_counts))

        # Letter frequency in first guesses
        letter_counts: Counter[str] = Counter()
        for guess in first_guesses:
            for ch in guess:
                letter_counts[ch] += 1
        letter_frequency[step] = dict(letter_counts)

        # Guess distribution (how many turns to solve)
        dist: dict[str | int, int] = {i: 0 for i in range(1, 7)}
        dist["X"] = 0
        for replay in snap.replays:
            if replay.solved:
                bucket = min(replay.turns, 6)
                dist[bucket] = dist.get(bucket, 0) + 1
            else:
                dist["X"] = dist.get("X", 0) + 1
        guess_distribution[step] = dist

    return {
        "steps": steps,
        "win_rates": win_rates,
        "avg_guesses": avg_guesses,
        "first_guess_diversity": first_guess_diversity,
        "first_guess_distribution": first_guess_distribution,
        "letter_frequency": letter_frequency,
        "guess_distribution": guess_distribution,
    }


# ---------------------------------------------------------------------------
# SVG chart helpers
# ---------------------------------------------------------------------------

_CHART_W = 600
_CHART_H = 300
_PAD = 50  # padding for axes


def _svg_line_chart(
    xs: list[float],
    ys: list[float],
    title: str,
    y_label: str,
    color: str = "#4a90d9",
    y_format: str = ".2f",
) -> str:
    """Render a simple SVG line chart with axes."""
    if not xs or not ys:
        return f"<p>No data for {html_mod.escape(title)}</p>"

    w, h, pad = _CHART_W, _CHART_H, _PAD
    plot_w = w - 2 * pad
    plot_h = h - 2 * pad

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add a small margin to y range so points don't sit on the edge
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_min_plot = y_min - y_range * 0.05
    y_max_plot = y_max + y_range * 0.05
    y_range_plot = y_max_plot - y_min_plot

    x_range = x_max - x_min if x_max != x_min else 1.0

    def tx(x: float) -> float:
        return pad + (x - x_min) / x_range * plot_w

    def ty(y: float) -> float:
        return pad + plot_h - (y - y_min_plot) / y_range_plot * plot_h

    # Build polyline points
    points = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs, ys, strict=True))

    # Axis ticks (up to 5 y-ticks, up to 5 x-ticks)
    y_ticks = _nice_ticks(y_min, y_max, 5)
    x_ticks = _nice_ticks(x_min, x_max, 5)

    y_tick_lines = ""
    for yt in y_ticks:
        yp = ty(yt)
        label = f"{yt:{y_format}}"
        y_tick_lines += (
            f'<line x1="{pad}" y1="{yp:.1f}" x2="{pad + plot_w}" y2="{yp:.1f}" '
            f'stroke="#e0e0e0" stroke-width="1"/>\n'
            f'<text x="{pad - 5}" y="{yp:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-size="11" fill="#666">{label}</text>\n'
        )

    x_tick_lines = ""
    for xt in x_ticks:
        xp = tx(xt)
        x_tick_lines += (
            f'<text x="{xp:.1f}" y="{pad + plot_h + 16}" text-anchor="middle" '
            f'font-size="11" fill="#666">{int(xt)}</text>\n'
        )

    # Data point circles
    circles = ""
    for x, y in zip(xs, ys, strict=True):
        circles += f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="3" fill="{color}"/>\n'

    return f"""<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
  <text x="{w // 2}" y="16" text-anchor="middle" font-size="14" font-weight="bold">{html_mod.escape(title)}</text>
  <text x="14" y="{h // 2}" text-anchor="middle" font-size="12" fill="#666"
        transform="rotate(-90,14,{h // 2})">{html_mod.escape(y_label)}</text>
  {y_tick_lines}
  {x_tick_lines}
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}"
        fill="none" stroke="#ccc" stroke-width="1"/>
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>
  {circles}
</svg>"""


def _nice_ticks(lo: float, hi: float, n: int) -> list[float]:
    """Generate up to *n* evenly-spaced tick values between *lo* and *hi*."""
    if lo == hi:
        return [lo]
    step = (hi - lo) / max(n - 1, 1)
    return [lo + i * step for i in range(n)]


def _svg_stacked_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    colors: list[str],
) -> str:
    """Render a stacked bar chart as SVG."""
    if not labels:
        return f"<p>No data for {html_mod.escape(title)}</p>"

    w, h, pad = _CHART_W, _CHART_H + 30, _PAD
    plot_w = w - 2 * pad
    plot_h = h - 2 * pad - 20  # extra space for legend

    n_bars = len(labels)
    bar_w = plot_w / max(n_bars, 1) * 0.7
    gap = plot_w / max(n_bars, 1) * 0.3

    # Compute max total height
    series_keys = list(series.keys())
    totals = []
    for i in range(n_bars):
        totals.append(sum(series[k][i] for k in series_keys))
    max_total = max(totals) if totals else 1.0
    if max_total == 0:
        max_total = 1.0

    bars_svg = ""
    for i in range(n_bars):
        x_base = pad + i * (bar_w + gap) + gap / 2
        y_offset = 0.0
        for j, key in enumerate(series_keys):
            val = series[key][i]
            bar_h = val / max_total * plot_h
            y_pos = pad + plot_h - y_offset - bar_h
            color = colors[j % len(colors)]
            bars_svg += (
                f'<rect x="{x_base:.1f}" y="{y_pos:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}" stroke="white" stroke-width="0.5"/>\n'
            )
            y_offset += bar_h

        # X label
        bars_svg += (
            f'<text x="{x_base + bar_w / 2:.1f}" y="{pad + plot_h + 14}" '
            f'text-anchor="middle" font-size="10" fill="#666">'
            f"{html_mod.escape(labels[i])}</text>\n"
        )

    # Legend
    legend = ""
    lx = pad
    for j, key in enumerate(series_keys):
        color = colors[j % len(colors)]
        legend += (
            f'<rect x="{lx}" y="{h - 18}" width="12" height="12" fill="{color}"/>'
            f'<text x="{lx + 16}" y="{h - 8}" font-size="11" fill="#333">'
            f"{html_mod.escape(key)}</text>"
        )
        lx += len(key) * 7 + 30

    return f"""<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
  <text x="{w // 2}" y="16" text-anchor="middle" font-size="14" font-weight="bold">{html_mod.escape(title)}</text>
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}"
        fill="none" stroke="#ccc" stroke-width="1"/>
  {bars_svg}
  {legend}
</svg>"""


# ---------------------------------------------------------------------------
# HTML tables and heatmap
# ---------------------------------------------------------------------------

_HEATMAP_COLORS = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
]


def _letter_heatmap_html(letter_freq: dict[int, dict[str, int]], steps: list[int]) -> str:
    """Render a letter frequency heatmap as an HTML table."""
    # Collect all letters that appear
    all_letters: set[str] = set()
    for counts in letter_freq.values():
        all_letters.update(counts.keys())
    letters = sorted(all_letters)

    if not letters:
        return "<p>No letter frequency data.</p>"

    # Find global max for color scaling
    max_count = max(
        (c for counts in letter_freq.values() for c in counts.values()),
        default=1,
    )
    if max_count == 0:
        max_count = 1

    header = "<tr><th>Step</th>" + "".join(f"<th>{ch}</th>" for ch in letters) + "</tr>"
    rows = ""
    for step in steps:
        counts = letter_freq.get(step, {})
        cells = ""
        for ch in letters:
            val = counts.get(ch, 0)
            idx = int(val / max_count * (len(_HEATMAP_COLORS) - 1))
            bg = _HEATMAP_COLORS[idx]
            text_color = "#fff" if idx >= 5 else "#333"
            cells += (
                f'<td style="background:{bg};color:{text_color};text-align:center;'
                f'padding:4px 6px;font-size:12px;">{val}</td>'
            )
        rows += f"<tr><td style='font-weight:bold;padding:4px 8px;'>{step}</td>{cells}</tr>\n"

    return f'<table style="border-collapse:collapse;margin:8px 0;">{header}\n{rows}</table>'


def _first_guess_table_html(first_guess_dist: dict[int, dict[str, int]], steps: list[int]) -> str:
    """Render a table of the top 10 first guesses at each checkpoint."""
    rows = ""
    for step in steps:
        dist = first_guess_dist.get(step, {})
        top10 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:10]
        entries = ", ".join(f"{word} ({count})" for word, count in top10)
        rows += (
            f"<tr><td style='font-weight:bold;padding:4px 8px;'>{step}</td>"
            f"<td style='padding:4px 8px;'>{html_mod.escape(entries) if entries else '(none)'}</td></tr>\n"
        )

    return (
        '<table style="border-collapse:collapse;margin:8px 0;">'
        "<tr><th style='padding:4px 8px;'>Step</th>"
        "<th style='padding:4px 8px;'>Top 10 First Guesses</th></tr>\n"
        f"{rows}</table>"
    )


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_strategy_html(snapshots: list[EvalSnapshot], output_path: str) -> None:
    """Generate an HTML page showing strategy evolution.

    Sections:
    1. Win Rate Over Training -- line chart (SVG)
    2. Average Guesses Over Training -- line chart
    3. First Guess Analysis -- table of top 10 first guesses per checkpoint
    4. Letter Frequency -- heatmap of letter usage in first guesses
    5. Guess Distribution -- stacked bar chart
    """
    data = analyze_strategy(snapshots)

    steps_float = [float(s) for s in data["steps"]]

    # 1. Win rate line chart
    win_rate_svg = _svg_line_chart(
        steps_float,
        data["win_rates"],
        "Win Rate Over Training",
        "Win Rate",
        color="#27ae60",
        y_format=".2f",
    )

    # 2. Average guesses line chart
    avg_guess_svg = _svg_line_chart(
        steps_float,
        data["avg_guesses"],
        "Average Guesses Over Training",
        "Avg Guesses",
        color="#e67e22",
        y_format=".2f",
    )

    # 3. First guess table
    first_guess_table = _first_guess_table_html(data["first_guess_distribution"], data["steps"])

    # 4. Letter frequency heatmap
    heatmap = _letter_heatmap_html(data["letter_frequency"], data["steps"])

    # 5. Guess distribution stacked bar chart
    guess_dist = data["guess_distribution"]
    bar_labels = [f"Step {s}" for s in data["steps"]]
    bar_keys = ["1", "2", "3", "4", "5", "6", "X"]
    bar_colors = [
        "#27ae60",
        "#2ecc71",
        "#f1c40f",
        "#e67e22",
        "#e74c3c",
        "#c0392b",
        "#7f8c8d",
    ]
    bar_series: dict[str, list[float]] = {}
    for key in bar_keys:
        bar_series[key] = []
        for step in data["steps"]:
            dist = guess_dist.get(step, {})
            # Keys in guess_distribution can be int or str 'X'
            k: str | int = int(key) if key != "X" else "X"
            bar_series[key].append(float(dist.get(k, 0)))

    guess_dist_svg = _svg_stacked_bar_chart(bar_labels, bar_series, "Guess Distribution Over Training", bar_colors)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Strategy Evolution</title>
<style>
  body {{ font-family: sans-serif; padding: 24px; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; margin-top: 32px; }}
  section {{ margin-bottom: 24px; }}
  table {{ font-size: 13px; }}
  th {{ text-align: left; padding: 4px 8px; background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Strategy Evolution</h1>

<section>
<h2>1. Win Rate Over Training</h2>
{win_rate_svg}
</section>

<section>
<h2>2. Average Guesses Over Training</h2>
{avg_guess_svg}
</section>

<section>
<h2>3. First Guess Analysis</h2>
{first_guess_table}
</section>

<section>
<h2>4. Letter Frequency in First Guesses</h2>
{heatmap}
</section>

<section>
<h2>5. Guess Distribution</h2>
{guess_dist_svg}
</section>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(page)
