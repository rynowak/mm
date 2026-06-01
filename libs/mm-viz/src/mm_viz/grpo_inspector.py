"""GRPO Step Inspector — self-contained HTML visualization of GRPO training steps.

Renders a single training step as a scrollable HTML page that tells the story
of the GRPO algorithm: game state, completions, rewards, advantages, policy
update, and summary.
"""

from __future__ import annotations

import html
from pathlib import Path

from mm_viz.data import GRPOStepData

# ---------------------------------------------------------------------------
# Color constants (Wordle palette)
# ---------------------------------------------------------------------------

_GREEN = "#6aaa64"
_YELLOW = "#c9b458"
_GRAY = "#787c7e"
_EMPTY = "#d3d6da"
_RED = "#c0392b"
_GREEN_BG = "#e8f5e9"
_GREEN_TEXT = "#2e7d32"
_RED_BG = "#ffebee"
_RED_TEXT = "#c62828"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = f"""\
* {{ box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    max-width: 920px;
    margin: 0 auto;
    padding: 20px 24px 60px;
    background: #fafafa;
    color: #222;
    line-height: 1.5;
}}
h1 {{
    font-size: 1.6rem;
    margin: 0 0 4px;
}}
.subtitle {{
    color: #666;
    font-size: 0.95rem;
    margin-bottom: 24px;
}}
.section {{
    margin: 28px 0;
    padding: 20px 24px;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}}
.section h2 {{
    font-size: 1.15rem;
    color: #333;
    border-bottom: 2px solid {_GREEN};
    padding-bottom: 8px;
    margin: 0 0 6px;
}}
.section .desc {{
    color: #666;
    font-size: 0.88rem;
    margin-bottom: 16px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 0.9rem;
}}
th {{
    padding: 8px 12px;
    border: 1px solid #ddd;
    text-align: center;
    background: #f5f5f5;
    font-weight: 600;
}}
td {{
    padding: 8px 12px;
    border: 1px solid #ddd;
    text-align: center;
}}
tr.best {{ background-color: {_GREEN_BG}; }}
tr.worst {{ background-color: {_RED_BG}; }}
.positive {{ background-color: {_GREEN_BG}; color: {_GREEN_TEXT}; font-weight: 600; }}
.negative {{ background-color: {_RED_BG}; color: {_RED_TEXT}; font-weight: 600; }}
.mono {{
    font-family: 'SF Mono', 'Fira Code', Monaco, Consolas, monospace;
    font-size: 0.88rem;
}}
.tile {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    font-weight: bold;
    font-size: 20px;
    color: #fff;
    border-radius: 4px;
    margin: 2px;
    text-transform: uppercase;
}}
.tile-row {{
    display: flex;
    gap: 4px;
    margin: 4px 0;
}}
.bar-chart {{
    margin: 12px 0;
}}
.bar-row {{
    display: flex;
    align-items: center;
    margin: 3px 0;
    font-size: 0.85rem;
}}
.bar-label {{
    width: 60px;
    text-align: right;
    padding-right: 8px;
    font-family: 'SF Mono', Monaco, monospace;
}}
.bar-container {{
    flex: 1;
    position: relative;
    height: 22px;
}}
.bar {{
    height: 22px;
    display: inline-block;
    border-radius: 3px;
    min-width: 2px;
}}
.bar-value {{
    padding-left: 6px;
    font-size: 0.82rem;
    color: #666;
    font-family: 'SF Mono', Monaco, monospace;
}}
.adv-chart {{
    margin: 16px 0;
}}
.adv-row {{
    display: flex;
    align-items: center;
    margin: 3px 0;
    font-size: 0.85rem;
}}
.adv-label {{
    width: 60px;
    text-align: right;
    padding-right: 8px;
    font-family: 'SF Mono', Monaco, monospace;
}}
.adv-track {{
    flex: 1;
    position: relative;
    height: 22px;
    background: #f0f0f0;
    border-radius: 3px;
}}
.adv-bar {{
    position: absolute;
    top: 0;
    height: 22px;
    border-radius: 3px;
}}
.adv-value {{
    padding-left: 6px;
    width: 70px;
    font-size: 0.82rem;
    color: #666;
    font-family: 'SF Mono', Monaco, monospace;
}}
.arrow-up {{ color: {_GREEN_TEXT}; }}
.arrow-down {{ color: {_RED_TEXT}; }}
.insight {{
    background: #f8f9fa;
    border-left: 3px solid {_GREEN};
    padding: 10px 14px;
    margin: 14px 0;
    font-size: 0.9rem;
    color: #444;
}}
.kl-badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.85rem;
    margin: 4px 0;
}}
.summary-box {{
    background: linear-gradient(135deg, #f0faf0, #f8f8ff);
    border: 1px solid #c8e6c9;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 0.95rem;
}}
.token-seq {{
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.82rem;
    background: #f5f5f5;
    padding: 8px 12px;
    border-radius: 4px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    color: #555;
    margin: 8px 0;
}}
.collapsible {{
    cursor: pointer;
    padding: 14px 20px;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin: 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.collapsible:hover {{ background: #f9f9f9; }}
.collapsible-content {{
    display: none;
    padding: 0;
}}
.collapsible-content.open {{
    display: block;
}}
.step-header {{
    font-weight: 600;
    font-size: 1rem;
}}
.step-metrics {{
    font-size: 0.85rem;
    color: #666;
    font-family: 'SF Mono', Monaco, monospace;
}}
.toggle-arrow {{
    font-size: 0.85rem;
    color: #999;
    transition: transform 0.2s;
}}
"""

_JS_COLLAPSIBLE = """\
function toggleStep(id) {
    var el = document.getElementById(id);
    var arrow = document.getElementById(id + '-arrow');
    if (el.classList.contains('open')) {
        el.classList.remove('open');
        arrow.textContent = '\\u25B6';
    } else {
        el.classList.add('open');
        arrow.textContent = '\\u25BC';
    }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text, quote=True)


def _sign(value: float) -> str:
    """Format a float with explicit sign."""
    return f"+{value:.4f}" if value >= 0 else f"{value:.4f}"


def _pct_change(old: float, new: float) -> str:
    """Percentage change string."""
    if old == 0:
        return "n/a"
    change = (new - old) / old * 100
    return f"{change:+.1f}%"


def _css_class_for_value(value: float) -> str:
    """Return 'positive' or 'negative' CSS class based on sign."""
    return "positive" if value >= 0 else "negative"


def _color_for_log_prob(lp: float) -> str:
    """Map a log prob to a background color (deeper red = less likely)."""
    # log probs are negative; closer to 0 = more likely
    # Clamp to [-5, 0] range for coloring
    clamped = max(-5.0, min(0.0, lp))
    # 0 -> green, -5 -> red
    t = -clamped / 5.0  # 0..1
    # Interpolate from green-ish to red-ish background
    r = int(232 + t * (255 - 232))
    g = int(245 - t * (245 - 235))
    b = int(233 - t * (233 - 238))
    return f"rgb({r},{g},{b})"


def _color_for_log_prob_text(lp: float) -> str:
    """Text color for log prob cell."""
    clamped = max(-5.0, min(0.0, lp))
    t = -clamped / 5.0
    if t > 0.5:
        return _RED_TEXT
    return _GREEN_TEXT


# ---------------------------------------------------------------------------
# Section 1: Game State
# ---------------------------------------------------------------------------


def _render_game_state(step_data: GRPOStepData) -> str:
    """Render the game state section with Wordle tiles."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append(
        f"<h2>1. Game State</h2>"
        f'<div class="desc">Step {step_data.step} &mdash; '
        f"The model sees this game state and must guess a word</div>"
    )

    # Parse game_state_tokens to extract guesses and feedback
    tokens = step_data.game_state_tokens
    # Format: [bos] g u e s s [green] [gray] ... [sep] ... [eos]
    # We need to reconstruct guess rows from the token sequence

    tile_rows = _parse_tiles_from_tokens(tokens)

    if tile_rows:
        for row in tile_rows:
            lines.append('<div class="tile-row">')
            for letter, color in row:
                bg = {
                    "green": _GREEN,
                    "yellow": _YELLOW,
                    "gray": _GRAY,
                }.get(color, _EMPTY)
                lines.append(f'<div class="tile" style="background:{bg};">{_esc(letter.upper())}</div>')
            lines.append("</div>")
    else:
        lines.append('<div style="color:#888;font-style:italic;">No prior guesses &mdash; this is the first turn</div>')

    # Show raw token sequence
    lines.append('<div class="token-seq">')
    lines.append(_esc(" ".join(tokens)))
    lines.append("</div>")

    lines.append("</div>")
    return "\n".join(lines)


def _parse_tiles_from_tokens(tokens: list[str]) -> list[list[tuple[str, str]]]:
    """Parse game state tokens into rows of (letter, feedback_color) pairs.

    Returns a list of rows, each row a list of 5 (letter, color) tuples.
    If the token sequence represents the start of a game with no guesses,
    returns an empty list.
    """
    rows: list[list[tuple[str, str]]] = []

    # Strip [bos] and [eos]
    inner = [t for t in tokens if t not in ("[bos]", "[eos]")]

    # Split on [sep]
    segments: list[list[str]] = []
    current: list[str] = []
    for t in inner:
        if t == "[sep]":
            if current:
                segments.append(current)
            current = []
        else:
            current.append(t)
    if current:
        segments.append(current)

    for seg in segments:
        # Each segment should be: letter letter letter letter letter [green] [yellow] ...
        letters: list[str] = []
        feedbacks: list[str] = []
        for t in seg:
            if t.startswith("[") and t.endswith("]"):
                fb_name = t[1:-1]  # green, yellow, gray
                if fb_name in ("green", "yellow", "gray"):
                    feedbacks.append(fb_name)
            else:
                if not feedbacks:  # Only collect letters before feedback tokens
                    letters.append(t)

        if letters and feedbacks and len(letters) == len(feedbacks):
            rows.append(list(zip(letters, feedbacks, strict=True)))

    return rows


# ---------------------------------------------------------------------------
# Section 2: Group of Completions
# ---------------------------------------------------------------------------


def _render_completions(step_data: GRPOStepData) -> str:
    """Render the completions table with per-character log probs."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append(
        "<h2>2. Group of Completions</h2>"
        '<div class="desc">The model generated '
        f"{len(step_data.completions)} candidate guesses. "
        "Each cell shows a character colored by its log probability "
        "(greener = more likely).</div>"
    )

    # Sort by reward (best first), keep track of indices
    indexed = list(enumerate(step_data.completions))
    indexed.sort(key=lambda x: x[1].reward, reverse=True)
    best_idx = indexed[0][0]
    worst_idx = indexed[-1][0]

    lines.append("<table>")
    lines.append("<tr><th>#</th><th>Guess</th>")
    # Column per character position
    for i in range(5):
        lines.append(f"<th>Char {i + 1}</th>")
    lines.append("<th>Total Log Prob</th><th>Reward</th></tr>")

    for rank, (orig_idx, comp) in enumerate(indexed):
        row_class = ""
        if orig_idx == best_idx and len(indexed) > 1:
            row_class = ' class="best"'
        elif orig_idx == worst_idx and len(indexed) > 1:
            row_class = ' class="worst"'

        total_lp = sum(comp.log_probs) if comp.log_probs else 0.0

        lines.append(f"<tr{row_class}>")
        lines.append(f"<td>{rank + 1}</td>")
        lines.append(f'<td class="mono" style="font-weight:bold;">{_esc(comp.text.upper())}</td>')

        # Per-character log probs
        for i in range(min(5, len(comp.log_probs))):
            lp = comp.log_probs[i]
            bg = _color_for_log_prob(lp)
            tc = _color_for_log_prob_text(lp)
            char = comp.tokens[i] if i < len(comp.tokens) else "?"
            lines.append(
                f'<td class="mono" style="background:{bg};color:{tc};">'
                f"{_esc(char.upper())} <small>({lp:.2f})</small></td>"
            )
        # Pad if fewer than 5
        for _ in range(5 - min(5, len(comp.log_probs))):
            lines.append("<td>&mdash;</td>")

        lines.append(f'<td class="mono">{total_lp:.3f}</td>')
        cls = _css_class_for_value(comp.reward)
        lines.append(f'<td class="{cls} mono">{comp.reward:+.3f}</td>')
        lines.append("</tr>")

    lines.append("</table>")
    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: Reward Scoring
# ---------------------------------------------------------------------------


def _render_rewards(step_data: GRPOStepData) -> str:
    """Render the reward breakdown table and distribution chart."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append(
        "<h2>3. Reward Scoring</h2>"
        '<div class="desc">Each completion is scored by the reward function. '
        "The breakdown shows which components contributed to the total reward.</div>"
    )

    # Gather all breakdown keys across completions
    all_keys: list[str] = []
    seen: set[str] = set()
    for comp in step_data.completions:
        for k in comp.reward_breakdown:
            if k not in seen and k != "total":
                all_keys.append(k)
                seen.add(k)

    # Table
    lines.append("<table>")
    header = "<tr><th>Guess</th>"
    for k in all_keys:
        header += f"<th>{_esc(k.replace('_', ' ').title())}</th>"
    header += "<th>Total Reward</th></tr>"
    lines.append(header)

    sorted_comps = sorted(step_data.completions, key=lambda c: c.reward, reverse=True)

    for comp in sorted_comps:
        lines.append("<tr>")
        lines.append(f'<td class="mono" style="font-weight:bold;">{_esc(comp.text.upper())}</td>')
        for k in all_keys:
            val = comp.reward_breakdown.get(k, 0.0)
            cls = _css_class_for_value(val) if val != 0 else ""
            cls_attr = f' class="{cls} mono"' if cls else ' class="mono"'
            lines.append(f"<td{cls_attr}>{val:+.2f}</td>")
        cls = _css_class_for_value(comp.reward)
        lines.append(f'<td class="{cls} mono" style="font-weight:bold;">{comp.reward:+.3f}</td>')
        lines.append("</tr>")

    lines.append("</table>")

    # Horizontal bar chart of rewards
    max_abs = max(abs(r) for r in step_data.rewards) if step_data.rewards else 1.0
    if max_abs == 0:
        max_abs = 1.0

    lines.append('<div class="bar-chart">')
    for comp in sorted_comps:
        width = abs(comp.reward) / max_abs * 100
        color = _GREEN if comp.reward >= 0 else _RED
        lines.append('<div class="bar-row">')
        lines.append(f'<div class="bar-label">{_esc(comp.text.upper())}</div>')
        lines.append(
            f'<div class="bar-container"><div class="bar" style="width:{width:.1f}%;background:{color};"></div></div>'
        )
        lines.append(f'<div class="bar-value">{comp.reward:+.3f}</div>')
        lines.append("</div>")
    lines.append("</div>")

    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Advantage Computation
# ---------------------------------------------------------------------------


def _render_advantages(step_data: GRPOStepData) -> str:
    """Render the advantage computation section."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append(
        "<h2>4. Advantage Computation</h2>"
        '<div class="desc">Advantages normalize rewards relative to the group. '
        "Positive = better than average (reinforce), negative = worse than average (suppress).</div>"
    )

    lines.append(
        f'<div style="margin:8px 0;">'
        f'<span class="mono">Group Mean: {step_data.group_mean:.4f}</span> &nbsp; '
        f'<span class="mono">Group Std: {step_data.group_std:.4f}</span></div>'
    )

    # Table
    lines.append("<table>")
    lines.append("<tr><th>Guess</th><th>Raw Reward</th><th>Advantage (normalized)</th></tr>")

    # Sort by advantage (best first)
    indexed = sorted(
        zip(step_data.completions, step_data.rewards, step_data.advantages, strict=True),
        key=lambda x: x[2],
        reverse=True,
    )

    for comp, reward, adv in indexed:
        r_cls = _css_class_for_value(reward)
        a_cls = _css_class_for_value(adv)
        lines.append("<tr>")
        lines.append(f'<td class="mono" style="font-weight:bold;">{_esc(comp.text.upper())}</td>')
        lines.append(f'<td class="{r_cls} mono">{reward:+.3f}</td>')
        lines.append(f'<td class="{a_cls} mono">{_sign(adv)}</td>')
        lines.append("</tr>")

    lines.append("</table>")

    # Advantage bar chart (centered at 0)
    max_abs_adv = max(abs(a) for a in step_data.advantages) if step_data.advantages else 1.0
    if max_abs_adv == 0:
        max_abs_adv = 1.0

    lines.append('<div class="adv-chart">')
    for comp, _reward, adv in indexed:
        bar_width = abs(adv) / max_abs_adv * 45  # max 45% of track width
        color = _GREEN if adv >= 0 else _RED

        if adv >= 0:
            # Bar extends right from center
            left = 50.0
            style = f"left:{left:.1f}%;width:{bar_width:.1f}%;background:{color};"
        else:
            # Bar extends left from center
            left = 50.0 - bar_width
            style = f"left:{left:.1f}%;width:{bar_width:.1f}%;background:{color};"

        lines.append('<div class="adv-row">')
        lines.append(f'<div class="adv-label">{_esc(comp.text.upper())}</div>')
        lines.append(
            f'<div class="adv-track">'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;'
            f'width:1px;background:#ccc;"></div>'
            f'<div class="adv-bar" style="{style}"></div>'
            f"</div>"
        )
        lines.append(f'<div class="adv-value">{_sign(adv)}</div>')
        lines.append("</div>")
    lines.append("</div>")

    lines.append(
        '<div class="insight">Completions above the group mean get positive '
        "advantages (reinforced), below get negative (suppressed).</div>"
    )

    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Policy Update
# ---------------------------------------------------------------------------


def _render_policy_update(step_data: GRPOStepData) -> str:
    """Render the policy update section showing probability changes."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append(
        "<h2>5. Policy Update</h2>"
        '<div class="desc">How the model\'s probability distribution shifted '
        "after this training step. Green arrows = reinforced, red arrows = suppressed.</div>"
    )

    lines.append("<table>")
    lines.append(
        "<tr><th>Guess</th><th>Old Probability</th><th>New Probability</th><th>Change</th><th>Direction</th></tr>"
    )

    # Pair completions with old/new probs
    items = list(
        zip(
            step_data.completions,
            step_data.old_probs,
            step_data.new_probs,
            step_data.advantages,
            strict=True,
        )
    )
    items.sort(key=lambda x: x[2] - x[1], reverse=True)  # Sort by change

    for comp, old_p, new_p, _adv in items:
        change = new_p - old_p
        direction = ""
        if change > 1e-8:
            direction = '<span class="arrow-up">&#9650; Reinforced</span>'
        elif change < -1e-8:
            direction = '<span class="arrow-down">&#9660; Suppressed</span>'
        else:
            direction = '<span style="color:#888;">&#9644; Unchanged</span>'

        change_cls = _css_class_for_value(change) if abs(change) > 1e-8 else ""
        change_attr = f' class="{change_cls} mono"' if change_cls else ' class="mono"'

        lines.append("<tr>")
        lines.append(f'<td class="mono" style="font-weight:bold;">{_esc(comp.text.upper())}</td>')
        lines.append(f'<td class="mono">{old_p:.6f}</td>')
        lines.append(f'<td class="mono">{new_p:.6f}</td>')
        lines.append(f"<td{change_attr}>{_sign(change)}</td>")
        lines.append(f"<td>{direction}</td>")
        lines.append("</tr>")

    lines.append("</table>")

    # KL divergence
    kl = step_data.kl_divergence
    if abs(kl) < 0.01:
        kl_interp = "very small &mdash; policy barely changed from reference"
        kl_color = _GREEN_BG
    elif abs(kl) < 0.1:
        kl_interp = "moderate &mdash; policy is diverging measurably from reference"
        kl_color = "#fff8e1"
    else:
        kl_interp = "large &mdash; policy has diverged significantly from reference"
        kl_color = _RED_BG

    lines.append(
        f'<div style="margin-top:12px;">'
        f'<span class="kl-badge" style="background:{kl_color};">'
        f"KL Divergence: {kl:.6f}</span> "
        f'<span style="font-size:0.88rem;color:#666;">{kl_interp}</span>'
        f"</div>"
    )

    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: Summary
# ---------------------------------------------------------------------------


def _render_summary(step_data: GRPOStepData) -> str:
    """Render the summary section."""
    lines: list[str] = []
    lines.append('<div class="section">')
    lines.append("<h2>6. Summary</h2>")

    # Find best and worst
    sorted_by_reward = sorted(step_data.completions, key=lambda c: c.reward, reverse=True)
    best = sorted_by_reward[0]
    worst = sorted_by_reward[-1]

    # Determine the reason from reward
    if best.reward > worst.reward:
        reason = f"it scored a reward of {best.reward:+.3f} vs {worst.reward:+.3f}"
    else:
        reason = "both scored equally"

    lines.append(
        '<div class="summary-box">'
        f"The model learned to prefer "
        f"<strong>{_esc(best.text.upper())}</strong> over "
        f"<strong>{_esc(worst.text.upper())}</strong> because {reason}."
        f"</div>"
    )

    kl = step_data.kl_divergence
    lines.append(
        f'<div style="margin-top:8px;font-size:0.9rem;color:#555;">'
        f"KL divergence from reference policy: "
        f'<span class="mono">{kl:.6f}</span>'
    )
    if abs(kl) < 0.01:
        lines.append(" &mdash; the policy is staying close to the reference (stable learning).")
    elif abs(kl) < 0.1:
        lines.append(" &mdash; moderate divergence (the policy is adapting).")
    else:
        lines.append(" &mdash; significant divergence (consider reducing learning rate or increasing KL penalty).")
    lines.append("</div>")

    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_grpo_step_html(step_data: GRPOStepData) -> str:
    """Render a single GRPO training step as a detailed, scrollable HTML page.

    The page tells the story of one GRPO step, section by section:

    1. GAME STATE - Wordle board with colored tiles
    2. GROUP OF COMPLETIONS - candidate guesses with per-character log probs
    3. REWARD SCORING - reward breakdown and distribution
    4. ADVANTAGE COMPUTATION - normalized advantages with bar chart
    5. POLICY UPDATE - probability shifts with direction arrows
    6. SUMMARY - one-line takeaway and KL interpretation
    """
    sections = [
        _render_game_state(step_data),
        _render_completions(step_data),
        _render_rewards(step_data),
        _render_advantages(step_data),
        _render_policy_update(step_data),
        _render_summary(step_data),
    ]

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GRPO Step Inspector &mdash; Step {step_data.step}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>GRPO Step Inspector</h1>
<div class="subtitle">Training step {step_data.step} &mdash; \
{len(step_data.completions)} completions &mdash; \
group mean reward {step_data.group_mean:.3f}</div>
{body}
</body>
</html>"""


def render_grpo_trajectory_html(step_data_list: list[GRPOStepData]) -> str:
    """Render multiple GRPO steps as a scrollable timeline.

    Shows how the policy evolves across training steps.
    Each step is a collapsible section with the full inspector view.
    Header shows step number and key metrics (reward mean, KL, entropy).
    """
    step_sections: list[str] = []

    for i, step_data in enumerate(step_data_list):
        step_id = f"step-{i}"

        # Metrics summary for the header
        reward_mean = step_data.group_mean
        kl = step_data.kl_divergence
        n_comp = len(step_data.completions)

        header = (
            f'<div class="collapsible" onclick="toggleStep(\'{step_id}\')">'
            f'<div><span class="step-header">Step {step_data.step}</span></div>'
            f'<div class="step-metrics">'
            f"reward: {reward_mean:+.3f} &nbsp; KL: {kl:.4f} &nbsp; "
            f"completions: {n_comp}"
            f"</div>"
            f'<div class="toggle-arrow" id="{step_id}-arrow">&#9654;</div>'
            f"</div>"
        )

        # Full step content (without the outer <html> wrapper)
        content_sections = [
            _render_game_state(step_data),
            _render_completions(step_data),
            _render_rewards(step_data),
            _render_advantages(step_data),
            _render_policy_update(step_data),
            _render_summary(step_data),
        ]
        content = "\n".join(content_sections)

        step_sections.append(f'{header}\n<div class="collapsible-content" id="{step_id}">{content}</div>')

    all_steps = "\n".join(step_sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GRPO Training Trajectory &mdash; {len(step_data_list)} Steps</title>
<style>{_CSS}</style>
<script>{_JS_COLLAPSIBLE}</script>
</head>
<body>
<h1>GRPO Training Trajectory</h1>
<div class="subtitle">{len(step_data_list)} training steps &mdash; \
click a step to expand</div>
{all_steps}
</body>
</html>"""


def render_grpo_step_from_file(path: str, output_path: str) -> None:
    """Load a GRPOStepData JSON file and render it as HTML.

    Args:
        path: Path to a GRPOStepData JSON file.
        output_path: Path where the HTML file will be written.
    """
    step_data = GRPOStepData.load(Path(path))
    html_content = render_grpo_step_html(step_data)
    Path(output_path).write_text(html_content)


def render_grpo_trajectory_from_dir(dir_path: str, output_path: str) -> None:
    """Load all GRPOStepData JSON files from a directory and render as timeline.

    Files are sorted by step number. Expects filenames like step-100.json.

    Args:
        dir_path: Directory containing GRPOStepData JSON files.
        output_path: Path where the HTML file will be written.
    """
    directory = Path(dir_path)
    json_files = sorted(directory.glob("*.json"))

    step_data_list: list[GRPOStepData] = []
    for f in json_files:
        step_data_list.append(GRPOStepData.load(f))

    # Sort by step number
    step_data_list.sort(key=lambda s: s.step)

    html_content = render_grpo_trajectory_html(step_data_list)
    Path(output_path).write_text(html_content)
