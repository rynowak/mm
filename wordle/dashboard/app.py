"""Live training dashboard for Wordle RL fine-tuning.

Usage:
    uv run python wordle/dashboard/app.py --run-dir runs/finetune-grpo/20260602_160145
    # Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from mm_viz import (
    GameReplay,
    GRPOStepData,
    render_game_html,
    render_grpo_step_html,
)

app = FastAPI()
RUN_DIR: Path = Path(".")


def _find_all_step_files() -> list[Path]:
    step_dir = RUN_DIR / "step_data"
    if not step_dir.exists():
        return []
    return sorted(step_dir.glob("step-*.json"), key=lambda f: int(f.stem.split("-")[1]))


def _find_all_evals() -> list[tuple[int, float, float]]:
    """Return (step, win_rate, avg_guesses) for all eval snapshots."""
    results = []
    for d in sorted(RUN_DIR.glob("eval-*"), key=lambda d: int(d.name.split("-")[1])):
        sf = d / "snapshot.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            results.append((
                data.get("step", 0),
                data.get("win_rate", 0.0),
                data.get("avg_guesses", 0.0),
            ))
    return results


def _load_eval_replays(step: int) -> list[GameReplay]:
    sf = RUN_DIR / f"eval-{step}" / "snapshot.json"
    if not sf.exists():
        return []
    data = json.loads(sf.read_text())
    return [
        GameReplay(
            target=r["target"],
            guesses=r["guesses"],
            feedback=r["feedback"],
            solved=r["solved"],
            turns=r["turns"],
        )
        for r in data.get("replays", [])
    ]


def _layout(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px;
               background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        nav {{ background: #333; padding: 10px 20px; margin: -20px -20px 20px; }}
        nav a {{ color: #fff; text-decoration: none; margin-right: 20px;
                font-weight: bold; }}
        nav a:hover {{ color: #6aaa64; }}
        .card {{ background: white; border-radius: 8px; padding: 20px;
                margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .game-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
        .game-card {{ flex: 0 0 200px; }}
        .meta {{ color: #666; font-size: 14px; }}
        .chart-bar {{ display: inline-block; background: #6aaa64; height: 20px;
                     margin: 2px 0; border-radius: 3px; }}
        table {{ border-collapse: collapse; }}
        td, th {{ padding: 4px 12px; text-align: left;
                 border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/games">Games</a>
        <a href="/grpo">GRPO Inspector</a>
    </nav>
    <div class="container">{body}</div>
</body>
</html>"""


# --- Main dashboard ---

@app.get("/", response_class=HTMLResponse)
async def index():
    body = """
    <h1>Wordle Training Dashboard</h1>
    <div hx-get="/fragment/dashboard" hx-trigger="load, every 10s"
         hx-swap="innerHTML">Loading...</div>
    """
    return HTMLResponse(_layout("Dashboard", body))


@app.get("/fragment/dashboard", response_class=HTMLResponse)
async def fragment_dashboard():
    evals = _find_all_evals()
    step_files = _find_all_step_files()
    latest_step = int(step_files[-1].stem.split("-")[1]) if step_files else 0

    html = f'<p class="meta">Run: {RUN_DIR.name} | Steps: {latest_step}</p>'

    # Win rate history
    if evals:
        html += "<div class='card'><h2>Eval Win Rate</h2><table>"
        html += "<tr><th>Step</th><th>Win Rate</th><th></th></tr>"
        for step, wr, _ag in evals:
            bar_w = int(wr * 300)
            html += (
                f"<tr><td>{step}</td><td>{wr:.0%}</td>"
                f'<td><div class="chart-bar" style="width:{bar_w}px"></div></td></tr>'
            )
        html += "</table></div>"

    # Latest eval games
    if evals:
        latest_step_eval = evals[-1][0]
        replays = _load_eval_replays(latest_step_eval)
        wins = sum(1 for r in replays if r.solved)
        html += f"<div class='card'><h2>Latest Eval (step {latest_step_eval})"
        html += f" — {wins}/{len(replays)} wins</h2>"
        html += "<div class='game-grid'>"
        for r in replays[:12]:
            html += f'<div class="game-card">{render_game_html(r)}'
            html += f'<p class="meta">{r.target}</p></div>'
        html += "</div></div>"

    return HTMLResponse(html)


# --- Games page ---

@app.get("/games", response_class=HTMLResponse)
async def games_page():
    body = """
    <h1>Game Replays</h1>
    <div hx-get="/fragment/games" hx-trigger="load, every 15s"
         hx-swap="innerHTML">Loading...</div>
    """
    return HTMLResponse(_layout("Games", body))


@app.get("/fragment/games", response_class=HTMLResponse)
async def fragment_games():
    evals = _find_all_evals()
    if not evals:
        return HTMLResponse("<p>No eval data yet</p>")

    html = "<p>Select an eval checkpoint:</p><ul>"
    for step, wr, _ag in reversed(evals):
        html += f'<li><a href="/games/{step}">Step {step} — {wr:.0%} win rate</a></li>'
    html += "</ul>"
    return HTMLResponse(html)


@app.get("/games/{step_num}", response_class=HTMLResponse)
async def games_step(step_num: int):
    replays = _load_eval_replays(step_num)
    if not replays:
        return HTMLResponse(_layout("Games", f"<p>No games for step {step_num}</p>"))

    wins = [r for r in replays if r.solved]
    losses = [r for r in replays if not r.solved]

    body = f"<h1>Step {step_num} — {len(wins)} wins, {len(losses)} losses</h1>"

    if wins:
        body += "<h2>Wins</h2><div class='game-grid'>"
        for r in wins:
            board = render_game_html(r)
            body += f'<div class="game-card">{board}'
            body += f'<p class="meta">{r.target} in {r.turns}</p></div>'
        body += "</div>"

    if losses:
        body += "<h2>Losses</h2><div class='game-grid'>"
        for r in losses[:20]:
            body += f'<div class="game-card">{render_game_html(r)}'
            body += f'<p class="meta">{r.target}</p></div>'
        body += "</div>"

    return HTMLResponse(_layout(f"Games — Step {step_num}", body))


# --- GRPO Inspector ---

@app.get("/grpo", response_class=HTMLResponse)
async def grpo_page():
    step_files = _find_all_step_files()
    if not step_files:
        return HTMLResponse(_layout("GRPO", "<p>No step data yet</p>"))

    body = "<h1>GRPO Inspector</h1><div class='card'><ul>"
    for f in reversed(step_files[-20:]):
        s = f.stem.split("-")[1]
        body += f'<li><a href="/grpo/{s}">Step {s}</a></li>'
    body += "</ul></div>"
    return HTMLResponse(_layout("GRPO", body))


@app.get("/grpo/{step_num}", response_class=HTMLResponse)
async def grpo_step(step_num: int):
    f = RUN_DIR / "step_data" / f"step-{step_num}.json"
    if not f.exists():
        return HTMLResponse(_layout("GRPO", f"<p>Step {step_num} not found</p>"))
    step_data = GRPOStepData.load(f)
    html = render_grpo_step_html(step_data)
    return HTMLResponse(_layout(f"GRPO Step {step_num}", html))


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Wordle training dashboard")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    RUN_DIR = Path(args.run_dir)
    print(f"Dashboard: {RUN_DIR}")
    print(f"Open http://localhost:{args.port}")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
