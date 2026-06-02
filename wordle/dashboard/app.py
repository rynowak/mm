"""Live training dashboard for Wordle RL fine-tuning.

Watches the active run directory and serves game replays, GRPO step
inspector, and training metrics via FastAPI + htmx.

Usage:
    uv run python wordle/dashboard/app.py --run-dir runs/finetune-grpo/20260602_154122
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


def _find_latest_step_data() -> GRPOStepData | None:
    step_dir = RUN_DIR / "step_data"
    if not step_dir.exists():
        return None
    files = sorted(step_dir.glob("step-*.json"), key=lambda f: int(f.stem.split("-")[1]))
    if not files:
        return None
    return GRPOStepData.load(files[-1])


def _find_all_step_files() -> list[Path]:
    step_dir = RUN_DIR / "step_data"
    if not step_dir.exists():
        return []
    return sorted(step_dir.glob("step-*.json"), key=lambda f: int(f.stem.split("-")[1]))


def _find_latest_eval() -> list[GameReplay] | None:
    eval_dir = RUN_DIR / "eval"
    if not eval_dir.exists():
        return None
    files = sorted(eval_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        return None
    data = json.loads(files[-1].read_text())
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
        body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        nav {{ background: #333; padding: 10px 20px; margin: -20px -20px 20px; }}
        nav a {{ color: #fff; text-decoration: none; margin-right: 20px; font-weight: bold; }}
        nav a:hover {{ color: #6aaa64; }}
        .card {{ background: white; border-radius: 8px; padding: 20px;
                margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .game-grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
        .game-card {{ flex: 0 0 200px; }}
        .status {{ padding: 4px 8px; border-radius: 4px; font-size: 14px; font-weight: bold; }}
        .status-pass {{ background: #e8f5e9; color: #2e7d32; }}
        .status-fail {{ background: #ffebee; color: #c62828; }}
        .meta {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/games">Game Replays</a>
        <a href="/grpo">GRPO Inspector</a>
    </nav>
    <div class="container">
        {body}
    </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    step_files = _find_all_step_files()
    latest_step = int(step_files[-1].stem.split("-")[1]) if step_files else 0

    eval_replays = _find_latest_eval()
    if eval_replays:
        wins = sum(1 for r in eval_replays if r.solved)
        total = len(eval_replays)
        win_rate = wins / total if total else 0
        avg_guesses = sum(r.turns for r in eval_replays if r.solved) / max(wins, 1)
    else:
        win_rate = 0
        avg_guesses = 0
        total = 0
        wins = 0

    body = f"""
    <h1>Wordle Training Dashboard</h1>
    <p class="meta">Run: {RUN_DIR.name} | Parent: {RUN_DIR.parent.name}</p>

    <div class="card" hx-get="/fragment/stats" hx-trigger="every 10s" hx-swap="innerHTML">
        <h2>Training Progress</h2>
        <p>Latest step data: <strong>{latest_step}</strong></p>
        <p>GRPO step files: <strong>{len(step_files)}</strong></p>
        <p>Eval win rate: <strong>{win_rate:.1%}</strong> ({wins}/{total})</p>
        <p>Avg guesses (winners): <strong>{avg_guesses:.1f}</strong></p>
    </div>

    <div class="card">
        <h2>Recent Games</h2>
        <div class="game-grid" hx-get="/fragment/recent-games" hx-trigger="every 15s" hx-swap="innerHTML">
            {
        "".join(
            f'<div class="game-card">{render_game_html(r)}</div>' for r in (eval_replays[:8] if eval_replays else [])
        )
        or "<p>No eval games yet</p>"
    }
        </div>
    </div>
    """
    return HTMLResponse(_layout("Wordle Dashboard", body))


@app.get("/fragment/stats", response_class=HTMLResponse)
async def fragment_stats():
    step_files = _find_all_step_files()
    latest_step = int(step_files[-1].stem.split("-")[1]) if step_files else 0
    eval_replays = _find_latest_eval()
    if eval_replays:
        wins = sum(1 for r in eval_replays if r.solved)
        total = len(eval_replays)
        win_rate = wins / total if total else 0
        avg_guesses = sum(r.turns for r in eval_replays if r.solved) / max(wins, 1)
    else:
        win_rate = 0
        avg_guesses = 0
        total = 0
        wins = 0

    return HTMLResponse(f"""
        <h2>Training Progress</h2>
        <p>Latest step data: <strong>{latest_step}</strong></p>
        <p>GRPO step files: <strong>{len(step_files)}</strong></p>
        <p>Eval win rate: <strong>{win_rate:.1%}</strong> ({wins}/{total})</p>
        <p>Avg guesses (winners): <strong>{avg_guesses:.1f}</strong></p>
    """)


@app.get("/fragment/recent-games", response_class=HTMLResponse)
async def fragment_recent_games():
    eval_replays = _find_latest_eval()
    if not eval_replays:
        return HTMLResponse("<p>No eval games yet</p>")
    html = "".join(f'<div class="game-card">{render_game_html(r)}</div>' for r in eval_replays[:8])
    return HTMLResponse(html)


@app.get("/games", response_class=HTMLResponse)
async def games_page():
    eval_replays = _find_latest_eval()
    if not eval_replays:
        body = "<h1>Game Replays</h1><p>No eval games yet. Waiting for first eval checkpoint...</p>"
        return HTMLResponse(_layout("Games", body))

    wins = [r for r in eval_replays if r.solved]
    losses = [r for r in eval_replays if not r.solved]

    games_html = ""
    if wins:
        games_html += "<h2>Wins</h2><div class='game-grid'>"
        for r in wins:
            board = render_game_html(r)
            games_html += f'<div class="game-card">{board}<p class="meta">{r.target} in {r.turns}</p></div>'
        games_html += "</div>"

    if losses:
        games_html += "<h2>Losses</h2><div class='game-grid'>"
        for r in losses[:20]:
            games_html += f'<div class="game-card">{render_game_html(r)}<p class="meta">{r.target}</p></div>'
        games_html += "</div>"

    body = f"""
    <h1>Game Replays</h1>
    <p class="meta">{len(wins)} wins, {len(losses)} losses out of {len(eval_replays)} games</p>
    <div hx-get="/fragment/games-content" hx-trigger="every 15s" hx-swap="innerHTML">
        {games_html}
    </div>
    """
    return HTMLResponse(_layout("Games", body))


@app.get("/fragment/games-content", response_class=HTMLResponse)
async def fragment_games_content():
    eval_replays = _find_latest_eval()
    if not eval_replays:
        return HTMLResponse("<p>No eval games yet</p>")

    wins = [r for r in eval_replays if r.solved]
    losses = [r for r in eval_replays if not r.solved]

    html = ""
    if wins:
        html += "<h2>Wins</h2><div class='game-grid'>"
        for r in wins:
            html += f'<div class="game-card">{render_game_html(r)}<p class="meta">{r.target} in {r.turns}</p></div>'
        html += "</div>"
    if losses:
        html += "<h2>Losses</h2><div class='game-grid'>"
        for r in losses[:20]:
            html += f'<div class="game-card">{render_game_html(r)}<p class="meta">{r.target}</p></div>'
        html += "</div>"

    return HTMLResponse(html)


@app.get("/grpo", response_class=HTMLResponse)
async def grpo_page():
    step_files = _find_all_step_files()
    if not step_files:
        body = "<h1>GRPO Inspector</h1><p>No step data yet. Waiting for training to emit step data...</p>"
        return HTMLResponse(_layout("GRPO Inspector", body))

    body = "<h1>GRPO Inspector</h1>"
    body += "<div class='card'><h2>Available Steps</h2><ul>"
    for f in reversed(step_files[-20:]):
        step_num = f.stem.split("-")[1]
        body += f'<li><a href="/grpo/{step_num}">Step {step_num}</a></li>'
    body += "</ul></div>"

    return HTMLResponse(_layout("GRPO Inspector", body))


@app.get("/grpo/{step_num}", response_class=HTMLResponse)
async def grpo_step_page(step_num: int):
    step_file = RUN_DIR / "step_data" / f"step-{step_num}.json"
    if not step_file.exists():
        return HTMLResponse(_layout("GRPO Inspector", f"<p>Step {step_num} not found</p>"))

    step_data = GRPOStepData.load(step_file)
    inspector_html = render_grpo_step_html(step_data)

    return HTMLResponse(_layout(f"GRPO Step {step_num}", f"<h1>GRPO Step {step_num}</h1>{inspector_html}"))


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Wordle training dashboard")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    RUN_DIR = Path(args.run_dir)
    if not RUN_DIR.exists():
        print(f"Error: {RUN_DIR} does not exist")
        return

    print(f"Dashboard for: {RUN_DIR}")
    print(f"Open http://localhost:{args.port}")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
