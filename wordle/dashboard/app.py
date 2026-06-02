"""Live training dashboard for Wordle RL fine-tuning.

Usage:
    uv run python wordle/dashboard/app.py --run-dir runs/finetune-grpo/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from mm_viz import (
    GameReplay,
    render_game_html,
)

app = FastAPI()
RUN_DIR: Path = Path(".")


def _all_step_files() -> list[Path]:
    d = RUN_DIR / "step_data"
    if not d.exists():
        return []
    return sorted(d.glob("step-*.json"), key=lambda f: int(f.stem.split("-")[1]))


def _all_evals() -> list[tuple[int, float, float]]:
    results = []
    for d in sorted(RUN_DIR.glob("eval-*"), key=lambda d: int(d.name.split("-")[1])):
        sf = d / "snapshot.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            results.append((data.get("step", 0), data.get("win_rate", 0.0), data.get("avg_guesses", 0.0)))
    return results


def _load_replays(step: int) -> list[GameReplay]:
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


def _load_step(path: Path) -> dict:
    return json.loads(path.read_text())


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, sans-serif; margin: 0; background: #1a1a2e; color: #eee; }
.top-bar { background: #16213e; padding: 8px 20px; display: flex; align-items: center; gap: 20px; }
.top-bar h1 { font-size: 18px; margin: 0; color: #6aaa64; }
.top-bar .stat { font-size: 14px; color: #aaa; }
.top-bar .stat b { color: #6aaa64; font-size: 18px; }
.main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; height: calc(100vh - 44px); }
.panel { background: #16213e; border-radius: 8px; padding: 12px; overflow-y: auto; }
.panel h2 { margin: 0 0 8px; font-size: 15px; color: #6aaa64; border-bottom: 1px solid #333; padding-bottom: 4px; }
.game-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.game-card { flex: 0 0 160px; font-size: 12px; }
.game-card .meta { color: #888; font-size: 11px; margin: 2px 0; }
.grpo-group { margin: 8px 0; }
.grpo-row { display: flex; align-items: center; gap: 8px; padding: 4px 0;
           border-bottom: 1px solid #222; font-size: 13px; font-family: monospace; }
.grpo-word { width: 60px; font-weight: bold; }
.grpo-reward { width: 60px; text-align: right; }
.grpo-adv { width: 80px; text-align: right; }
.grpo-bar { height: 14px; border-radius: 2px; min-width: 2px; }
.positive { color: #6aaa64; }
.negative { color: #c0392b; }
.chart-row { display: flex; align-items: center; gap: 6px; font-size: 12px; padding: 2px 0; }
.chart-label { width: 50px; text-align: right; color: #888; }
.chart-bar { height: 16px; background: #6aaa64; border-radius: 2px; }
.chart-val { width: 40px; color: #aaa; }
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Wordle Live</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>{CSS}</style>
</head><body>
<div class="top-bar"
     hx-get="/frag/topbar" hx-trigger="every 5s" hx-swap="innerHTML">
    <h1>Wordle Training</h1><span class="stat">Loading...</span>
</div>
<div class="main">
    <div class="panel" hx-get="/frag/grpo" hx-trigger="load, every 5s" hx-swap="innerHTML">
        Loading GRPO...
    </div>
    <div class="panel" hx-get="/frag/games" hx-trigger="load, every 10s" hx-swap="innerHTML">
        Loading games...
    </div>
</div>
</body></html>""")


@app.get("/frag/topbar", response_class=HTMLResponse)
async def frag_topbar():
    evals = _all_evals()
    steps = _all_step_files()
    latest = int(steps[-1].stem.split("-")[1]) if steps else 0

    html = "<h1>Wordle Training</h1>"
    html += f'<span class="stat">Step <b>{latest}</b></span>'

    if evals:
        _, wr, ag = evals[-1]
        html += f'<span class="stat">Eval Win Rate <b>{wr:.0%}</b></span>'
        html += f'<span class="stat">Avg Guesses <b>{ag:.1f}</b></span>'

    if len(evals) > 1:
        prev_wr = evals[-2][1]
        curr_wr = evals[-1][1]
        delta = curr_wr - prev_wr
        color = "positive" if delta >= 0 else "negative"
        html += f'<span class="stat {color}">({delta:+.0%})</span>'

    return HTMLResponse(html)


@app.get("/frag/grpo", response_class=HTMLResponse)
async def frag_grpo():
    files = _all_step_files()
    if not files:
        return HTMLResponse("<h2>GRPO Steps</h2><p>Waiting for data...</p>")

    # Show last 3 steps
    html = "<h2>Latest GRPO Steps</h2>"
    for f in reversed(files[-3:]):
        data = _load_step(f)
        step = data.get("step", 0)
        kl = data.get("kl_divergence", 0)
        completions = data.get("completions", [])
        rewards = data.get("rewards", [])
        advantages = data.get("advantages", [])
        game_text = data.get("game_state_text", "")

        html += '<div class="grpo-group">'
        html += f"<b>Step {step}</b> "
        html += f'<span style="color:#888">KL={kl:.3f} | {game_text[:40]}</span>'

        if completions:
            # Sort by reward
            items = sorted(
                zip(completions, rewards, advantages, strict=True),
                key=lambda x: x[1],
                reverse=True,
            )
            for comp, rew, adv in items:
                word = comp.get("text", "?????")
                color = "positive" if adv >= 0 else "negative"
                bar_w = int(abs(adv) * 40)
                html += '<div class="grpo-row">'
                html += f'<span class="grpo-word">{word}</span>'
                html += f'<span class="grpo-reward">{rew:+.3f}</span>'
                html += f'<span class="grpo-adv {color}">{adv:+.3f}</span>'
                html += f'<div class="grpo-bar {color}" style="width:{bar_w}px"></div>'
                html += "</div>"

        html += "</div>"

    # Win rate chart from evals
    evals = _all_evals()
    if evals:
        html += "<h2>Eval Win Rate History</h2>"
        for step, wr, _ag in evals:
            bar_w = int(wr * 200)
            html += '<div class="chart-row">'
            html += f'<span class="chart-label">{step}</span>'
            html += f'<div class="chart-bar" style="width:{bar_w}px"></div>'
            html += f'<span class="chart-val">{wr:.0%}</span>'
            html += "</div>"

    return HTMLResponse(html)


@app.get("/frag/games", response_class=HTMLResponse)
async def frag_games():
    evals = _all_evals()
    if not evals:
        return HTMLResponse("<h2>Eval Games</h2><p>Waiting for first eval...</p>")

    latest_step = evals[-1][0]
    replays = _load_replays(latest_step)
    if not replays:
        return HTMLResponse("<h2>Eval Games</h2><p>No replays</p>")

    wins = [r for r in replays if r.solved]
    html = f"<h2>Step {latest_step} — {len(wins)}/{len(replays)} wins</h2>"
    html += '<div class="game-grid">'
    for r in replays:
        status = f"✓ {r.turns}" if r.solved else "✗"
        html += f'<div class="game-card">{render_game_html(r)}'
        html += f'<p class="meta">{r.target} {status}</p></div>'
    html += "</div>"

    return HTMLResponse(html)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser()
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
