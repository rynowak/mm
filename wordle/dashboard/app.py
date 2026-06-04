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
from mm_viz import GameReplay, render_game_html

app = FastAPI()
RUN_DIR: Path = Path(".")


def _load_live() -> dict | None:
    f = RUN_DIR / "live" / "latest.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _all_evals() -> list[tuple[int, float, float]]:
    results = []
    for d in sorted(RUN_DIR.glob("eval-*"), key=lambda d: int(d.name.split("-")[1])):
        sf = d / "snapshot.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            results.append(
                (
                    data.get("step", 0),
                    data.get("win_rate", 0.0),
                    data.get("avg_guesses", 0.0),
                )
            )
    return results


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #0f0f1a; color: #eee; }
.top { background: #16213e; padding: 10px 20px; display: flex;
       align-items: center; gap: 24px; border-bottom: 2px solid #6aaa64; }
.top h1 { font-size: 20px; color: #6aaa64; }
.top .s { font-size: 14px; color: #aaa; }
.top .s b { color: #fff; font-size: 16px; }
.main { padding: 12px; }
.section { margin-bottom: 16px; }
.section h2 { font-size: 14px; color: #6aaa64; margin-bottom: 8px;
              text-transform: uppercase; letter-spacing: 1px; }
.games { display: flex; flex-wrap: wrap; gap: 8px; }
.gc { background: #16213e; border-radius: 6px; padding: 6px;
      width: 160px; font-size: 11px; }
.gc .meta { color: #888; font-size: 11px; margin-top: 4px; }
.gc .win { color: #6aaa64; font-weight: bold; }
.gc .loss { color: #c0392b; font-weight: bold; }
.gc .turns { margin-top: 4px; font-family: monospace; font-size: 10px; }
.gc .turn-row { padding: 1px 0; }
.gc .positive { color: #6aaa64; }
.gc .negative { color: #c0392b; }
.wr-row { display: flex; align-items: center; gap: 8px;
          font-size: 13px; padding: 2px 0; }
.wr-label { width: 50px; text-align: right; color: #888; }
.wr-bar { height: 18px; background: #6aaa64; border-radius: 3px;
          transition: width 0.3s; }
.wr-val { width: 40px; color: #ccc; }
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Wordle Live</title>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>{CSS}</style>
</head><body>
<div class="top" hx-get="/f/top" hx-trigger="every 3s" hx-swap="innerHTML">
    <h1>Wordle</h1><span class="s">Loading...</span>
</div>
<div class="main">
    <div class="section" hx-get="/f/live" hx-trigger="load, every 3s" hx-swap="innerHTML">
        Loading...
    </div>
    <div class="section" hx-get="/f/history" hx-trigger="load, every 10s" hx-swap="innerHTML">
    </div>
</div>
</body></html>""")


@app.get("/f/top", response_class=HTMLResponse)
async def frag_top():
    live = _load_live()
    evals = _all_evals()

    step = live["step"] if live else 0
    n_games = len(live["games"]) if live else 0
    wins = sum(1 for g in live["games"] if g["solved"]) if live else 0

    html = "<h1>Wordle</h1>"
    html += f'<span class="s">Step <b>{step}</b></span>'
    html += f'<span class="s">Batch <b>{wins}/{n_games}</b> wins</span>'

    if live:
        loss = live.get("loss", 0)
        kl = live.get("kl_div", 0)
        clip = live.get("clip_fraction", 0)
        html += f'<span class="s">Loss <b>{loss:.4f}</b></span>'
        html += f'<span class="s">KL <b>{kl:.3f}</b></span>'
        html += f'<span class="s">Clip <b>{clip:.2f}</b></span>'

    if evals:
        _, wr, ag = evals[-1]
        html += f'<span class="s">Eval <b>{wr:.0%}</b></span>'

    return HTMLResponse(html)


@app.get("/f/live", response_class=HTMLResponse)
async def frag_live():
    live = _load_live()
    if not live:
        return HTMLResponse("<h2>Current Step</h2><p>Waiting...</p>")

    step = live["step"]
    games = live["games"]
    wins = sum(1 for g in games if g["solved"])

    html = f"<h2>Step {step} — {wins}/{len(games)} wins</h2>"
    html += '<div class="games">'
    for g in games:
        r = GameReplay(
            target=g["target"],
            guesses=g["guesses"],
            feedback=g["feedback"],
            solved=g["solved"],
            turns=g["turns"],
        )
        status_cls = "win" if r.solved else "loss"
        status_txt = f"solved in {r.turns}" if r.solved else "failed"
        turn_rewards = g.get("turn_rewards", [])
        reward_strs = [f"{tr:+.1f}" for tr in turn_rewards]
        reward_display = " ".join(reward_strs) if reward_strs else ""

        html += f'<div class="gc">{render_game_html(r)}'
        html += f'<p class="meta">{r.target} '
        html += f'<span class="{status_cls}">{status_txt}</span>'
        if reward_display:
            html += f' <span style="color:#888">[{reward_display}]</span>'
        html += "</p>"

        html += "</div>"
    html += "</div>"
    return HTMLResponse(html)


def _load_history() -> list[dict]:
    """Read training history from history.jsonl."""
    f = RUN_DIR / "live" / "history.jsonl"
    if not f.exists():
        return []
    lines = f.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line]


def _svg_line_chart(
    points: list[tuple[int, float]], width: int = 400, height: int = 120, color: str = "#6aaa64", label: str = ""
) -> str:
    """Render a simple SVG line chart."""
    if len(points) < 2:
        return ""
    steps = [p[0] for p in points]
    vals = [p[1] for p in points]
    min_s, max_s = min(steps), max(steps)
    min_v, max_v = min(vals), max(vals)
    v_range = max_v - min_v if max_v != min_v else 1.0
    s_range = max_s - min_s if max_s != min_s else 1.0

    pad = 40
    cw = width - pad
    ch = height - 20

    coords = []
    for s, v in points:
        x = pad + (s - min_s) / s_range * cw
        y = 10 + (1 - (v - min_v) / v_range) * ch
        coords.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(coords)
    svg = f'<svg width="{width}" height="{height}" style="background:#111;border-radius:4px">'
    if label:
        svg += f'<text x="{pad}" y="12" fill="#888" font-size="10">{label}</text>'
    svg += f'<text x="2" y="{10 + ch}" fill="#666" font-size="9">{min_v:.3f}</text>'
    svg += f'<text x="2" y="18" fill="#666" font-size="9">{max_v:.3f}</text>'
    svg += f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    svg += "</svg>"
    return svg


@app.get("/f/history", response_class=HTMLResponse)
async def frag_history():
    html = ""
    history = _load_history()

    if history:
        loss_pts = [(h["step"], h["loss"]) for h in history]
        kl_pts = [(h["step"], h["kl_div"]) for h in history]

        html += '<div style="display:flex;flex-wrap:wrap;gap:8px">'
        html += _svg_line_chart(loss_pts, label="Loss", color="#e07c4c")
        html += _svg_line_chart(kl_pts, label="KL Divergence", color="#c9b458")
        html += "</div>"

    evals = _all_evals()
    if evals:
        html += "<h2>Eval History</h2>"
        for step, wr, _ag in evals:
            bar_w = int(wr * 300)
            html += '<div class="wr-row">'
            html += f'<span class="wr-label">{step}</span>'
            html += f'<div class="wr-bar" style="width:{bar_w}px"></div>'
            html += f'<span class="wr-val">{wr:.0%}</span>'
            html += "</div>"

    return HTMLResponse(html if html else "")


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
