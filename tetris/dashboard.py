"""Dashboard for Tetris RL training — live game view, metrics, and replay."""

import asyncio
import contextlib
import queue
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Shared state between training thread and async WebSocket handlers
# ---------------------------------------------------------------------------
event_queues: list[queue.Queue[dict[str, Any]]] = []
completed_games: deque[dict[str, Any]] = deque(maxlen=200)
metrics_history: list[dict[str, Any]] = []
training_status: dict[str, Any] = {"state": "starting"}
_current_game_steps: list[dict[str, Any]] = []


def training_callback(event: dict[str, Any]) -> None:
    """Called from the training thread for each event."""
    if event["type"] == "game_start":
        _current_game_steps.clear()
        _current_game_steps.append(event)
    elif event["type"] == "step":
        _current_game_steps.append(event)
    elif event["type"] == "episode_end":
        completed_games.append(
            {
                "game_id": event["game_id"],
                "lines": event["lines"],
                "score": event["score"],
                "pieces": event["pieces"],
                "episode": event["episode"],
                "steps": list(_current_game_steps),
            }
        )
        metrics_history.append(
            {
                "episode": event["episode"],
                "lines": event["lines"],
                "score": event["score"],
                "epsilon": event["epsilon"],
                "loss": event["loss"],
                "elapsed": event["elapsed"],
            }
        )
        _current_game_steps.clear()
    elif event["type"] == "training_complete":
        training_status["state"] = "complete"
        training_status.update(event)

    for q in event_queues:
        with contextlib.suppress(queue.Full):
            q.put_nowait(event)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def _start_training() -> None:
    from train import train

    training_status["state"] = "training"
    train(on_event=training_callback)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # type: ignore[no-any-explicit]
    thread = threading.Thread(target=_start_training, daemon=True)
    thread.start()
    yield


app = FastAPI(title="Tetris RL Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/api/games")
async def get_games() -> list[dict[str, Any]]:
    return [
        {"game_id": g["game_id"], "lines": g["lines"], "score": g["score"], "pieces": g["pieces"]}
        for g in completed_games
    ]


@app.get("/api/games/{game_id}")
async def get_game(game_id: int) -> dict[str, Any]:
    for g in completed_games:
        if g["game_id"] == game_id:
            return g
    return {"error": "not found"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=5000)
    event_queues.append(q)

    try:
        recent = metrics_history[-500:] if metrics_history else []
        await websocket.send_json({"type": "init", "metrics": recent, "status": training_status["state"]})

        loop = asyncio.get_running_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=1))
                await websocket.send_json(event)
            except queue.Empty:
                continue
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if q in event_queues:
            event_queues.remove(q)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
