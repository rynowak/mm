"""Generate SINGLE bufos (one per image, large subject) from the input-grid reference.

Single-output holds identity (the input grid is the reference) and renders a ~450-550px
subject vs ~300px grid cells -> much sharper training data, and no slicing + exact captions
(one known pose per image). Concurrent for speed. Diversity from the same matrix as the grids.

Env: OUTDIR, N_VAR (variations per item), LIMIT (first N items, for tests), WORKERS.
Run locally: needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import traceback

_base = os.path.dirname(__file__)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_base, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


oe = _load("openrouter_edit")
gb = _load("grid_batch")

INPUT_GRID = os.path.expanduser("~/Bufo/grid/input_grid.png")
OUTDIR = os.path.expanduser(os.environ.get("OUTDIR", "~/Bufo/dataset3/singles"))
MODEL = "google/gemini-2.5-flash-image"
N_VAR = int(os.environ.get("N_VAR", "3"))
LIMIT = int(os.environ.get("LIMIT", "0"))
WORKERS = int(os.environ.get("WORKERS", "6"))
MASTER = gb.MASTER

PROMPT = (
    "This image shows the bufo frog character in several poses. Draw this exact same character {item}, "
    "as ONE large figure that fills the frame, centered, plain white background. Keep its identity, colors, "
    "its large round flat-set stylized cartoon eyes, and its flat soft-shaded cartoon style identical to "
    "the reference."
)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    items = MASTER[:LIMIT] if LIMIT else MASTER
    tasks = [(i, item, v) for i, item in enumerate(items) for v in range(N_VAR)]
    todo = [(i, item, v) for (i, item, v) in tasks if not os.path.exists(os.path.join(OUTDIR, f"{i:03d}_{v}.png"))]
    print(f"items={len(items)} N_VAR={N_VAR} -> {len(tasks)} singles; todo={len(todo)}", flush=True)

    def gen(t: tuple[int, str, int]) -> str:
        i, item, v = t
        out = os.path.join(OUTDIR, f"{i:03d}_{v}.png")
        try:
            oe.edit_image(INPUT_GRID, PROMPT.format(item=item), MODEL, out)
            return f"ok {i:03d}_{v} {item}"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return f"FAIL {i:03d}_{v}: {e}"

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(gen, todo):
            done += 1
            if done % 10 == 0 or "FAIL" in r:
                print(f"[{done}/{len(todo)}] {r}", flush=True)
    print("DONE singles in", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
