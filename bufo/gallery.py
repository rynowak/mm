"""Visual keep/drop review gallery for the bufo corpus (single-file FastAPI).

Browse every prepared bufo with its caption + heuristic chips (scene-like,
near-dup), filter by category, and toggle keep/drop. Decisions persist to
``curation.jsonl``, which ``prepare()`` applies on the next run.

    uv run python -m bufo.gallery            # serve at http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse

from bufo.data import load_curation, save_curation

_PER_PAGE = 60


def set_keep(data_dir: str | Path, file_name: str, keep: bool) -> None:
    """Update one image's keep flag in ``curation.jsonl`` (preserving any caption)."""
    curation = load_curation(data_dir)
    rec = curation.get(file_name, {"file_name": file_name})
    rec["keep"] = keep
    curation[file_name] = rec
    save_curation(data_dir, curation)


def _load_heuristics(data_dir: Path) -> dict[str, dict]:
    report = data_dir / "curate-report.json"
    if not report.exists():
        return {}
    return {s["file_name"]: s for s in json.loads(report.read_text())["images"]}


def _records(data_dir: Path) -> list[dict]:
    meta = data_dir / "metadata.jsonl"
    return [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]


def _card(rec: dict, stats: dict, dropped: bool) -> str:
    fn = rec["file_name"]
    chips = []
    if stats.get("scene_like"):
        chips.append('<span class="chip scene">scene</span>')
    if stats.get("dup_group", -1) >= 0:
        chips.append(f'<span class="chip dup">dup {stats["dup_group"]}</span>')
    chip_html = "".join(chips)
    cls = "card dropped" if dropped else "card"
    label = "Restore" if dropped else "Drop"
    next_keep = "true" if dropped else "false"
    return f"""<div class="{cls}" id="card-{fn}">
      <img src="/img/{fn}" loading="lazy" />
      <div class="cap">{rec["caption"].split(",")[0]}</div>
      <div class="chips">{chip_html}</div>
      <button hx-post="/mark" hx-vals='{{"file_name":"{fn}","keep":"{next_keep}"}}'
              hx-target="#card-{fn}" hx-swap="outerHTML">{label}</button>
    </div>"""


_STYLE = """
<style>
  body{font-family:system-ui;margin:0;background:#fafafa}
  header{position:sticky;top:0;background:#fff;padding:10px 16px;border-bottom:1px solid #ddd}
  a{margin-right:10px;text-decoration:none;color:#0a7}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;padding:16px}
  .card{background:#fff;border:2px solid #eee;border-radius:8px;padding:6px;text-align:center}
  .card.dropped{opacity:.4;border-color:#e44}
  .card img{width:100%;border-radius:4px}
  .cap{font-size:12px;margin:4px 0;height:2.4em;overflow:hidden}
  .chip{font-size:10px;padding:1px 5px;border-radius:6px;margin:1px}
  .chip.scene{background:#fe9}.chip.dup{background:#cdf}
  button{cursor:pointer;border:0;background:#0a7;color:#fff;border-radius:6px;padding:4px 10px}
  .card.dropped button{background:#e44}
</style>
<script src="https://unpkg.com/htmx.org@1.9.10"></script>
"""


def create_app(data_dir: str | Path = "bufo/data") -> FastAPI:
    root = Path(data_dir)
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index(page: int = 0, filter: str = "all") -> str:  # noqa: A002 — query param name
        recs = _records(root)
        stats = _load_heuristics(root)
        curation = load_curation(root)
        dropped = {fn for fn, c in curation.items() if not c.get("keep", True)}

        def keep_rec(r: dict) -> bool:
            s = stats.get(r["file_name"], {})
            return {
                "all": True,
                "scene": s.get("scene_like", False),
                "dups": s.get("dup_group", -1) >= 0,
                "dropped": r["file_name"] in dropped,
                "kept": r["file_name"] not in dropped,
            }.get(filter, True)

        filtered = [r for r in recs if keep_rec(r)]
        total = len(filtered)
        page_recs = filtered[page * _PER_PAGE : (page + 1) * _PER_PAGE]
        cards = "".join(_card(r, stats.get(r["file_name"], {}), r["file_name"] in dropped) for r in page_recs)
        tabs = " ".join(f'<a href="/?filter={f}">{f}</a>' for f in ("all", "scene", "dups", "kept", "dropped"))
        nav = ""
        if page > 0:
            nav += f'<a href="/?page={page - 1}&filter={filter}">&larr; prev</a>'
        if (page + 1) * _PER_PAGE < total:
            nav += f'<a href="/?page={page + 1}&filter={filter}">next &rarr;</a>'
        return f"""<!doctype html><html><head>{_STYLE}</head><body>
          <header>{tabs} &nbsp;|&nbsp; {total} in '{filter}' · {len(dropped)} dropped &nbsp; {nav}</header>
          <div class="grid">{cards}</div></body></html>"""

    @app.get("/img/{name}")
    def img(name: str) -> FileResponse:
        return FileResponse(root / "images" / name)

    @app.post("/mark", response_class=HTMLResponse)
    def mark(file_name: str = Form(...), keep: str = Form(...)) -> str:
        keep_bool = keep == "true"
        set_keep(root, file_name, keep_bool)
        stats = _load_heuristics(root).get(file_name, {})
        rec = next((r for r in _records(root) if r["file_name"] == file_name), {"file_name": file_name, "caption": ""})
        return _card(rec, stats, dropped=not keep_bool)

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Bufo data review gallery")
    parser.add_argument("--data-dir", type=str, default="bufo/data")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Review gallery at http://127.0.0.1:{args.port}  (data: {args.data_dir})")
    uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
