"""Caption the curated cells (from their grid/position) into a training metadata.jsonl.

Each surviving cell grid_NN_MM maps to the matrix item the grid requested at position MM
(the grid is rendered + sliced in reading order). Caption = the winning full-FT schema with
that pose/expression. Writes ~/Bufo/dataset/metadata.jsonl (file_name = cell basename).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

_spec = importlib.util.spec_from_file_location("gb", os.path.join(os.path.dirname(__file__), "grid_batch.py"))
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)

CELLS = os.path.expanduser("~/Bufo/dataset/cells")
META = os.path.expanduser("~/Bufo/dataset/metadata.jsonl")
MASTER, CELLN = gb.MASTER, gb.CELLS
PREFIX, SUFFIX = "olive green adult bufo", "soft-shaded cartoon sticker"


def item_for(fn: str) -> str:
    m = re.match(r"grid_(\d+)_(\d+)\.png", fn)
    if not m:
        return ""
    gi, mm = int(m.group(1)), int(m.group(2))
    return MASTER[(gi * CELLN + mm) % len(MASTER)]


def main() -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(CELLS, "grid_*.png"))):
        fn = os.path.basename(p)
        item = item_for(fn)
        cap = f"{PREFIX}, {item}, {SUFFIX}" if item else f"{PREFIX}, {SUFFIX}"
        rows.append({"file_name": fn, "caption": cap})
    with open(META, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {META}")
    for r in rows[:8]:
        print(f"  {r['file_name']}: {r['caption']}")


if __name__ == "__main__":
    main()
