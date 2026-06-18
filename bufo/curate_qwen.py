"""Qwen2.5-VL curation: keep only canonical-style bufos (muted olive-green palette +
big round expressive eyes) so the LoRA trains on a consistent identity instead of the
full all-the-bufo style mix. Writes keep/drop + sub-scores to ``curation_canonical.jsonl``.

Resumable + incremental (re-run skips already-judged files; flushes every batch).

    python -m bufo.curate_qwen --data-dir /mnt/ray/bufo-data --limit 60   # quick sample
    python -m bufo.curate_qwen --data-dir /mnt/ray/bufo-data              # full pass (GPU)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from bufo.vlm_judge import DEFAULT_MODEL_ID, VLMJudge, _vision_inputs

if TYPE_CHECKING:
    from PIL import Image

# Tightened after reviewing the first pass: (1) require a CARTOON FROG (the v1 prompt
# kept a human face, a cube, and a realistic toad photo); (2) define canonical
# POSITIVELY and allow props/emotions (the v1 prompt dropped clearly-canonical bufos
# that were crying / heart-eyed / holding things).
PROMPT = (
    "You are curating images of the 'bufo' frog emoji, keeping only ones in the ORIGINAL "
    "canonical style. The canonical bufo is a SIMPLE, FLAT CARTOON frog with: soft MUTED "
    "olive/sage green skin (not bright, neon, or saturated), LARGE round expressive eyes, "
    "and clean bold outlines — the classic 'concerned frog' sticker look. KEEP it even when "
    "it wears props or clothing, holds an object, or shows an emotion (crying, hearts, angry, "
    "etc.) — as long as the underlying character is this canonical cartoon bufo. DROP only "
    "clearly OFF-STYLE images: photographs or realistic/3D toads; human faces or non-frog "
    "subjects; plain objects, logos, or signs; bright neon or non-green coloring; tiny dot "
    "eyes or googly/closed eyes; a distinctly different art style; or images that are mostly "
    "text. Reply with STRICT JSON only: "
    '{"cartoon_frog": 0 or 1, "muted_green": 0 or 1, "expressive_eyes": 0 or 1, '
    '"canonical_bufo": 0 or 1, "reason": "<=8 words"}. Set canonical_bufo=1 if it is the '
    "canonical cartoon bufo character (props/emotions are fine); set it 0 only for the "
    "off-style cases listed above."
)

_BOOL_TRUE = {1, "1", "yes", "true", "Yes", "True"}
_FIELDS = ("cartoon_frog", "muted_green", "expressive_eyes", "canonical_bufo")


def parse_curation(text: str) -> dict:
    """Extract the strict-JSON verdict; default to drop on any parse failure."""
    fail = {**{k: 0 for k in _FIELDS}, "reason": "parse-fail"}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return dict(fail)
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return dict(fail)
    out = {k: (1 if d.get(k) in _BOOL_TRUE else 0) for k in _FIELDS}
    out["reason"] = str(d.get("reason", ""))[:60]
    return out


class QwenCurator:
    """Qwen2.5-VL wrapped to classify a bufo image as canonical-style or not."""

    def __init__(self, judge: VLMJudge):
        self.model = judge.model
        self.processor = judge.processor

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL_ID) -> QwenCurator:
        return cls(VLMJudge.load(model_id))

    def classify(self, image: Image.Image, max_new_tokens: int = 64) -> dict:
        import torch

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _vision_inputs(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]  # noqa: B905 — py3.9 no strict=
        return parse_curation(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0])


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {json.loads(line)["file_name"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}


def _save(path: Path, recs: dict[str, dict]) -> None:
    path.write_text("\n".join(json.dumps(recs[k]) for k in sorted(recs)) + ("\n" if recs else ""))


def curate(
    data_dir: str | Path = "bufo/data",
    *,
    out_name: str = "curation_canonical.jsonl",
    limit: int | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    flush_every: int = 25,
) -> dict[str, dict]:
    """Classify each image canonical/off-style into ``out_name`` (resumable + incremental)."""
    from PIL import Image

    root = Path(data_dir)
    images = root / "images"
    files = sorted(p.name for p in images.glob("*.png"))
    if limit is not None:
        files = files[:limit]
    out = _load(root / out_name)
    todo = [f for f in files if f not in out]
    print(f"Curating {len(todo)}/{len(files)} images (skip {len(out)} done) -> {out_name}", flush=True)

    rc = QwenCurator.load(model_id)
    for i, fn in enumerate(todo, 1):
        with Image.open(images / fn) as im:
            rec = rc.classify(im.convert("RGB"))
        rec["file_name"] = fn
        out[fn] = rec
        if i % flush_every == 0 or i == len(todo):
            _save(root / out_name, out)
            kept = sum(v.get("canonical_bufo", 0) for v in out.values())
            print(f"  {i}/{len(todo)} | canonical kept {kept}/{len(out)}", flush=True)
    _save(root / out_name, out)
    kept = sum(v.get("canonical_bufo", 0) for v in out.values())
    print(f"Done. canonical kept {kept}/{len(out)} -> {root / out_name}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen2.5-VL canonical-bufo curation")
    ap.add_argument("--data-dir", type=str, default="bufo/data")
    ap.add_argument("--out-name", type=str, default="curation_canonical.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    args = ap.parse_args()
    curate(args.data_dir, out_name=args.out_name, limit=args.limit, model_id=args.model)


if __name__ == "__main__":
    main()
