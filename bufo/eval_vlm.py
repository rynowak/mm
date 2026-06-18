"""VLM-judge eval for bufo IDENTITY (+ duplication/artifacts) — the metric CLIP can't give.

CLIP scores "frogginess" (semantic presence of a green frog), so it rates a generic
cute frog as on-target. To know whether a generation is *the bufo character* — muted
sage/olive skin, wide head, big earnest wide-set eyes — and whether it's a clean single
subject (not the "dozens of bufos" tiling), we ask a VLM (Qwen2.5-VL) directly.

Reuses ``VLMJudge`` for model loading; ships its own identity-focused prompt.

    python -m bufo.eval_vlm --images-dir /mnt/ray/bufo-runs/sdxl-1024/eval-1500/images
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bufo.vlm_judge import DEFAULT_MODEL_ID, VLMJudge, _vision_inputs

BUFO_SYSTEM = (
    "You are a STRICT judge of whether an image is the specific 'bufo' Slack-emoji "
    "character. The canonical bufo is a FLAT cartoon frog with soft MUTED sage/olive-"
    "green skin (NOT bright, neon, or saturated green), a wide rounded head, a simple "
    "rounded body, clean bold outlines, and most distinctively LARGE round EARNEST, "
    "slightly-concerned, forward-facing wide-set eyes. A generic cute cartoon frog "
    "(bright/saturated green, googly or sparkly eyes, glossy/3D shading, naturalistic "
    "frog proportions) is NOT bufo. Reply with ONLY one JSON object, no prose, no markdown."
)
BUFO_USER = (
    "Return STRICT JSON with exactly these keys:\n"
    '  "is_bufo": 0 or 1 — 1 ONLY if it is clearly THE bufo character (muted sage/olive, '
    "wide head, big earnest wide-set eyes). A generic/bright/googly cartoon frog => 0.\n"
    '  "identity": integer 1-5 — how strongly it matches canonical bufo '
    "(5 = unmistakably bufo, 1 = generic frog).\n"
    '  "single_subject": 0 or 1 — 1 if exactly ONE frog fills the frame; '
    "0 if there are multiple, duplicated, or tiled frogs.\n"
    '  "artifacts": 0 or 1 — 1 if melted/glitchy/noisy/distorted or broken anatomy.\n'
    '  "reason": one line <= 15 words.\n'
    "Output JSON only."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class BufoVerdict:
    is_bufo: int
    identity: int
    single_subject: int
    artifacts: int
    reason: str
    file_name: str = ""
    parse_failed: bool = False

    @property
    def clean_bufo(self) -> bool:
        """The headline: a real bufo, exactly one of it, no artifacts."""
        return self.is_bufo == 1 and self.single_subject == 1 and self.artifacts == 0


def _coerce_int(value: Any, lo: int, hi: int, default: int) -> int:
    if isinstance(value, bool):
        value = int(value)
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def parse_bufo_json(text: str) -> BufoVerdict:
    """Extract a ``BufoVerdict``; worst-case (not bufo, artifacts) on parse failure."""
    worst = {"is_bufo": 0, "identity": 1, "single_subject": 0, "artifacts": 1}
    m = _JSON_RE.search(text)
    if not m:
        return BufoVerdict(**worst, reason="parse-fail: no json", parse_failed=True)
    try:
        d = json.loads(m.group(0))
    except (ValueError, TypeError):
        return BufoVerdict(**worst, reason="parse-fail: bad json", parse_failed=True)
    return BufoVerdict(
        is_bufo=1 if _coerce_int(d.get("is_bufo"), 0, 1, 0) else 0,
        identity=_coerce_int(d.get("identity"), 1, 5, 1),
        single_subject=1 if _coerce_int(d.get("single_subject"), 0, 1, 0) else 0,
        artifacts=1 if _coerce_int(d.get("artifacts"), 0, 1, 1) else 0,
        reason=str(d.get("reason", ""))[:80],
    )


@dataclass
class BufoAggregate:
    n: int
    bufo_rate: float  # fraction judged THE bufo character
    mean_identity: float  # mean 1-5
    single_rate: float  # fraction with exactly one subject (1 - duplication)
    artifact_rate: float
    clean_bufo_rate: float  # is_bufo AND single AND no artifacts — the headline


def aggregate(verdicts: list[BufoVerdict]) -> BufoAggregate:
    n = len(verdicts)
    if n == 0:
        return BufoAggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return BufoAggregate(
        n=n,
        bufo_rate=sum(v.is_bufo for v in verdicts) / n,
        mean_identity=sum(v.identity for v in verdicts) / n,
        single_rate=sum(v.single_subject for v in verdicts) / n,
        artifact_rate=sum(v.artifacts for v in verdicts) / n,
        clean_bufo_rate=sum(1 for v in verdicts if v.clean_bufo) / n,
    )


def _generate(judge: VLMJudge, image: Any) -> str:
    import torch

    messages = [
        {"role": "system", "content": BUFO_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": BUFO_USER},
            ],
        },
    ]
    text = judge.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = _vision_inputs(messages)
    inputs = judge.processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(judge.device)
    with torch.no_grad():
        generated = judge.model.generate(**inputs, max_new_tokens=judge.max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]  # noqa: B905 — py3.9 no strict=
    return judge.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def score_dir(
    images_dir: str | Path, *, model_id: str = DEFAULT_MODEL_ID, limit: int | None = None
) -> tuple[list[BufoVerdict], BufoAggregate]:
    """Judge every PNG in ``images_dir`` for bufo identity + cleanliness."""
    from PIL import Image

    files = sorted(Path(images_dir).glob("*.png"))
    if limit is not None:
        files = files[:limit]
    judge = VLMJudge.load(model_id)
    verdicts: list[BufoVerdict] = []
    for i, f in enumerate(files, 1):
        with Image.open(f) as im:
            v = parse_bufo_json(_generate(judge, im))
        v.file_name = f.name
        verdicts.append(v)
        if i % 10 == 0 or i == len(files):
            print(f"  judged {i}/{len(files)}", flush=True)
    return verdicts, aggregate(verdicts)


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM-judge bufo identity + duplication/artifacts")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="Write per-image verdicts + aggregate JSON here")
    args = ap.parse_args()

    verdicts, agg = score_dir(args.images_dir, model_id=args.model, limit=args.limit)
    print(
        f"BUFO-VLM n={agg.n} | bufo_rate {agg.bufo_rate:.2f} | mean_identity {agg.mean_identity:.2f} | "
        f"single_rate {agg.single_rate:.2f} | artifact_rate {agg.artifact_rate:.2f} | "
        f"clean_bufo_rate {agg.clean_bufo_rate:.2f}",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps({"aggregate": asdict(agg), "verdicts": [asdict(v) for v in verdicts]}, indent=1)
        )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
