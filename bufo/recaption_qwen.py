"""Recaption the bufo corpus with Qwen2.5-VL + few-shot prompting.

BLIP-large produced noisy grounding ("araffe", "close up", wrong nouns like
"dinosaur"/"person"), so we use Qwen2.5-VL — a much stronger captioner — with an
explicit instruction and a handful of image+answer few-shots that pin the desired
output: a short phrase of *visible props/pose/expression only*. The result is merged
with the filename action into the same caption schema and stored as overrides in a
SEPARATE ``curation.qwen.jsonl`` (so the active ``curation.jsonl`` is untouched until
we A/B-gate and promote it).

    python -m bufo.recaption_qwen --data-dir /mnt/ray/bufo-data        # all (needs GPU)
    python -m bufo.recaption_qwen --data-dir /mnt/ray/bufo-data --limit 24

Resumable + incremental: the output file is rewritten every flush, and a re-run
skips file_names already captioned, so a VPN drop / preemption never loses progress.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from bufo.data import SUFFIX, TRIGGER, filename_phrase
from bufo.vlm_judge import DEFAULT_MODEL_ID, VLMJudge, _vision_inputs

if TYPE_CHECKING:
    from PIL import Image

# Instruction repeated on every turn so the few-shots and the target share framing.
# Descriptive (not terse): diffusion fine-tuning conditions better on a rich sentence
# covering pose/action, props+colors, and expression — kept strictly grounded so we
# don't reintroduce the BLIP hallucinations.
INSTRUCTION = (
    "You are writing a detailed caption for a cartoon frog ('bufo') sticker, to train "
    "an image generator. In ONE rich sentence, describe what the frog is doing and how "
    "it looks: its pose and action, any props or clothing and their colors, its facial "
    "expression, and other clearly visible details. Be specific and concrete, but "
    "describe ONLY what is actually visible — never guess or invent. Do not mention that "
    "it is a close-up and do not describe the plain background. Start with the action, "
    "lowercase, no trailing period."
)

# Few-shot anchors: (filename in <data>/images, ideal rich description). These set the
# DETAIL BAR — the model mirrors their length and specificity (props, colors, expression).
FEWSHOT: list[tuple[str, str]] = [
    (
        "bufo-goes-to-space.png",
        "floating in a puffy white astronaut suit and round glass helmet, arms held out, eyes wide and calm",
    ),
    (
        "bufo-offers-a-gavel.png",
        "holding up a brown wooden gavel in one webbed hand, mouth open mid-announcement with eyebrows raised",
    ),
    (
        "bufo-bandana.png",
        "wearing a red bandana tied across its forehead, eyes narrowed into a tough, determined squint",
    ),
    (
        "bufo-roasted.png",
        "engulfed in bright orange flames, eyes wide and mouth open in a panicked yelp",
    ),
]

# Wrapper phrases to strip from the model's reply while PRESERVING the description
# (unlike recaption.clean_detail, which flattens to a word-bag).
_STRIP_PREFIXES = (
    "the image shows",
    "this image shows",
    "the picture shows",
    "this is",
    "the bufo is",
    "the frog is",
    "a cartoon frog",
    "it is",
    "close-up of",
    "close up of",
    "close-up",
    "close up",
)


def clean_descriptive(text: str) -> str:
    """Light cleanup that keeps the descriptive sentence: lowercase, drop meta/close-up
    wrappers, collapse whitespace, trim stray punctuation."""
    s = text.strip().lower()
    for p in _STRIP_PREFIXES:
        s = s.replace(p, " ")
    return re.sub(r"\s+", " ", s).strip(" .,")


class QwenRecaptioner:
    """Qwen2.5-VL captioner producing rich, grounded visual descriptions via few-shot."""

    def __init__(self, judge: VLMJudge, fewshot: list[tuple[Image.Image, str]]):
        self.model = judge.model
        self.processor = judge.processor
        self.fewshot = fewshot

    @classmethod
    def load(cls, data_dir: Path, model_id: str = DEFAULT_MODEL_ID) -> QwenRecaptioner:
        from PIL import Image

        judge = VLMJudge.load(model_id)
        images_dir = Path(data_dir) / "images"
        shots: list[tuple[Image.Image, str]] = []
        for fn, answer in FEWSHOT:
            p = images_dir / fn
            if p.exists():
                shots.append((Image.open(p).convert("RGB"), answer))
        return cls(judge, shots)

    def _messages(self, target: Image.Image) -> list[dict]:
        msgs: list[dict] = []
        for img, answer in self.fewshot:
            msgs.append(
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": INSTRUCTION}]}
            )
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        msgs.append(
            {"role": "user", "content": [{"type": "image", "image": target}, {"type": "text", "text": INSTRUCTION}]}
        )
        return msgs

    def describe(self, image: Image.Image, max_new_tokens: int = 96) -> str:
        import torch

        messages = self._messages(image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _vision_inputs(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]  # noqa: B905 — py3.9 has no strict=
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def caption_for_image(self, file_name: str, image: Image.Image) -> str:
        # Keep the meme action (the shortcode handle) AND the rich Qwen description, so
        # the model learns both what to call it and what it looks like.
        desc = clean_descriptive(self.describe(image))
        action = filename_phrase(file_name)
        parts = [p for p in (action, desc) if p]
        return f"{TRIGGER} " + ", ".join(parts) + SUFFIX


def recaption_dataset_qwen(
    data_dir: str | Path = "bufo/data",
    *,
    out_name: str = "curation.qwen.jsonl",
    limit: int | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    flush_every: int = 25,
) -> int:
    """Qwen-recaption each image into ``out_name`` (resumable + incremental)."""
    from PIL import Image

    root = Path(data_dir)
    images_dir = root / "images"
    records = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]

    # Resume: keep+caption state already written to out_name is reused as-is.
    out = load_curation_named(root, out_name)
    done = {fn for fn, e in out.items() if e.get("caption")}
    todo = [r for r in records if r["file_name"] not in done]
    print(f"Qwen-recaptioning {len(todo)}/{len(records)} images (skipping {len(done)} done) -> {out_name}", flush=True)

    rc = QwenRecaptioner.load(root, model_id)
    for i, rec in enumerate(todo, 1):
        fn = rec["file_name"]
        with Image.open(images_dir / fn) as im:
            caption = rc.caption_for_image(fn, im.convert("RGB"))
        entry = out.get(fn, {"file_name": fn})
        entry["caption"] = caption
        entry.setdefault("keep", True)
        out[fn] = entry
        if i % flush_every == 0 or i == len(todo):
            save_curation_named(root, out, out_name)
            print(f"  recaptioned {i}/{len(todo)} (flushed)", flush=True)
    save_curation_named(root, out, out_name)
    print(f"Wrote {len(out)} Qwen recaptions to {root / out_name}", flush=True)
    return len(todo)


def load_curation_named(data_dir: str | Path, name: str) -> dict[str, dict]:
    """``load_curation`` for an arbitrary file name (default reads ``curation.jsonl``)."""
    path = Path(data_dir) / name
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["file_name"]] = rec
    return out


def save_curation_named(data_dir: str | Path, curation: dict[str, dict], name: str) -> None:
    path = Path(data_dir) / name
    lines = [json.dumps(curation[k]) for k in sorted(curation)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recaption the bufo corpus with Qwen2.5-VL + few-shot")
    parser.add_argument("--data-dir", type=str, default="bufo/data")
    parser.add_argument("--out-name", type=str, default="curation.qwen.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    recaption_dataset_qwen(args.data_dir, out_name=args.out_name, limit=args.limit, model_id=args.model)


if __name__ == "__main__":
    main()
