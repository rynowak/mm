"""Recaption the bufo corpus with a vision model, for visual grounding.

The filename encodes the *action* (humans named each bufo by what it does); a VLM
adds *visual* detail (color, expression, props). We merge both into the caption
schema and write the result as caption overrides in ``curation.jsonl`` (which
``prepare()`` applies). Eval-gated: A/B vs filename captions before adopting.

    uv run python -m bufo.recaption              # recaption all (downloads BLIP)
    uv run python -m bufo.recaption --limit 32   # quick subset
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from mm_training import get_device
from PIL import Image

from bufo.data import SUFFIX, TRIGGER, filename_phrase, load_curation, save_curation

_DEFAULT_VLM = "Salesforce/blip-image-captioning-large"
# Words/phrases to drop from VLM detail — redundant with the bufo/cartoon schema.
_DROP_PHRASES = ["a cartoon of", "a drawing of", "an image of", "a picture of", "an illustration of", "cartoon of"]
_DROP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "and",
    "is",
    "it",
    "cartoon",
    "drawing",
    "image",
    "picture",
    "illustration",
    "frog",
    "toad",
    "green",
}


def clean_detail(caption: str) -> str:
    """Strip a VLM caption down to grounding words (color/props/expression).

    >>> clean_detail("a cartoon of a green frog wearing a red hat")
    'wearing red hat'
    """
    s = caption.lower().strip()
    for phrase in _DROP_PHRASES:
        s = s.replace(phrase, " ")
    words = [w for w in re.split(r"[^a-z0-9]+", s) if w and w not in _DROP_WORDS]
    return " ".join(words).strip()


def merge_caption(action: str, detail: str) -> str:
    """Combine filename action + cleaned VLM detail into the caption schema.

    Detail words already present in the action are dropped, e.g.
    ``merge_caption("offers cash money", "holding money")`` keeps only "holding"
    from the detail -> ``"bufo offers cash money, holding" + SUFFIX``.
    """
    action_words = set(action.split())
    detail = " ".join(w for w in detail.split() if w not in action_words)  # dedupe vs action
    parts = [p for p in (action, detail) if p]
    head = f"{TRIGGER} " + ", ".join(parts) if parts else TRIGGER
    return f"{head}{SUFFIX}"


@dataclass
class Recaptioner:
    model: object
    processor: object
    device: torch.device

    @classmethod
    def load(cls, model_name: str = _DEFAULT_VLM, device: torch.device | None = None) -> Recaptioner:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = device or get_device()
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        processor = BlipProcessor.from_pretrained(model_name)
        return cls(model=model, processor=processor, device=device)

    @torch.no_grad()
    def describe(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def caption_for_image(self, file_name: str, image: Image.Image) -> str:
        return merge_caption(filename_phrase(file_name), clean_detail(self.describe(image)))


def recaption_dataset(
    data_dir: str | Path = "bufo/data", *, limit: int | None = None, model_name: str = _DEFAULT_VLM
) -> int:
    """Recaption each image and store as caption overrides in curation.jsonl."""
    root = Path(data_dir)
    images_dir = root / "images"
    records = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    print(f"Recaptioning {len(records)} images with {model_name}...")
    rc = Recaptioner.load(model_name)
    curation = load_curation(root)
    for i, rec in enumerate(records, 1):
        fn = rec["file_name"]
        with Image.open(images_dir / fn) as im:
            caption = rc.caption_for_image(fn, im)
        entry = curation.get(fn, {"file_name": fn})
        entry["caption"] = caption
        entry.setdefault("keep", True)
        curation[fn] = entry
        if i % 100 == 0 or i == len(records):
            print(f"  recaptioned {i}/{len(records)}")
    save_curation(root, curation)
    print(f"Wrote {len(records)} recaptions to {root / 'curation.jsonl'}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recaption the bufo corpus with a VLM")
    parser.add_argument("--data-dir", type=str, default="bufo/data")
    parser.add_argument("--model", type=str, default=_DEFAULT_VLM)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    recaption_dataset(args.data_dir, limit=args.limit, model_name=args.model)


if __name__ == "__main__":
    main()
