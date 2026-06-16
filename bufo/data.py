"""Bufo data pipeline: download → caption → preprocess → dataset.

Source corpus: https://github.com/knobiknows/all-the-bufo (public, ~1.4k PNGs).

``prepare()`` downloads the raw PNGs, composites their transparency onto a white
background, pads to square and resizes, derives a caption from each filename, and
writes ``metadata.jsonl`` (HF imagefolder convention). ``BufoDataset`` then serves
``(pixel_values in [-1, 1], tokenized caption)`` pairs for LoRA training.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import CLIPTokenizer

    from bufo.config import DataConfig

_USER_AGENT = "mm-bufo-sample/0.1 (+https://github.com/rynowak/mm)"
_TREE_URL = "https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


# ----------------------------------------------------------------------------
# Captions
# ----------------------------------------------------------------------------


# The caption schema is the shared contract between training captions and the
# inference (emoji) interface: ``{TRIGGER} {action}{SUFFIX}``. ``bufo`` is the
# constant trigger; the action rides as a phrase (not "a bufo of …", which read
# wrong for the verb-style names). The suffix encodes the flat-cartoon / legible
# sticker style we filter and prompt toward (see docs/bufo-improvements.md).
TRIGGER = "bufo"
SUFFIX = ", flat cartoon frog emoji sticker, bold simple shapes, white background"


def filename_phrase(name: str) -> str:
    """Extract the action phrase from a bufo filename or shortcode.

    Strips the ``bufo`` trigger from each token and keeps the remainder, so the
    action is what's left.

    >>> filename_phrase("bufo-offers-cash-money.png")
    'offers cash money'
    >>> filename_phrase("awesomebufo")
    'awesome'
    >>> filename_phrase("bufo")
    ''
    """
    stem = Path(name).stem.lower()
    words = (re.sub("bufo", "", w) for w in re.split(r"[-_\s]+", stem))
    return " ".join(w for w in words if w).strip()


def caption_for(filename: str) -> str:
    """Training caption in the schema ``{TRIGGER} {action}{SUFFIX}``.

    >>> caption_for("cowboy-bufo.png")
    'bufo cowboy, flat cartoon frog emoji sticker, bold simple shapes, white background'
    >>> caption_for("bufo.png")
    'bufo, flat cartoon frog emoji sticker, bold simple shapes, white background'
    """
    phrase = filename_phrase(filename)
    head = f"{TRIGGER} {phrase}" if phrase else TRIGGER
    return f"{head}{SUFFIX}"


def shortcode_to_prompt(code: str) -> str:
    """Map a Slack emoji shortcode to a generation prompt in the training schema.

    >>> shortcode_to_prompt(":bufo-offers-cash-money:")
    'bufo offers cash money, flat cartoon frog emoji sticker, bold simple shapes, white background'
    """
    return caption_for(code.strip(":"))


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------


def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted https hosts)
        return resp.read()


def list_bufo_files(cfg: DataConfig) -> list[str]:
    """List repo-relative PNG paths under ``cfg.subdir``, minus excluded ones."""
    tree = json.loads(_get(_TREE_URL.format(repo=cfg.repo, ref=cfg.ref)))
    if "tree" not in tree:
        raise RuntimeError(f"Unexpected GitHub tree response: {list(tree)[:5]}")
    prefix = f"{cfg.subdir}/"
    paths = [t["path"] for t in tree["tree"] if t["path"].startswith(prefix) and t["path"].lower().endswith(".png")]
    paths = [p for p in paths if not any(sub in Path(p).name for sub in cfg.exclude_substrings)]
    return sorted(paths)


def _download_one(repo: str, ref: str, path: str, raw_dir: Path) -> Path | None:
    dest = raw_dir / Path(path).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # Percent-encode so non-ASCII filenames (e.g. "straße-bufo.png") form valid URLs.
    url = _RAW_URL.format(repo=repo, ref=ref, path=urllib.parse.quote(path))
    for attempt in range(3):
        try:
            dest.write_bytes(_get(url))
            return dest
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError):
            if attempt == 2:
                return None
    return None


def download_images(cfg: DataConfig, paths: Sequence[str], raw_dir: Path, max_workers: int = 16) -> list[Path]:
    """Download ``paths`` into ``raw_dir`` (parallel, idempotent). Returns saved files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, cfg.repo, cfg.ref, p, raw_dir): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad file must not abort the batch
                print(f"  skip {futures[fut]}: {exc}")
                result = None
            if result is not None:
                saved.append(result)
            if i % 100 == 0 or i == len(paths):
                print(f"  downloaded {i}/{len(paths)}")
    return saved


# ----------------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------------


def to_square_rgb(img: Image.Image, resolution: int, *, crop: bool = False) -> Image.Image:
    """Composite transparency onto white, square it, resize to ``resolution``.

    ``crop`` center-crops to square (tighter framing, may trim wide images);
    otherwise pad to square (default — never loses subject, adds white border).
    """
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(white, rgba).convert("RGB")
    w, h = flat.size
    if crop:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        flat = flat.crop((left, top, left + side, top + side))
    else:
        side = max(w, h)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(flat, ((side - w) // 2, (side - h) // 2))
        flat = square
    return flat.resize((resolution, resolution), Image.Resampling.LANCZOS)


def preprocess(
    raw_files: Sequence[Path], images_dir: Path, resolution: int, *, crop: bool = False
) -> list[dict[str, str]]:
    """Resize each raw PNG into ``images_dir`` and return metadata records."""
    images_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for src in sorted(raw_files):
        try:
            with Image.open(src) as im:
                out = to_square_rgb(im, resolution, crop=crop)
        except OSError:
            print(f"  skip unreadable {src.name}")
            continue
        out_name = f"{src.stem}.png"
        out.save(images_dir / out_name)
        records.append({"file_name": out_name, "caption": caption_for(src.name)})
    return records


def load_curation(data_dir: str | Path) -> dict[str, dict]:
    """Load per-image keep/drop + caption overrides from ``curation.jsonl``.

    Returns ``{file_name: {keep: bool, caption?: str}}`` (empty if no file).
    """
    path = Path(data_dir) / "curation.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["file_name"]] = rec
    return out


def save_curation(data_dir: str | Path, curation: dict[str, dict]) -> None:
    """Write ``curation.jsonl`` (one record per file_name, sorted)."""
    path = Path(data_dir) / "curation.jsonl"
    with open(path, "w") as f:
        for _, rec in sorted(curation.items()):
            f.write(json.dumps(rec) + "\n")


def apply_curation(records: list[dict[str, str]], curation: dict[str, dict]) -> list[dict[str, str]]:
    """Drop images marked ``keep: false`` and apply any caption overrides."""
    kept: list[dict[str, str]] = []
    for rec in records:
        c = curation.get(rec["file_name"])
        if c is not None and not c.get("keep", True):
            continue
        if c is not None and c.get("caption"):
            rec = {**rec, "caption": c["caption"]}
        kept.append(rec)
    return kept


def prepare(cfg: DataConfig, *, limit: int | None = None) -> Path:
    """Download + preprocess the bufo corpus. Returns the metadata.jsonl path."""
    root = Path(cfg.data_dir)
    raw_dir, images_dir = root / "raw", root / "images"
    print(f"Listing bufo PNGs from {cfg.repo}@{cfg.ref}...")
    paths = list_bufo_files(cfg)
    if limit is not None:
        paths = paths[:limit]
    print(f"  {len(paths)} files (after exclusions: {cfg.exclude_substrings})")
    raw_files = download_images(cfg, paths, raw_dir)
    print(f"Preprocessing {len(raw_files)} images to {cfg.resolution}px (crop={cfg.crop})...")
    records = preprocess(raw_files, images_dir, cfg.resolution, crop=cfg.crop)
    curation = load_curation(root)
    if curation:
        before = len(records)
        records = apply_curation(records, curation)
        print(f"Applied curation.jsonl: {before} -> {len(records)} kept")
    meta_path = root / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} records to {meta_path}")
    return meta_path


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------


class BufoDataset(Dataset):
    """Serves (pixel_values, input_ids) for SD/SDXL/FLUX LoRA training.

    ``pixel_values`` are CHW float tensors normalized to ``[-1, 1]`` (the range the
    VAE encoder expects); ``input_ids`` are the CLIP-tokenized caption.

    - sd15: only ``input_ids`` (CLIP, 77).
    - sdxl: ``input_ids`` + ``input_ids_2`` (both CLIP tokenizers, 77 each).
    - flux: ``input_ids`` (CLIP, 77, pooled by the CLIP encoder) + ``input_ids_2``
      (T5, ``tokenizer_2_max_length`` tokens — 256 or 512). Pass ``base_kind="flux"``
      so the T5 stream is truncated/padded to ``tokenizer_2_max_length`` rather than
      the tokenizer's (very large) default ``model_max_length``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: CLIPTokenizer,
        *,
        tokenizer_2: CLIPTokenizer | None = None,
        resolution: int = 512,
        random_flip: bool = True,
        base_kind: str = "sd15",
        tokenizer_2_max_length: int = 512,
    ) -> None:
        root = Path(data_dir)
        meta_path = root / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found — run `python -m bufo.prepare` first.")
        self.images_dir = root / "images"
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2  # SDXL: second CLIP; flux: T5; None for sd15
        self.resolution = resolution
        self.random_flip = random_flip
        self.base_kind = base_kind
        self.tokenizer_2_max_length = tokenizer_2_max_length
        with open(meta_path) as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"No records in {meta_path}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _encode(tokenizer: CLIPTokenizer, caption: str, max_length: int | None = None) -> torch.Tensor:
        return tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=max_length if max_length is not None else tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        with Image.open(self.images_dir / rec["file_name"]) as im:
            img = im.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        if self.random_flip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0  # CHW in [-1, 1]
        item = {"pixel_values": pixel_values, "input_ids": self._encode(self.tokenizer, rec["caption"])}
        if self.tokenizer_2 is not None:
            # flux: T5 gets its own (larger) token budget; sdxl: second CLIP at 77.
            max_2 = self.tokenizer_2_max_length if self.base_kind == "flux" else None
            item["input_ids_2"] = self._encode(self.tokenizer_2, rec["caption"], max_length=max_2)
        return item
