"""CLIP-based metrics for bufo generation — pure embedding + scoring.

Kept free of diffusers/pipeline code so the metric *math* is unit-testable with
fabricated embeddings (mirrors the wordle3 ``metrics.py`` vs ``steplog.py`` split).
All embeddings are L2-normalized, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

# Default texts for the concept / style metrics.
CONCEPT_TEXT = "a bufo, a green cartoon frog sticker"
CARTOON_TEXT = "a flat cartoon sticker"
PHOTO_TEXT = "a photograph"
CLIPSCORE_W = 2.5  # CLIPScore rescale, calibrated for ViT-B/32
EMOJI_PX = 48  # Slack-emoji render size for the legibility metric


# ---------------------------------------------------------------------------
# Pure metric math (no model needed — this is the unit-test surface)
# ---------------------------------------------------------------------------


def cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine for L2-normalized rows: (N,D)·(M,D)^T -> (N,M)."""
    return a @ b.T


def clipscore(cos: torch.Tensor, w: float = CLIPSCORE_W) -> torch.Tensor:
    """CLIPScore convention: ``w * max(cos, 0)``."""
    return w * cos.clamp(min=0.0)


def mean_pairwise_distance(emb: torch.Tensor) -> float:
    """Mean ``1 - cos`` over distinct pairs (i<j). 0.0 if fewer than 2 rows.

    Low value = the images cluster together = mode collapse.
    """
    n = emb.shape[0]
    if n < 2:
        return 0.0
    cos = emb @ emb.T
    iu = torch.triu_indices(n, n, offset=1)
    return float((1.0 - cos[iu[0], iu[1]]).mean())


def nearest_neighbor(gen: torch.Tensor, train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each generated row, the max cosine to any train row + its index.

    High max-cosine (→1.0) means the generation closely matches a *single* training
    image (memorization / copying). Returns ``(values, indices)`` shape (N_gen,).
    """
    sims = gen @ train.T
    vals, idx = sims.max(dim=1)
    return vals, idx


def top_k_mean_similarity(gen: torch.Tensor, train: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Per-gen mean cosine to its ``k`` nearest training bufos — the *identity*
    signal ("does this look like bufos in general?").

    Grounded in real bufo images, unlike a generic text concept which can't tell
    our specific character from any green cartoon frog. Top-k (not top-1) smooths
    the single-match noise that the memorization guard intentionally keeps.
    """
    sims = gen @ train.T
    k = min(k, sims.shape[1])
    topk, _ = sims.topk(k, dim=1)
    return topk.mean(dim=1)


def style_score(img_emb: torch.Tensor, cartoon_emb: torch.Tensor, photo_emb: torch.Tensor) -> torch.Tensor:
    """Per-image cartoon-vs-photo margin: ``cos(img, cartoon) - cos(img, photo)``.

    Positive = reads as flat cartoon; negative = drifting photoreal.
    """
    return img_emb @ cartoon_emb - img_emb @ photo_emb


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def to_emoji(img: Image.Image, px: int = EMOJI_PX) -> Image.Image:
    """Downscale to ``px`` square — the information an emoji actually carries.

    Embedding this (instead of the full-res image) measures whether the bufo
    survives Slack-emoji size.
    """
    return img.convert("RGB").resize((px, px), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Embedder (loads CLIP lazily; not needed by the math tests)
# ---------------------------------------------------------------------------


@dataclass
class ClipEmbedder:
    """Wraps a CLIP model + processor, returning L2-normalized embeddings (CPU)."""

    model: object
    processor: object
    device: torch.device
    model_name: str

    @classmethod
    def load(cls, model_name: str, device: torch.device) -> ClipEmbedder:
        from transformers import CLIPModel, CLIPProcessor

        # use_safetensors=True: avoids torch.load of .bin, which the cluster's
        # torch<2.6 refuses (CVE-2025-32434). Pick a CLIP repo that ships safetensors.
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        return cls(model=model, processor=processor, device=device, model_name=model_name)

    @torch.no_grad()
    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 16) -> torch.Tensor:
        # Project into the joint CLIP space explicitly: get_image_features in
        # transformers 5.x returns the pre-projection vision output, not the
        # joint embedding, so we run vision_model -> visual_projection ourselves.
        if not images:
            return torch.empty(0)
        out: list[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
            feats = self.model.visual_projection(vision.pooler_output)
            out.append(F.normalize(feats, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        text = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        feats = self.model.text_projection(text.pooler_output)
        return F.normalize(feats, dim=-1).cpu()


# ---------------------------------------------------------------------------
# Train-image embedding cache (for the memorization metric)
# ---------------------------------------------------------------------------


def _cache_key(model_name: str, stats: Sequence[tuple[str, int, int]]) -> str:
    h = hashlib.sha1(model_name.encode())  # noqa: S324 — cache key, not security
    for name, mtime, size in stats:
        h.update(f"{name}:{mtime}:{size}".encode())
    return h.hexdigest()[:16]


def load_or_build_train_embeddings(
    embedder: ClipEmbedder,
    data_dir: str | Path,
    cache_dir: str | Path = "runs/.cache/clip_train_emb",
) -> tuple[torch.Tensor, list[str]]:
    """Return (N,D) normalized train-image embeddings + aligned filenames.

    Cached under ``cache_dir`` keyed by (clip model, file name/mtime/size); a key
    mismatch (dataset or model changed) triggers a rebuild.
    """
    root = Path(data_dir)
    records = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line.strip()]
    images_dir = root / "images"
    filenames = [r["file_name"] for r in records]
    stats = []
    for fn in filenames:
        st = (images_dir / fn).stat()
        stats.append((fn, int(st.st_mtime), st.st_size))
    key = _cache_key(embedder.model_name, stats)
    cache_path = Path(cache_dir) / f"{key}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path)
        return blob["embeddings"], blob["filenames"]

    images = [Image.open(images_dir / fn) for fn in filenames]
    embeddings = embedder.embed_images(images)
    for im in images:
        im.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"embeddings": embeddings, "filenames": filenames, "model_name": embedder.model_name, "key": key}
    torch.save(blob, cache_path)
    return embeddings, filenames
