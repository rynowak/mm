"""CLIP metric math (offline, always runs) + gated real-CLIP integration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from bufo.clip_metrics import (
    clipscore,
    cosine_matrix,
    mean_pairwise_distance,
    nearest_neighbor,
    style_score,
    to_emoji,
    top_k_mean_similarity,
)
from bufo.config import EvalConfig

_EVAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval-bufo.yaml"


# --------------------------------------------------------------------------
# Tier 1 — pure math on fabricated (normalized) embeddings, no model download
# --------------------------------------------------------------------------


def test_cosine_matrix_identity_and_orthogonal():
    a = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    cos = cosine_matrix(a, a)
    assert cos[0, 0] == pytest.approx(1.0)
    assert cos[0, 1] == pytest.approx(0.0)


def test_clipscore_clamps_and_scales():
    cos = torch.tensor([-0.2, 0.0, 0.4])
    out = clipscore(cos, w=2.5)
    assert out.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_mean_pairwise_distance():
    same = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert mean_pairwise_distance(same) == pytest.approx(0.0)  # collapse
    orth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert mean_pairwise_distance(orth) == pytest.approx(1.0)
    assert mean_pairwise_distance(torch.tensor([[1.0, 0.0]])) == 0.0  # n<2


def test_nearest_neighbor_finds_planted_duplicate():
    train = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), dim=-1)
    gen = F.normalize(torch.tensor([[0.0, 1.0], [0.9, 0.1]]), dim=-1)
    vals, idx = nearest_neighbor(gen, train)
    assert vals[0] == pytest.approx(1.0)  # exact match to train row 1
    assert idx[0].item() == 1


def test_top_k_mean_similarity():
    train = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # orthonormal
    gen = torch.tensor([[1.0, 0.0]])
    assert top_k_mean_similarity(gen, train, k=1)[0].item() == pytest.approx(1.0)
    assert top_k_mean_similarity(gen, train, k=2)[0].item() == pytest.approx(0.5)  # mean of [1, 0]
    # k clamps to the number of train rows
    assert top_k_mean_similarity(gen, train, k=9)[0].item() == pytest.approx(0.5)


def test_style_score_sign():
    cartoon = F.normalize(torch.tensor([1.0, 0.0]), dim=-1)
    photo = F.normalize(torch.tensor([0.0, 1.0]), dim=-1)
    cartoonish = F.normalize(torch.tensor([[1.0, 0.1]]), dim=-1)
    assert style_score(cartoonish, cartoon, photo).item() > 0


def test_to_emoji_downscales():
    img = Image.new("RGB", (256, 256), (0, 200, 0))
    small = to_emoji(img, px=48)
    assert small.size == (48, 48)


def test_eval_config_loads_and_step_subset():
    cfg = EvalConfig.from_yaml(_EVAL_CONFIG)
    assert cfg.prompts
    assert set(cfg.step_prompts) <= set(cfg.prompts)  # cheap subset invariant
    assert "{subject}" in cfg.prompt_template


# --------------------------------------------------------------------------
# Tier 2 — gated real CLIP (downloads ViT-B/32, ~600MB)
# --------------------------------------------------------------------------

_clip = pytest.mark.skipif(
    os.environ.get("BUFO_CLIP_SMOKE") != "1",
    reason="Set BUFO_CLIP_SMOKE=1 to run the real-CLIP test (downloads ViT-B/32).",
)


@_clip
def test_embedder_shapes_and_norm():
    from mm_training import get_device

    from bufo.clip_metrics import ClipEmbedder

    emb = ClipEmbedder.load("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", get_device())
    imgs = [Image.new("RGB", (32, 32), c) for c in [(200, 0, 0), (0, 200, 0), (0, 0, 200)]]
    img_emb = emb.embed_images(imgs)
    txt_emb = emb.embed_texts(["a bufo", "a frog"])
    assert img_emb.shape[0] == 3
    assert txt_emb.shape[0] == 2
    assert torch.allclose(img_emb.norm(dim=-1), torch.ones(3), atol=1e-4)


@_clip
def test_train_embedding_cache_roundtrip(tmp_path):
    import json

    from mm_training import get_device

    from bufo.clip_metrics import ClipEmbedder, load_or_build_train_embeddings

    (tmp_path / "images").mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), (60 * i, 100, 140)).save(tmp_path / "images" / f"b{i}.png")
    (tmp_path / "metadata.jsonl").write_text(
        "\n".join(json.dumps({"file_name": f"b{i}.png", "caption": f"bufo {i}"}) for i in range(3))
    )
    emb = ClipEmbedder.load("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", get_device())
    e1, n1 = load_or_build_train_embeddings(emb, tmp_path, cache_dir=tmp_path / ".cache")
    e2, n2 = load_or_build_train_embeddings(emb, tmp_path, cache_dir=tmp_path / ".cache")
    assert n1 == n2 == ["b0.png", "b1.png", "b2.png"]
    assert torch.equal(e1, e2)
