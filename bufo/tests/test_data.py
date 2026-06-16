"""Fast, offline tests for the bufo data pipeline (no network, no model download)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from bufo.data import (
    SUFFIX,
    BufoDataset,
    apply_curation,
    caption_for,
    filename_phrase,
    load_curation,
    save_curation,
    shortcode_to_prompt,
    to_square_rgb,
)


class _StubTokenizer:
    """Minimal CLIPTokenizer stand-in: returns zeroed input_ids of max_length."""

    model_max_length = 16

    def __call__(self, text: str, *, padding=None, truncation=None, max_length: int = 16, return_tensors=None):
        return SimpleNamespace(input_ids=torch.zeros((1, max_length), dtype=torch.long))


class _StubT5Tokenizer:
    """T5TokenizerFast stand-in with a huge default model_max_length, so the dataset
    must pass an explicit max_length for flux (not rely on the tokenizer default)."""

    model_max_length = 1_000_000

    def __call__(self, text: str, *, padding=None, truncation=None, max_length: int = 256, return_tensors=None):
        return SimpleNamespace(input_ids=torch.zeros((1, max_length), dtype=torch.long))


def test_filename_phrase_strips_trigger():
    assert filename_phrase("bufo-offers-cash-money.png") == "offers cash money"
    assert filename_phrase("awesomebufo") == "awesome"
    assert filename_phrase("bufo") == ""
    assert filename_phrase("this-is-fine-bufo.png") == "this is fine"


def test_caption_uses_schema():
    assert caption_for("cowboy-bufo.png") == f"bufo cowboy{SUFFIX}"
    assert caption_for("bufo.png") == f"bufo{SUFFIX}"  # bare trigger, no dangling space
    assert "flat cartoon" in caption_for("bufo-sip.png")


def test_shortcode_matches_caption_path():
    # The emoji interface and training captions must share one schema.
    assert shortcode_to_prompt(":bufo-offers-cash-money:") == caption_for("bufo-offers-cash-money.png")


def test_to_square_rgb_composites_transparency_on_white():
    wide = Image.new("RGBA", (40, 20), (0, 0, 0, 0))  # fully transparent, non-square
    out = to_square_rgb(wide, 64)
    assert out.size == (64, 64)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white


def _build_dataset(tmp_path, n: int = 3):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(n):
        Image.new("RGB", (24, 24), (10 * i, 0, 0)).save(images_dir / f"b{i}.png")
    (tmp_path / "metadata.jsonl").write_text(
        "\n".join(json.dumps({"file_name": f"b{i}.png", "caption": f"a bufo of {i}"}) for i in range(n))
    )


def test_dataset_serves_normalized_pairs(tmp_path):
    _build_dataset(tmp_path, 3)
    ds = BufoDataset(tmp_path, _StubTokenizer(), resolution=16, random_flip=False)
    assert len(ds) == 3
    item = ds[0]
    assert item["pixel_values"].shape == (3, 16, 16)
    assert item["pixel_values"].min() >= -1.0
    assert item["pixel_values"].max() <= 1.0
    assert item["input_ids"].shape == (16,)


def test_dataset_sdxl_second_clip_at_77(tmp_path):
    # sdxl: both tokenizers at their CLIP model_max_length (16 here).
    _build_dataset(tmp_path, 2)
    ds = BufoDataset(tmp_path, _StubTokenizer(), tokenizer_2=_StubTokenizer(), resolution=16, base_kind="sdxl")
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["input_ids_2"].shape == (16,)  # second CLIP, not the T5 budget


def test_dataset_flux_t5_uses_explicit_max_length(tmp_path):
    # flux: CLIP at 16, T5 at the configured budget (256), NOT the tokenizer's huge default.
    _build_dataset(tmp_path, 2)
    ds = BufoDataset(
        tmp_path,
        _StubTokenizer(),
        tokenizer_2=_StubT5Tokenizer(),
        resolution=16,
        base_kind="flux",
        tokenizer_2_max_length=256,
    )
    item = ds[0]
    assert item["input_ids"].shape == (16,)  # CLIP (77 in practice)
    assert item["input_ids_2"].shape == (256,)  # T5 budget, bounded by the dataset


def test_dataset_requires_prepared_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        BufoDataset(tmp_path, _StubTokenizer())


def test_to_square_rgb_crop_vs_pad():
    wide = Image.new("RGBA", (80, 40), (0, 200, 0, 255))
    assert to_square_rgb(wide, 64, crop=True).size == (64, 64)
    assert to_square_rgb(wide, 64, crop=False).size == (64, 64)


def test_curation_roundtrip_and_apply(tmp_path):
    curation = {
        "b0.png": {"file_name": "b0.png", "keep": False},
        "b1.png": {"file_name": "b1.png", "keep": True, "caption": "bufo override"},
    }
    save_curation(tmp_path, curation)
    assert load_curation(tmp_path) == curation

    records = [{"file_name": f"b{i}.png", "caption": f"bufo {i}"} for i in range(3)]
    kept = apply_curation(records, curation)
    names = [r["file_name"] for r in kept]
    assert names == ["b1.png", "b2.png"]  # b0 dropped
    assert kept[0]["caption"] == "bufo override"  # b1 overridden
    assert kept[1]["caption"] == "bufo 2"  # b2 untouched


def test_load_curation_absent(tmp_path):
    assert load_curation(tmp_path) == {}
