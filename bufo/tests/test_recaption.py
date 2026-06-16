"""Offline tests for recaption merge logic (no VLM download)."""

from __future__ import annotations

from bufo.data import SUFFIX, TRIGGER
from bufo.recaption import clean_detail, merge_caption


def test_clean_detail_strips_boilerplate():
    assert clean_detail("a cartoon of a green frog wearing a red hat") == "wearing red hat"
    assert clean_detail("an image of a toad") == ""  # all boilerplate -> empty


def test_merge_caption_dedupes_and_uses_schema():
    out = merge_caption("offers cash money", "holding money")
    assert out == f"{TRIGGER} offers cash money, holding{SUFFIX}"  # "money" deduped from detail


def test_merge_caption_empty_detail():
    assert merge_caption("cowboy", "") == f"{TRIGGER} cowboy{SUFFIX}"


def test_merge_caption_no_action():
    # bare trigger when there's no action and no detail
    assert merge_caption("", "") == f"{TRIGGER}{SUFFIX}"
