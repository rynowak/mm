"""Offline tests for the review gallery helpers (no HTTP server)."""

from __future__ import annotations

from bufo.data import load_curation
from bufo.gallery import _card, set_keep


def test_set_keep_persists(tmp_path):
    set_keep(tmp_path, "bufo-sip.png", keep=False)
    assert load_curation(tmp_path)["bufo-sip.png"]["keep"] is False
    set_keep(tmp_path, "bufo-sip.png", keep=True)  # restore
    assert load_curation(tmp_path)["bufo-sip.png"]["keep"] is True


def test_set_keep_preserves_caption(tmp_path):
    from bufo.data import save_curation

    save_curation(tmp_path, {"b.png": {"file_name": "b.png", "keep": True, "caption": "bufo custom"}})
    set_keep(tmp_path, "b.png", keep=False)
    rec = load_curation(tmp_path)["b.png"]
    assert rec["keep"] is False
    assert rec["caption"] == "bufo custom"  # override survives a drop toggle


def test_card_reflects_state_and_chips():
    rec = {"file_name": "bufo-sip.png", "caption": "bufo sip, flat cartoon"}
    kept = _card(rec, {"scene_like": False, "dup_group": -1}, dropped=False)
    assert "Drop" in kept and "card dropped" not in kept

    flagged = _card(rec, {"scene_like": True, "dup_group": 3}, dropped=True)
    assert "Restore" in flagged and "card dropped" in flagged
    assert "scene" in flagged and "dup 3" in flagged
