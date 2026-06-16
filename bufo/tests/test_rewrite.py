"""Offline tests for the rewriter (rules path + clean_action); LLM gated."""

from __future__ import annotations

import os

import pytest

from bufo.data import SUFFIX, TRIGGER
from bufo.rewrite import RulesRewriter, clean_action, get_rewriter


def test_clean_action():
    assert clean_action("Offering cash money.") == "offering cash money"
    assert clean_action("a bufo dressed as a ninja\nextra") == "dressed as a ninja"
    assert clean_action("  ") == ""


def test_rules_rewriter_shortcode():
    r = RulesRewriter()
    assert r.rewrite(":bufo-offers-cash-money:") == f"{TRIGGER} offers cash money{SUFFIX}"


def test_get_rewriter_default_is_rules():
    assert isinstance(get_rewriter(), RulesRewriter)
    assert isinstance(get_rewriter("rules"), RulesRewriter)
    with pytest.raises(ValueError, match="Unknown rewriter"):
        get_rewriter("bogus")


@pytest.mark.skipif(os.environ.get("BUFO_LLM_SMOKE") != "1", reason="Set BUFO_LLM_SMOKE=1 (downloads a ~3GB LLM).")
def test_llm_rewriter_returns_schema():
    from bufo.rewrite import LLMRewriter

    out = LLMRewriter().rewrite("sad monday bufo")
    assert out.startswith(f"{TRIGGER} ") and out.endswith(SUFFIX)
