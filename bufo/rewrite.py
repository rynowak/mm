"""Map a user request (emoji shortcode or free-form) to a training-schema prompt.

Two backends behind one `Rewriter` interface:
- `RulesRewriter` (default): deterministic shortcode -> schema, no model.
- `LLMRewriter`: a small local instruct LLM turns *arbitrary* intent into the
  schema action phrase (the DALL·E-3 prompt-rewrite trick, constrained to our terse
  sticker schema rather than prose).
"""

from __future__ import annotations

from typing import Protocol

import torch
from mm_training import get_device

from bufo.data import SUFFIX, TRIGGER, shortcode_to_prompt

_DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
_SYSTEM = (
    "You turn a user's request into a SHORT action phrase for a 'bufo' — a green "
    "cartoon frog emoji. Reply with ONLY the action phrase: 2-6 lowercase words, no "
    "punctuation, describing what the frog is doing, its expression, or its props. "
    "Do NOT include the word 'bufo'.\n"
    "Examples:\n"
    ":bufo-offers-cash-money: -> offering cash money\n"
    "sad monday bufo -> crying about monday\n"
    "a bufo celebrating a promotion -> celebrating with confetti\n"
    "ninja -> dressed as a ninja"
)


def clean_action(text: str) -> str:
    """First line, lowercased, 'bufo' dropped, surrounding punctuation trimmed.

    >>> clean_action("Offering cash money.")
    'offering cash money'
    >>> clean_action("a bufo dressed as a ninja\\nextra")
    'dressed as a ninja'
    """
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = line.lower().strip(" .!?,:;\"'`")
    words = [w for w in line.split() if w != "bufo"]
    while words and words[0] in ("a", "an", "the"):  # drop a leading article
        words.pop(0)
    return " ".join(words).strip()


class Rewriter(Protocol):
    def rewrite(self, query: str) -> str: ...


class RulesRewriter:
    """Deterministic shortcode/phrase -> schema prompt (no model)."""

    def rewrite(self, query: str) -> str:
        return shortcode_to_prompt(query)


class LLMRewriter:
    """Small instruct LLM mapping free-form intent -> the caption schema."""

    def __init__(self, model_name: str = _DEFAULT_LLM, device: torch.device | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def _action(self, query: str) -> str:
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query.strip()}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=24, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )
        raw = self.tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return clean_action(raw)

    def rewrite(self, query: str) -> str:
        action = self._action(query)
        return f"{TRIGGER} {action}{SUFFIX}" if action else shortcode_to_prompt(query)


def get_rewriter(kind: str = "rules", **kwargs) -> Rewriter:
    """Factory: 'rules' (default, deterministic) or 'llm' (lazy-loads the model)."""
    if kind == "llm":
        return LLMRewriter(**kwargs)
    if kind == "rules":
        return RulesRewriter()
    raise ValueError(f"Unknown rewriter kind: {kind}")
