"""V3 tokenizer: re-exports the shared constraint-state tokenizer (ADR-5).

The representation now lives in ``mm_wordle`` so V2 and V3 share one source of
truth. V3 imports it from here.
"""

from mm_wordle.constraint_tokenizer import ConstraintTokenizer

__all__ = ["ConstraintTokenizer"]
