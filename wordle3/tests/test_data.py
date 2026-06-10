"""Tests for the V3 word-only pre-train data pipeline."""

import torch
from mm_wordle import ConstraintTokenizer, PatternMatrix, load_full_word_set

from wordle3.data import WordOnlyDataset, collate_padded, generate_retrieval_examples


def test_dataset_length_matches_words():
    tok = ConstraintTokenizer()
    ds = WordOnlyDataset(["crane", "slate", "audio"], tok)
    assert len(ds) == 3


def test_example_shapes_and_mask():
    tok = ConstraintTokenizer()
    ds = WordOnlyDataset(["crane"], tok)
    input_ids, target_ids, loss_mask = ds[0]
    # prompt (7) + word (5) = 12 tokens -> sequences of length 11
    assert input_ids.shape == (11,)
    assert target_ids.shape == (11,)
    assert loss_mask.sum().item() == 5.0  # only the 5 letter positions
    assert loss_mask[:6].sum().item() == 0.0  # prompt positions unmasked-out
    assert loss_mask[6:].sum().item() == 5.0


def test_target_positions_decode_to_word():
    tok = ConstraintTokenizer()
    ds = WordOnlyDataset(["audio"], tok)
    _, target_ids, loss_mask = ds[0]
    letters = [int(t) for t, m in zip(target_ids.tolist(), loss_mask.tolist(), strict=True) if m == 1.0]
    assert tok.decode_letters(letters) == "audio"


def test_prompt_is_empty_constraint_state():
    tok = ConstraintTokenizer()
    ds = WordOnlyDataset(["crane"], tok)
    input_ids, _, _ = ds[0]
    # input is (prompt + word)[:-1]; first 7 are the empty prompt
    assert input_ids[:7].tolist() == tok.empty_prompt()


def test_collate_stacks_batch():
    tok = ConstraintTokenizer()
    ds = WordOnlyDataset(["crane", "slate", "audio", "ample"], tok)
    batch = [ds[i] for i in range(4)]
    input_ids, target_ids, loss_masks = collate_padded(batch, tok.pad_id)
    assert input_ids.shape == (4, 11)
    assert target_ids.shape == (4, 11)
    assert loss_masks.shape == (4, 11)
    assert torch.equal(loss_masks.sum(dim=1), torch.full((4,), 5.0))


def test_generate_retrieval_examples_targets_are_words_and_states_tight():
    tok = ConstraintTokenizer()
    words = load_full_word_set()[:80]
    pm = PatternMatrix.from_words(words)
    examples = generate_retrieval_examples(tok, pm, words[:30], games_per_word=2, seed=0, max_candidates=3)

    assert len(examples) > 0
    valid = set(words)
    empty_len = len(tok.empty_prompt())
    saw_constrained = False
    for prompt, target in examples:
        assert prompt[0] == tok.bos_id
        assert len(target) == 5
        assert tok.decode_letters(target) in valid  # target is the answer word
        if len(prompt) > empty_len:
            saw_constrained = True
    # Tight states carry real constraints (facts beyond the empty [bos] ????? [sep]).
    assert saw_constrained
