"""Tests for the batched RL rollout (sampling + gradient-capable log-probs)."""

import torch
import torch.nn.functional as F
from mm_model import GPT, GPTConfig
from mm_wordle import ConstraintTokenizer

from wordle3.metrics import build_letter_mask
from wordle3.rollout import group_log_probs, sample_group


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=265, context_len=128, dropout=0.0)).eval()


def test_sample_group_shapes_and_letters():
    tok = ConstraintTokenizer()
    model = _model()
    mask = build_letter_mask(tok, torch.device("cpu"))
    words, gen = sample_group(model, tok.empty_prompt(), 6, torch.device("cpu"), mask, tok)
    assert gen.shape == (6, 5)
    assert len(words) == 6
    for w in words:
        assert len(w) == 5 and w.isalpha()
    assert set(gen.flatten().tolist()) <= tok.letter_ids  # only letter tokens sampled


def test_sample_group_reproducible_with_seed():
    tok = ConstraintTokenizer()
    model = _model()
    mask = build_letter_mask(tok, torch.device("cpu"))
    torch.manual_seed(123)
    _, g1 = sample_group(model, tok.empty_prompt(), 4, torch.device("cpu"), mask, tok)
    torch.manual_seed(123)
    _, g2 = sample_group(model, tok.empty_prompt(), 4, torch.device("cpu"), mask, tok)
    assert torch.equal(g1, g2)


def test_group_log_probs_matches_per_sequence_reference():
    tok = ConstraintTokenizer()
    model = _model()
    device = torch.device("cpu")
    mask = build_letter_mask(tok, device)
    prompt = tok.empty_prompt()
    letters = sorted(tok.letter_ids)
    gen = torch.tensor([[letters[i] for i in range(5)], [letters[i + 5] for i in range(5)]])  # (2, 5)

    got = group_log_probs(model, prompt, gen, device, mask)

    # Reference: forward prompt+row per sequence, masked log-softmax, gather, sum.
    ref = []
    for row in gen:
        full = torch.tensor(prompt + row.tolist()).unsqueeze(0)
        logits, _, _ = model(full)
        gen_logits = logits[0, len(prompt) - 1 : len(prompt) + 4, :] + mask
        lp = F.log_softmax(gen_logits, dim=-1)
        ref.append(lp.gather(1, row.unsqueeze(1)).squeeze(1).sum())
    ref_t = torch.stack(ref)

    assert torch.allclose(got, ref_t, atol=1e-5)


def test_group_log_probs_has_gradient_outside_no_grad():
    tok = ConstraintTokenizer()
    model = _model()
    device = torch.device("cpu")
    mask = build_letter_mask(tok, device)
    letters = sorted(tok.letter_ids)
    gen = torch.tensor([[letters[i] for i in range(5)]])
    lp = group_log_probs(model, tok.empty_prompt(), gen, device, mask)
    assert lp.requires_grad  # usable for the PPO current-policy pass
