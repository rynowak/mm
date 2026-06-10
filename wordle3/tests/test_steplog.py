"""Tests for eval-target sampling (must be decoupled from the training seed)."""

from wordle3.splits import Split
from wordle3.steplog import EVAL_SEED, sample_eval_targets


def _split() -> Split:
    return Split(
        seed=42,  # the training seed — eval sampling must NOT depend on this
        holdout_frac=0.1,
        train_answers=[f"t{i:04d}" for i in range(300)],
        holdout=[f"h{i:03d}" for i in range(120)],
    )


def test_eval_seed_differs_from_training_seed():
    # The whole point of the fix: eval sampling is decoupled from the training seed.
    assert EVAL_SEED != 42


def test_sample_eval_targets_deterministic():
    split = _split()
    a = sample_eval_targets(split, 16, 64)
    b = sample_eval_targets(split, 16, 64)
    assert a == b  # fixed EVAL_SEED -> reproducible across runs


def test_sample_eval_targets_sizes_and_pools():
    split = _split()
    step_train, step_holdout, big_train, big_holdout = sample_eval_targets(split, 16, 64)
    assert len(step_train) == 16 and len(step_holdout) == 16
    assert len(big_train) == 64 and len(big_holdout) == 64
    # Hold-out eval sets come only from the explicit hold-out (the generalization set).
    assert set(step_holdout) <= set(split.holdout)
    assert set(big_holdout) <= set(split.holdout)
    assert set(step_train) <= set(split.train_answers)
    assert set(big_train) <= set(split.train_answers)


def test_step_games_zero_disables_step_sets():
    split = _split()
    step_train, step_holdout, big_train, _ = sample_eval_targets(split, 0, 32)
    assert step_train == [] and step_holdout == []
    assert len(big_train) == 32
