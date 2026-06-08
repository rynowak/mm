"""Hold-out hard gate on the committed split.json artifact.

If this fails, the generalization metric is meaningless — regenerate the split
with ``wordle3/make_split.py`` and investigate any divergence.
"""

from mm_wordle import load_full_word_set, split_answers

from wordle3.splits import load_split


def test_split_partitions_universe():
    split = load_split()
    train, holdout = set(split.train_answers), set(split.holdout)
    universe = set(load_full_word_set())
    assert train.isdisjoint(holdout), "train and hold-out overlap"
    assert train | holdout == universe, "split does not cover the universe"
    assert len(train) + len(holdout) == len(universe) == 14855


def test_split_holdout_nonempty():
    split = load_split()
    assert len(split.holdout) > 0
    assert len(split.train_answers) > 0


def test_committed_split_matches_generator():
    """The committed file must equal a fresh deterministic split for its seed/frac."""
    split = load_split()
    train, holdout = split_answers(holdout_frac=split.holdout_frac, seed=split.seed)
    assert split.train_answers == train
    assert split.holdout == holdout
