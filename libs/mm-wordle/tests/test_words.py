"""Tests for word lists."""

from mm_wordle.words import (
    all_valid_words,
    load_answers,
    load_full_word_set,
    load_valid_guesses,
    split_answers,
)


def test_load_answers_returns_list():
    answers = load_answers()
    assert isinstance(answers, list)
    assert len(answers) > 500


def test_load_answers_all_five_letters():
    answers = load_answers()
    for word in answers:
        assert len(word) == 5, f"Answer '{word}' is not 5 letters"
        assert word.isalpha(), f"Answer '{word}' contains non-alpha characters"
        assert word.islower(), f"Answer '{word}' is not lowercase"


def test_load_valid_guesses_returns_list():
    guesses = load_valid_guesses()
    assert isinstance(guesses, list)
    assert len(guesses) > 2000


def test_load_valid_guesses_all_five_letters():
    guesses = load_valid_guesses()
    for word in guesses:
        assert len(word) == 5, f"Guess '{word}' is not 5 letters"
        assert word.isalpha(), f"Guess '{word}' contains non-alpha characters"
        assert word.islower(), f"Guess '{word}' is not lowercase"


def test_all_valid_words_is_union():
    answers = set(load_answers())
    guesses = set(load_valid_guesses())
    valid = all_valid_words()
    assert valid == answers | guesses


def test_answers_not_in_guesses():
    """Answers and guesses should be disjoint lists."""
    answers = set(load_answers())
    guesses = set(load_valid_guesses())
    # The lists should not overlap (answers are separate from additional guesses)
    overlap = answers & guesses
    assert len(overlap) == 0, f"Found {len(overlap)} words in both lists"


def test_known_words_present():
    """Spot-check that some well-known Wordle words are in the answers."""
    answers = set(load_answers())
    for word in ["crane", "slate", "audio", "arise", "about"]:
        assert word in answers, f"Expected '{word}' in answers"


# --- V3 full word set + hold-out split ---


def test_full_word_set_size():
    words = load_full_word_set()
    assert len(words) == 14855


def test_full_word_set_sorted_unique_five_letters():
    words = load_full_word_set()
    assert words == sorted(words), "full word set must be sorted"
    assert len(set(words)) == len(words), "full word set must be unique"
    for word in words:
        assert len(word) == 5 and word.isalpha() and word.islower()


def test_split_is_deterministic():
    train_a, holdout_a = split_answers(holdout_frac=0.10, seed=1234)
    train_b, holdout_b = split_answers(holdout_frac=0.10, seed=1234)
    assert train_a == train_b
    assert holdout_a == holdout_b


def test_split_seed_changes_partition():
    _, holdout_a = split_answers(holdout_frac=0.10, seed=1234)
    _, holdout_b = split_answers(holdout_frac=0.10, seed=9999)
    assert set(holdout_a) != set(holdout_b)


def test_split_disjoint_and_covers_universe():
    """The hold-out hard gate: train and hold-out partition the full set exactly."""
    train, holdout = split_answers(holdout_frac=0.10, seed=1234)
    train_set, holdout_set = set(train), set(holdout)
    universe = set(load_full_word_set())
    assert train_set.isdisjoint(holdout_set), "train and hold-out must be disjoint"
    assert train_set | holdout_set == universe, "train ∪ hold-out must equal the full set"
    assert len(train) + len(holdout) == len(universe)


def test_split_holdout_fraction():
    train, holdout = split_answers(holdout_frac=0.10, seed=1234)
    assert len(holdout) == int(14855 * 0.10)
    assert len(holdout) > 0 and len(train) > 0


def test_split_lists_sorted():
    train, holdout = split_answers(holdout_frac=0.10, seed=1234)
    assert train == sorted(train)
    assert holdout == sorted(holdout)
