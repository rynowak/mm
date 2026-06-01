"""Tests for word lists."""

from mm_wordle.words import all_valid_words, load_answers, load_valid_guesses


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
