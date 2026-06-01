"""Tests for game state and environment."""

from mm_wordle.game import LetterFeedback, WordleEnv

GREEN = LetterFeedback.GREEN
YELLOW = LetterFeedback.YELLOW
GRAY = LetterFeedback.GRAY


class TestComputeFeedback:
    """Tests for WordleEnv.compute_feedback."""

    def test_all_green(self):
        fb = WordleEnv.compute_feedback("crane", "crane")
        assert fb == [GREEN, GREEN, GREEN, GREEN, GREEN]

    def test_all_gray(self):
        fb = WordleEnv.compute_feedback("which", "fudgy")
        assert fb == [GRAY, GRAY, GRAY, GRAY, GRAY]

    def test_basic_yellow(self):
        fb = WordleEnv.compute_feedback("raise", "spare")
        # r: in spare but not at 0 -> YELLOW
        # a: in spare but not at 1 -> YELLOW
        # i: not in spare -> GRAY
        # s: in spare but not at 3 -> YELLOW
        # e: matches at 4 -> GREEN
        assert fb == [YELLOW, YELLOW, GRAY, YELLOW, GREEN]

    def test_mixed_feedback(self):
        fb = WordleEnv.compute_feedback("hello", "world")
        # hello=h,e,l,l,o vs world=w,o,r,l,d
        # Green pass: l at 3 matches world[3]='l' -> GREEN
        # Yellow pass: h=GRAY, e=GRAY, l at 2: no unmatched 'l' left -> GRAY
        # o at 4: world[1]='o' unmatched -> YELLOW
        assert fb == [GRAY, GRAY, GRAY, GREEN, YELLOW]

    def test_duplicate_letters_in_guess_spec_example(self):
        """The example from the spec: target='abbey', guess='aabbb'."""
        fb = WordleEnv.compute_feedback("aabbb", "abbey")
        assert fb == [GREEN, GRAY, GREEN, YELLOW, GRAY]

    def test_duplicate_letters_green_takes_priority(self):
        """When guess has duplicate letters, greens consume target letters first."""
        fb = WordleEnv.compute_feedback("speed", "creep")
        # s: not in creep -> GRAY
        # p: in creep at 4, but not at 1 -> YELLOW
        # e: creep[2]='e' -> GREEN
        # e: creep[3]='e' -> GREEN
        # d: not in creep -> GRAY
        assert fb == [GRAY, YELLOW, GREEN, GREEN, GRAY]

    def test_duplicate_letters_limited_yellows(self):
        """Only as many yellows as unmatched occurrences in target."""
        fb = WordleEnv.compute_feedback("geese", "steal")
        # geese=g,e,e,s,e vs steal=s,t,e,a,l
        # Green pass: e at 2 matches steal[2]='e' -> GREEN. target_matched[2]=True.
        # Yellow pass: g=GRAY, e at 1: no unmatched e -> GRAY,
        # s at 3: steal[0]='s' unmatched -> YELLOW, e at 4: no unmatched e -> GRAY
        assert fb == [GRAY, GRAY, GREEN, YELLOW, GRAY]

    def test_duplicate_target_letters(self):
        """Target has duplicate letters, guess has one."""
        fb = WordleEnv.compute_feedback("broad", "foods")
        # broad=b,r,o,a,d vs foods=f,o,o,d,s
        # Green pass: o at 2 matches foods[2]='o' -> GREEN. target_matched[2]=True.
        # Yellow pass: b=GRAY, r=GRAY, a=GRAY, d at 4: foods[3]='d' unmatched -> YELLOW
        assert fb == [GRAY, GRAY, GREEN, GRAY, YELLOW]

    def test_all_same_letter_guess(self):
        """Guess is all same letter."""
        fb = WordleEnv.compute_feedback("sssss", "class")
        # target 'class' has 's' at positions 3 and 4
        # s at 0: not at 0, but unmatched s at 3 -> YELLOW
        # s at 1: not at 1, but unmatched s at 4 -> YELLOW
        # s at 2: not at 2, no more unmatched s -> GRAY
        # s at 3: matches target[3]='s' -> GREEN (assigned in pass 1)
        # s at 4: matches target[4]='s' -> GREEN (assigned in pass 1)
        # Wait, greens first: s at 3 -> GREEN, s at 4 -> GREEN
        # Then yellows: s at 0: no more unmatched -> GRAY
        # s at 1: GRAY, s at 2: GRAY
        assert fb == [GRAY, GRAY, GRAY, GREEN, GREEN]


class TestWordleEnv:
    """Tests for WordleEnv lifecycle."""

    def test_reset_random(self):
        env = WordleEnv()
        state = env.reset()
        assert len(state.target) == 5
        assert state.turn == 0
        assert state.guesses == []
        assert not state.solved
        assert not state.failed

    def test_reset_specific_word(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        assert state.target == "crane"

    def test_step_correct_guess(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, done = env.step(state, "crane")
        assert done
        assert state.solved
        assert not state.failed
        assert state.turn == 1

    def test_step_wrong_guess(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, done = env.step(state, "house")
        assert not done
        assert not state.solved
        assert not state.failed
        assert state.turn == 1
        assert len(state.guesses) == 1

    def test_step_six_wrong_guesses_fails(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        words = ["house", "plant", "drive", "blimp", "foggy", "dusty"]
        for i, word in enumerate(words):
            state, done = env.step(state, word)
            if i < 5:
                assert not done
            else:
                assert done
                assert state.failed
                assert not state.solved

    def test_step_solve_on_last_turn(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        words = ["house", "plant", "drive", "blimp", "foggy", "crane"]
        for word in words:
            state, done = env.step(state, word)
        assert done
        assert state.solved
        assert not state.failed
        assert state.turn == 6

    def test_step_lowercases_guess(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, done = env.step(state, "CRANE")
        assert done
        assert state.solved


class TestRender:
    """Tests for WordleEnv.render."""

    def test_render_empty(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        rendered = env.render(state)
        assert "Turn 0/6" in rendered

    def test_render_solved(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "crane")
        rendered = env.render(state)
        assert "Solved in 1 guess!" in rendered

    def test_render_failed(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        for word in ["house", "plant", "drive", "blimp", "foggy", "dusty"]:
            state, _ = env.step(state, word)
        rendered = env.render(state)
        assert "Failed!" in rendered
        assert "CRANE" in rendered

    def test_render_in_progress(self):
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "house")
        rendered = env.render(state)
        assert "Turn 1/6" in rendered
        assert "H O U S E" in rendered
