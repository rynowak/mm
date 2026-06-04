"""Tests for the Tetris environment."""

import mm_tetris


class TestPieces:
    def test_all_pieces_have_four_rotations(self) -> None:
        for name in mm_tetris.PIECE_NAMES:
            assert len(mm_tetris.ROTATIONS[name]) == 4

    def test_all_pieces_have_four_cells(self) -> None:
        for name in mm_tetris.PIECE_NAMES:
            for rot, shape in enumerate(mm_tetris.ROTATIONS[name]):
                assert len(shape.cells) == 4, f"{name} rot {rot} has {len(shape.cells)} cells"

    def test_piece_dimensions_consistent(self) -> None:
        for name in mm_tetris.PIECE_NAMES:
            for shape in mm_tetris.ROTATIONS[name]:
                for r, c in shape.cells:
                    assert 0 <= r < shape.height, f"{name}: row {r} >= height {shape.height}"
                    assert 0 <= c < shape.width, f"{name}: col {c} >= width {shape.width}"

    def test_seven_piece_types(self) -> None:
        assert len(mm_tetris.PIECE_NAMES) == 7
        assert mm_tetris.NUM_PIECE_TYPES == 7


class TestReset:
    def test_empty_grid(self) -> None:
        state = mm_tetris.reset(seed=0)
        for row in state.grid:
            assert not any(row)

    def test_initial_values(self) -> None:
        state = mm_tetris.reset(seed=0)
        assert state.score == 0
        assert state.lines_cleared == 0
        assert state.pieces_placed == 0
        assert not state.game_over

    def test_deterministic_piece(self) -> None:
        s1 = mm_tetris.reset(seed=42)
        s2 = mm_tetris.reset(seed=42)
        assert s1.current_piece == s2.current_piece

    def test_different_seeds_can_differ(self) -> None:
        pieces = {mm_tetris.reset(seed=i).current_piece for i in range(100)}
        assert len(pieces) > 1


class TestValidActions:
    def test_fresh_board_has_valid_actions(self) -> None:
        state = mm_tetris.reset(seed=0)
        mask = mm_tetris.valid_action_mask(state)
        assert any(mask)

    def test_game_over_has_no_actions(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.game_over = True
        mask = mm_tetris.valid_action_mask(state)
        assert not any(mask)

    def test_wide_pieces_cant_go_far_right(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.current_piece = "I"
        mask = mm_tetris.valid_action_mask(state)
        # I-piece horizontal (rot 0) has width 4, valid cols 0-6
        for col in range(7):
            action = mm_tetris.placement_to_action(0, col)
            assert mask[action], f"I horizontal should fit at col {col}"
        for col in range(7, 10):
            action = mm_tetris.placement_to_action(0, col)
            assert not mask[action], f"I horizontal should NOT fit at col {col}"

    def test_action_count(self) -> None:
        assert mm_tetris.NUM_ACTIONS == 40


class TestStep:
    def test_piece_placed_on_bottom(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.current_piece = "O"
        action = mm_tetris.placement_to_action(0, 0)
        new_state = mm_tetris.step(state, action)
        assert new_state.grid[19][0]
        assert new_state.grid[19][1]
        assert new_state.grid[18][0]
        assert new_state.grid[18][1]

    def test_pieces_placed_increments(self) -> None:
        state = mm_tetris.reset(seed=0)
        mask = mm_tetris.valid_action_mask(state)
        action = next(i for i, v in enumerate(mask) if v)
        new_state = mm_tetris.step(state, action)
        assert new_state.pieces_placed == 1

    def test_new_piece_spawned(self) -> None:
        state = mm_tetris.reset(seed=42)
        mask = mm_tetris.valid_action_mask(state)
        action = next(i for i, v in enumerate(mask) if v)
        new_state = mm_tetris.step(state, action)
        assert new_state.current_piece in mm_tetris.PIECE_NAMES
        assert not new_state.game_over

    def test_deterministic_gameplay(self) -> None:
        actions = [0, 5, 10, 15, 20]
        states_a: list[mm_tetris.TetrisState] = []
        states_b: list[mm_tetris.TetrisState] = []
        for run_states in [states_a, states_b]:
            state = mm_tetris.reset(seed=99)
            for a in actions:
                mask = mm_tetris.valid_action_mask(state)
                if not mask[a]:
                    break
                state = mm_tetris.step(state, a)
                run_states.append(state)
        for a, b in zip(states_a, states_b, strict=True):
            assert a.grid == b.grid
            assert a.score == b.score

    def test_invalid_action_ends_game(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.current_piece = "I"
        action = mm_tetris.placement_to_action(0, 9)  # I-piece won't fit at col 9
        new_state = mm_tetris.step(state, action)
        assert new_state.game_over


class TestLineClear:
    def test_full_row_cleared(self) -> None:
        state = mm_tetris.reset(seed=0)
        # Fill the bottom row except one cell
        for col in range(mm_tetris.GRID_WIDTH):
            state.grid[19][col] = True
        state.grid[19][0] = False
        state.grid[19][1] = False

        # Place an O-piece at col 0 to complete the bottom row
        state.current_piece = "O"
        action = mm_tetris.placement_to_action(0, 0)
        new_state = mm_tetris.step(state, action)
        assert new_state.lines_cleared >= 1
        assert new_state.score > 0


class TestStateEncoding:
    def test_flat_size(self) -> None:
        state = mm_tetris.reset(seed=0)
        flat = mm_tetris.state_to_flat(state)
        assert len(flat) == mm_tetris.STATE_SIZE

    def test_state_size_constant(self) -> None:
        assert mm_tetris.STATE_SIZE == 207

    def test_piece_onehot(self) -> None:
        state = mm_tetris.reset(seed=0)
        flat = mm_tetris.state_to_flat(state)
        piece_part = flat[mm_tetris.GRID_WIDTH * mm_tetris.GRID_HEIGHT :]
        assert sum(piece_part) == 1.0
        assert len(piece_part) == mm_tetris.NUM_PIECE_TYPES


class TestHelpers:
    def test_column_heights_empty(self) -> None:
        state = mm_tetris.reset(seed=0)
        assert all(h == 0 for h in state.column_heights())

    def test_column_heights_after_placement(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.current_piece = "O"
        new_state = mm_tetris.step(state, mm_tetris.placement_to_action(0, 0))
        heights = new_state.column_heights()
        assert heights[0] == 2
        assert heights[1] == 2
        assert heights[2] == 0

    def test_count_holes(self) -> None:
        state = mm_tetris.reset(seed=0)
        state.grid[18][0] = True  # block above
        state.grid[19][0] = False  # hole below
        assert state.count_holes() == 1

    def test_render_runs(self) -> None:
        state = mm_tetris.reset(seed=0)
        text = mm_tetris.render(state)
        assert "Piece:" in text
        assert "Score:" in text

    def test_action_encoding_roundtrip(self) -> None:
        for rot in range(mm_tetris.NUM_ROTATIONS):
            for col in range(mm_tetris.GRID_WIDTH):
                action = mm_tetris.placement_to_action(rot, col)
                r, c = mm_tetris.action_to_placement(action)
                assert r == rot and c == col
