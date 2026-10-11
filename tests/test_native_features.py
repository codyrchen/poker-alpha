"""Native card-feature parity (Phase 11 support).

The infoset-key parity tests already pin the end-to-end encoder; these pin
the individual feature components (strength ladder, draws, nuts flag,
blockers, texture, river percentile bucket) so a future failure localizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_alpha.abstraction.features import card_features, river_percentile

from native_helpers import native


def _assert_match(hole, board, buckets=20):
    f = card_features(hole, board)
    nf = native.debug_features(list(hole), list(board), buckets)
    assert nf["strength"] == f.strength, (hole, board)
    assert nf["draw"] == f.draw, (hole, board)
    assert nf["nut"] == f.nut, (hole, board)
    assert nf["blocker"] == f.blocker, (hole, board)
    assert nf["texture"] == f.texture, (hole, board)
    if len(board) == 5 and buckets:
        pct = river_percentile(hole, board)
        expect = min(int(pct * buckets), buckets - 1)
        assert nf["pct_bucket"] == expect, (hole, board)


def _random_cases(n, board_n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        cards = rng.choice(52, size=2 + board_n, replace=False).tolist()
        yield tuple(cards[:2]), tuple(cards[2:])


def test_flop_features_parity():
    for hole, board in _random_cases(1500, 3, seed=21):
        _assert_match(hole, board)


def test_turn_features_parity():
    for hole, board in _random_cases(1500, 4, seed=22):
        _assert_match(hole, board)


def test_river_features_parity():
    for hole, board in _random_cases(600, 5, seed=23):
        _assert_match(hole, board)


@pytest.mark.slow
def test_river_features_parity_large():
    for hole, board in _random_cases(5000, 5, seed=24):
        _assert_match(hole, board)


def test_monotone_and_paired_boards():
    """Directed cases: nut flushes, boards that play, blockers."""
    # As Ks on a monotone spade board: nut flush, current nuts.
    cases = [
        ((51, 50), (49, 48, 44)),       # AsKs on QsJs7s — nut flush
        ((12, 11), (49, 48, 44)),       # AdKd (no spade) on spades
        ((51, 11), (49, 48, 44, 43)),   # As blocker, 4-flush board
        ((0, 13), (9, 22, 35, 48, 2)),  # quads potential board
        ((12, 25), (8, 21, 34, 47, 7)), # board two pair
        ((5, 18), (12, 25, 38, 51, 0)), # quad aces on board — board plays
    ]
    for hole, board in cases:
        _assert_match(hole, board)


def _flush_heavy_boards(n, seed):
    """Boards with exactly 3, 4 or 5 cards of one suit (the cases where the
    native river table deduplicates flush-suit holdings by exact card)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        k = 3 + i % 3
        suit = int(rng.integers(4))
        suited = rng.choice(13, size=k, replace=False) * 0 + suit * 13 + rng.choice(13, size=k, replace=False)
        others = [c for c in range(52) if c // 13 != suit]
        rest = rng.choice(others, size=5 - k, replace=False).tolist()
        board = [int(c) for c in suited] + rest
        live = [c for c in range(52) if c not in board]
        hole = rng.choice(live, size=2, replace=False).tolist()
        yield tuple(hole), tuple(board)


def test_river_percentile_fine_buckets_on_flush_boards():
    # 255 buckets resolves percentile differences of < 1/255; pins the
    # deduplicated native river table (Phase II) against the Python
    # 1,081-holding enumeration on the boards where dedup is most delicate.
    for hole, board in _flush_heavy_boards(900, seed=31):
        _assert_match(hole, board, buckets=255)
    for hole, board in _random_cases(600, 5, seed=32):
        _assert_match(hole, board, buckets=255)
