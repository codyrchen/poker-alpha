"""Phase 11: Hold'em player statistics with uncertainty and recency."""

from types import SimpleNamespace as A

import pytest

from poker_alpha.opponent.statistics import (STATS, DecayedBeta, HandSummary,
                                             PlayerStatistics, opportunities)

POS = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG"}
PLAYERS = {0: "btn", 1: "sb", 2: "bb", 3: "utg"}


def hand(i, actions, showdown=(), won=None):
    return HandSummary(hand_index=i, players=PLAYERS, positions=POS,
                       actions=[A(street=s, seat=seat, kind=k, amount=amt)
                                for s, seat, k, amt in actions],
                       big_blind_seat=2, big_blind=2.0,
                       showdown_seats=tuple(showdown), won=won or {})


def ops(h):
    return {(seat, stat, occ) for seat, stat, _, occ in opportunities(h)}


# UTG opens, BTN 3-bets, UTG calls; flop: UTG checks, BTN c-bets, UTG
# check-raises, BTN folds.
LINE = [(0, 3, "raise", 6), (0, 0, "raise", 18), (0, 1, "fold", 0),
        (0, 2, "fold", 0), (0, 3, "call", 18),
        (1, 3, "check", 0), (1, 0, "bet", 12), (1, 3, "raise", 40),
        (1, 0, "fold", 0)]


def test_preflop_opportunities():
    o = ops(hand(1, LINE))
    assert (3, "vpip", True) in o and (3, "pfr", True) in o
    assert (0, "three_bet", True) in o and (0, "cold_call", False) in o
    assert (1, "three_bet", False) not in o            # SB faced the 3-bet
    assert (1, "four_bet", False) in o and (2, "four_bet", False) in o
    assert (3, "fold_to_three_bet", False) in o and (3, "four_bet", False) in o
    assert (3, "limp", False) in o                      # first-in, raised
    assert (2, "vpip", False) in o


def test_postflop_opportunities():
    o = ops(hand(1, LINE))
    assert (0, "flop_cbet", True) in o
    assert (3, "fold_to_flop_cbet", False) in o
    assert (3, "check_raise", True) in o
    assert (0, "fold_to_raise", True) in o
    assert (3, "aggression", True) in o and (0, "aggression", True) in o
    assert (0, "aggression", False) in o                # the fold
    assert (0, "wtsd", False) in o and (3, "wtsd", False) in o


def test_barrels_showdown_and_wsd():
    line = [(0, 3, "raise", 6), (0, 0, "call", 6), (0, 1, "fold", 0),
            (0, 2, "fold", 0),
            (1, 3, "bet", 8), (1, 0, "call", 8),
            (2, 3, "bet", 20), (2, 0, "call", 20),
            (3, 3, "check", 0), (3, 0, "check", 0)]
    o = ops(hand(2, line, showdown=(0, 3), won={0: 70}))
    assert (3, "turn_barrel", True) in o
    assert (3, "river_barrel", False) in o
    assert (0, "wtsd", True) in o and (0, "wsd", True) in o
    assert (3, "wsd", False) in o
    assert (0, "cold_call", True) in o


def test_walk_does_not_count_vpip_for_bb_and_all_in_call_resolved():
    walk = hand(3, [(0, 3, "fold", 0), (0, 0, "fold", 0), (0, 1, "fold", 0)])
    assert not any(seat == 2 and stat == "vpip" for seat, stat, _ in ops(walk))
    shove_call = hand(4, [(0, 3, "all_in", 50), (0, 0, "all_in", 30),
                          (0, 1, "fold", 0), (0, 2, "fold", 0)])
    o = ops(shove_call)
    assert (0, "cold_call", True) in o       # all-in for less = a call
    assert (3, "pfr", True) in o


def test_sample_size_changes_uncertainty_not_mean():
    small, large = PlayerStatistics(), PlayerStatistics()
    for i in range(4):
        small.add_hand(hand(i, [(0, 3, "raise" if i < 3 else "fold", 6)]))
    for i in range(400):
        large.add_hand(hand(i, [(0, 3, "raise" if i % 4 else "fold", 6)]))
    a = small.estimate("utg", "pfr")
    b = large.estimate("utg", "pfr")
    assert a.raw_successes / a.raw_opportunities == 0.75
    assert b.raw_successes / b.raw_opportunities == 0.75
    assert a.width > 4 * b.width
    assert "3/4" in a.format()
    assert small.estimate("utg", "pfr", "pos:UTG").raw_opportunities == 4
    assert small.estimate("nobody", "vpip").format() == "vpip: no data"


def test_scopes_by_street():
    st = PlayerStatistics()
    st.add_hand(hand(1, LINE))
    assert st.estimate("btn", "aggression", "street:flop").raw_opportunities == 2
    assert st.estimate("btn", "aggression", "street:turn").raw_opportunities == 0


def test_recency_weighting_tracks_change():
    st = PlayerStatistics(decay=0.95)
    flat = PlayerStatistics(decay=1.0)
    for i in range(200):
        kind = "raise" if i < 100 else "fold"     # player tightens up
        h = hand(i, [(0, 3, kind, 6)])
        st.add_hand(h)
        flat.add_hand(h)
    assert st.estimate("utg", "pfr").mean < 0.1
    assert flat.estimate("utg", "pfr").mean == pytest.approx(101 / 202)
    assert st.estimate("utg", "pfr").effective_samples < 25
    with pytest.raises(ValueError):
        PlayerStatistics(decay=0)


def test_decayed_beta_matches_plain_beta_without_decay():
    d = DecayedBeta()
    for i, occ in enumerate([True, False, True, True]):
        d.observe(occ, i)
    assert d.alpha == 4 and d.beta == 2
    lo, hi = d.interval()
    assert lo < d.mean() < hi


def test_all_stats_reportable():
    st = PlayerStatistics()
    st.add_hand(hand(1, LINE))
    assert set(st.summary("utg")) == set(STATS)
    with pytest.raises(ValueError):
        st.estimate("utg", "nonsense")
