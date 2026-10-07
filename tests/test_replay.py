"""Phase 15: canonical events, JSON format and replay."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from poker_alpha.history import (HandStarted, PlayerCalled, PlayerFolded,
                                 PotAwarded, ReplayError, dump_hands,
                                 load_hands, parse_hands, replay_hand)
from poker_alpha.holdem import Street, is_valid
from poker_alpha.opponent import PlayerStatistics
from poker_alpha.poker import card_code

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hands" / "sample.json"


@pytest.fixture(scope="module")
def hands():
    return load_hands(FIXTURE)


def test_replay_reconstructs_states_and_decisions(hands):
    res = replay_hand(hands[0])
    assert res.hand_id == "demo-1"
    assert [d.action for d in res.decisions] == ["raise", "bet", "bet"]
    pre, flop, turn = (d.state for d in res.decisions)
    assert pre.street == Street.PREFLOP and pre.amount_to_call == 1.0
    assert pre.hero_position == "BTN"
    assert flop.board == tuple(card_code(c) for c in ("Qs", "Js", "4h"))
    assert flop.pot == pytest.approx(5.5)
    assert turn.pot == pytest.approx(12.5)
    for _, obs in res.snapshots:
        assert is_valid(obs)
    assert res.awards == {0: 21.5}
    assert res.net[0] == pytest.approx(6.5) and res.net[2] == pytest.approx(-6)
    assert sum(res.net.values()) == pytest.approx(0.0)


def test_replay_showdown_and_no_card_leak(hands):
    res = replay_hand(hands[1])
    first = res.decisions[0].state
    assert first.hero_cards == (card_code("Qh"), card_code("Qd"))
    assert first.shown_cards == ()
    for seat in first.seats:
        assert not hasattr(seat, "hole_cards")
    assert res.awards == {0: 100.0}
    assert res.net == {0: 50.0, 1: -50.0}
    assert res.summary.showdown_seats == (0, 1)
    assert res.decisions[1].state.amount_to_call == pytest.approx(40.0)


def test_round_trip_json(hands):
    again = parse_hands(json.loads(json.dumps(dump_hands(hands))))
    assert again == hands
    with pytest.raises(ValueError):
        parse_hands({"format": "other/v1", "hands": []})
    with pytest.raises(ValueError):
        parse_hands({"format": "pokeralpha.hand/v1",
                     "events": [{"type": "Nope"}]})


def test_summaries_feed_player_statistics(hands):
    stats = PlayerStatistics()
    for k, h in enumerate(hands):
        stats.add_hand(replay_hand(h, hand_index=k).summary)
    assert stats.estimate("hero", "pfr").raw_successes == 2
    assert stats.estimate("hero", "flop_cbet").raw_successes == 1
    assert stats.estimate("villain", "wsd").raw_successes == 1


def _mutate(hands, idx, new):
    events = list(hands[0])
    events[idx] = new
    return events


def test_inconsistent_histories_rejected(hands):
    h = hands[0]
    out_of_turn = _mutate(hands, 4, PlayerFolded(seat=5))
    with pytest.raises(ReplayError, match="out of turn"):
        replay_hand(out_of_turn)
    bad_award = list(h[:-1]) + [PotAwarded(seat=0, amount=20.0)]
    with pytest.raises(ReplayError, match="PotAwarded"):
        replay_hand(bad_award)
    with pytest.raises(ReplayError, match="mid-hand"):
        replay_hand(h[:12])
    bad_blind = list(h)
    from poker_alpha.history import BlindPosted
    bad_blind[1] = BlindPosted(seat=1, amount=1.0, blind="sb")
    with pytest.raises(ReplayError, match="sb post"):
        replay_hand(bad_blind)
    with pytest.raises(ReplayError):
        replay_hand(h[1:])


def test_cli_replay_and_recommend():
    out = subprocess.run([sys.executable, "-m", "poker_alpha.replay",
                          str(FIXTURE)], capture_output=True, text=True,
                         check=True, cwd=ROOT).stdout
    assert "hero action: raise 2.5" in out and "seat 0: +6.5" in out
    out = subprocess.run([sys.executable, "-m", "poker_alpha.replay",
                          str(FIXTURE), "--recommend", "--rollouts", "150",
                          "--equity-sims", "200"], capture_output=True,
                         text=True, check=True, cwd=ROOT).stdout
    assert "Recommendation:" in out and "actual:" in out
