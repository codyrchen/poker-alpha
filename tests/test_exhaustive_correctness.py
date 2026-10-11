"""Phase II Program 9: exhaustive reduced-deck enumeration, metamorphic
checks and fault injection (the invariant checker must catch deliberately
broken engine behaviour; the mutations live only in these tests)."""

import json
from dataclasses import replace

import pytest

from poker_alpha.games import holdem as holdem_mod
from poker_alpha.games.holdem import HoldemGame
from poker_alpha.poker.cards import codes
from poker_alpha.research.exhaustive import (ReducedDeckEnumerator, permute_suits, replay_from_root,
                                             sample_states, state_from_dict, state_to_dict, swap_hole_order)


def tiny_game():
    return HoldemGame(starting_stack=2.0, bet_fractions={"b100": 1.0}, raise_cap=1)


def tiny_deck():
    return codes([r + s for r in "AKQ" for s in "shd"])


@pytest.fixture(scope="module")
def report():
    return ReducedDeckEnumerator(tiny_game(), tiny_deck()).run()


def test_exhaustive_enumeration_has_no_violations(report):
    assert report.states > 100000
    assert report.violations == []
    assert report.terminal_nodes > 0 and report.chance_nodes > 0 and report.decision_nodes > 0
    assert report.states == report.terminal_nodes + report.chance_nodes + report.decision_nodes
    assert not report.truncated
    # every branching count is a legal action-set size (2..4 here)
    assert set(report.branching) <= {2, 3, 4}


def test_enumeration_is_deterministic(report):
    again = ReducedDeckEnumerator(tiny_game(), tiny_deck()).run()
    assert again.states == report.states and again.infosets == report.infosets


def test_metamorphic_suit_hole_serialization_replay():
    g = tiny_game()
    en = ReducedDeckEnumerator(g, tiny_deck())
    for s in sample_states(en, 300, seed=1):
        t = permute_suits(s, (2, 0, 3, 1))
        assert g.is_terminal(t) == g.is_terminal(s)
        if g.is_terminal(s):
            assert g.utility(t) == pytest.approx(g.utility(s))
        else:
            assert g.legal_actions(t) == g.legal_actions(s)
        u = swap_hole_order(s)
        if g.is_terminal(s):
            assert g.utility(u) == pytest.approx(g.utility(s))
        assert state_from_dict(json.loads(json.dumps(state_to_dict(s)))) == s
        r = replay_from_root(g, s)
        assert r.streets == s.streets and r.contrib == pytest.approx(s.contrib)
        assert r.folded == s.folded and r.all_in == s.all_in


# -- fault injection ---------------------------------------------------------

def _run_small(game, monkeypatch=None, max_states=30000):
    return ReducedDeckEnumerator(game, tiny_deck(), max_states=max_states).run()


def test_fault_contrib_cache_corruption_is_caught(monkeypatch):
    g = tiny_game()
    orig = HoldemGame.next_state

    def bad_next_state(self, state, action):
        s = orig(self, state, action)
        if action == "c" and s.contrib[0] > 0:
            return replace(s, contrib=(s.contrib[0] + 0.25, s.contrib[1]))
        return s
    monkeypatch.setattr(HoldemGame, "next_state", bad_next_state)
    rep = _run_small(g)
    kinds = {v["violation"] for v in rep.violations}
    assert "contrib cache != replayed contributions" in kinds


def test_fault_duplicate_card_is_caught(monkeypatch):
    g = tiny_game()
    en = ReducedDeckEnumerator(g, tiny_deck(), max_states=5000)
    orig = en.chance_outcomes

    def dup(state):
        out = orig(state)
        if state.holes is None:
            return [replace(out[0], holes=((out[0].holes[0]), (out[0].holes[0][0], out[0].holes[1][1])))] + out[1:3]
        return out
    monkeypatch.setattr(en, "chance_outcomes", dup)
    rep = en.run()
    assert any(v["violation"] == "duplicate card" for v in rep.violations)


def test_fault_fold_not_terminal_is_caught(monkeypatch):
    g = tiny_game()
    monkeypatch.setattr(HoldemGame, "is_terminal", lambda self, s: s.holes is not None and s.folded == -1
                        and len(s.board) == 5 and self._betting_closed(s))
    rep = _run_small(g, max_states=20000)
    assert any(v["violation"] == "folded state is not terminal" for v in rep.violations)


def test_fault_raise_against_all_in_is_caught(monkeypatch):
    g = tiny_game()
    orig = HoldemGame._legal_uncached

    def bad_legal(self, state):
        acts = orig(self, state)
        if "a" not in acts and "f" in acts:
            acts = acts + ["a"]      # offer a shove against an all-in opponent
        return acts
    monkeypatch.setattr(HoldemGame, "_legal_uncached", bad_legal)
    rep = _run_small(g, max_states=60000)
    kinds = {v["violation"] for v in rep.violations}
    assert kinds & {"raise offered against an all-in opponent", "raise offered without chips beyond the call"}


def test_fault_wrong_min_raise_is_caught(monkeypatch):
    # With NLHE sizing enforced, a bet below the minimum increment must not be
    # offered; break the minimum-increment rule and check the sized-bet
    # invariant flags the illegal sizing via the engine's own replay.
    g = HoldemGame(starting_stack=6.0, bet_fractions={"b10": 0.1, "b100": 1.0}, raise_cap=2, enforce_min_raise=True)
    rep_ok = ReducedDeckEnumerator(g, tiny_deck(), max_states=20000).run()
    assert rep_ok.violations == []
    monkeypatch.setattr(HoldemGame, "_min_increment", lambda self, streets: 0.0)
    # fresh instance: legal actions are memoized per game object
    g = HoldemGame(starting_stack=6.0, bet_fractions={"b10": 0.1, "b100": 1.0}, raise_cap=2, enforce_min_raise=True)
    en = ReducedDeckEnumerator(g, tiny_deck(), max_states=20000)
    orig_check = en.check

    def check_with_min_raise(s, depth):
        bad = orig_check(s, depth)
        if not g.is_terminal(s) and not g.is_chance(s):
            street_paid, total, p, _ = g._replay(s)
            owe = street_paid[1 - p] - street_paid[p]
            for a in g.legal_actions(s):
                if a not in ("f", "c", "a"):
                    add = g.raise_add(a, s.street, owe, total[0] + total[1], street_paid[p])
                    if add - owe < holdem_mod.BIG_BLIND - 1e-9:
                        bad.append("bet below NLHE minimum offered")
        return bad
    en.check = check_with_min_raise
    rep = en.run()
    assert any(v["violation"] == "bet below NLHE minimum offered" for v in rep.violations)
