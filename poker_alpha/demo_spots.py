"""Bundled demo spots — every one runs through the real pipeline.

UI-free on purpose: the Streamlit app, the local API and the desktop
overlay all serve these same states, and none of them should drag a UI
dependency in to do it. The labels state what actually happens with the
release strategy + confidence gate (verified by tests).
"""

from __future__ import annotations

def _hu(hero_cards, board, pot, actions, committed, stacks, bets=(0.0, 0.0)):
    return {
        "num_seats": 2, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": hero_cards, "board": board, "pot": pot, "actor": 0,
        "seats": [{"stack": stacks[0], "committed": committed[0], "bet": bets[0]},
                  {"stack": stacks[1], "committed": committed[1], "bet": bets[1]}],
        "actions": actions}


_PRE = [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
        {"street": 0, "seat": 1, "kind": "call", "amount": 1.5}]
_CHECKS_TO_RIVER = [{"street": 1, "seat": 1, "kind": "check"},
                    {"street": 1, "seat": 0, "kind": "check"},
                    {"street": 2, "seat": 1, "kind": "check"},
                    {"street": 2, "seat": 0, "kind": "check"}]

DEMO_SPOTS = {
    "River, checked to hero — solver accepted": _hu(
        "Ah Kd", "Qs 7d 2c Th 3h", 5.0,
        _PRE + _CHECKS_TO_RIVER + [{"street": 3, "seat": 1, "kind": "check"}],
        (2.5, 2.5), (97.5, 97.5)),
    "River, facing a bet — solver at low confidence (off-tree size)": _hu(
        "9h 9c", "Qs 7d 2c Th 3h", 7.5,
        _PRE + _CHECKS_TO_RIVER + [{"street": 3, "seat": 1, "kind": "bet",
                                    "amount": 2.5}],
        (2.5, 5.0), (97.5, 95.0), bets=(0.0, 2.5)),
    "Preflop open decision — gate rejects (seed disagreement)": _hu(
        "Ad Qs", "", 1.5, [], (0.5, 1.0), (99.5, 99.0), bets=(0.5, 1.0)),
    "Preflop vs 3-bet — rollout fallback": _hu(
        "Ts Td", "", 11.5,
        [{"street": 0, "seat": 0, "kind": "raise", "amount": 2.5},
         {"street": 0, "seat": 1, "kind": "raise", "amount": 9.0}],
        (2.5, 9.0), (97.5, 91.0), bets=(2.5, 9.0)),
    "6-max flop c-bet — multiway (rollout only)": {
        "num_seats": 6, "hero_seat": 0, "dealer": 0,
        "small_blind": 0.5, "big_blind": 1.0,
        "hero_cards": "As Ks", "board": "Qs Js 4h", "pot": 13.5, "actor": 0,
        "seats": [{"stack": 96.5}, {"stack": 99.5, "folded": True},
                  {"stack": 90.0}, {"stack": 100, "folded": True},
                  {"stack": 100, "folded": True}, {"stack": 100, "folded": True}],
        "actions": [
            {"street": 0, "seat": 3, "kind": "fold"},
            {"street": 0, "seat": 4, "kind": "fold"},
            {"street": 0, "seat": 5, "kind": "fold"},
            {"street": 0, "seat": 0, "kind": "raise", "amount": 3.5},
            {"street": 0, "seat": 1, "kind": "fold"},
            {"street": 0, "seat": 2, "kind": "call", "amount": 2.5},
            {"street": 1, "seat": 2, "kind": "check"}],
    },
}


__all__ = ["DEMO_SPOTS"]
