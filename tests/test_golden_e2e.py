"""End-to-end golden tests (Phase 69).

Fixed inputs run through the whole pipeline (observe -> validate -> analyse)
and the decision-relevant output is compared with ``tests/golden/e2e_v1.json``.
Any change in recognized state, refusal codes, method, recommendation,
candidates, EVs (4 dp) or equity shows up here as a reviewable diff.

Regenerate after an INTENDED change only:

    POKERALPHA_UPDATE_GOLDEN=1 pytest tests/test_golden_e2e.py

The PokerNow case uses the single real TUNING frame: it guards against
regressions, it is not evidence of recognition accuracy.
"""

import json
import os
from pathlib import Path

import pytest

from poker_alpha.decision import DecisionConfig
from poker_alpha.pipeline import analyze, load_solver, observe_manual

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "e2e_v1.json"
UPDATE = os.environ.get("POKERALPHA_UPDATE_GOLDEN") == "1"

ROLLOUT = DecisionConfig(equity_simulations=600, rollout_simulations=200, seed=11)
HEURISTIC = DecisionConfig(equity_simulations=600, rollout_simulations=0, seed=11)


def hu(**kw):
    d = {"num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
         "big_blind": 1.0, "hero_cards": "As Ks", "board": "", "actor": 0,
         "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                   {"stack": 99.0, "bet": 1.0, "committed": 1.0}],
         "pot": 1.5, "actions": []}
    d.update(kw)
    return d


def r4(x):
    return None if x is None else round(float(x), 4)


def digest(report, state):
    return {
        "state": state.describe() if state.hero_cards else None,
        "dealer": state.dealer, "pot": r4(state.pot),
        "stacks": [r4(s.stack) for s in state.seats],
        "bets": [r4(s.current_bet) for s in state.seats],
        "method": report.method, "recommended": report.recommended,
        "confidence": report.confidence,
        "refusal_codes": report.details.get("refusal_codes", []),
        "assumptions": report.details.get("assumptions", []),
        "cascade": [(c["source"], c["status"], c.get("code"))
                    for c in report.details["source_cascade"]],
        "equity": r4(report.hero_equity),
        "candidates": [(c.label, r4(c.amount_to), r4(c.probability), r4(c.ev_bb))
                       for c in report.candidates],
    }


def manual_cases():
    flop = hu(board="Qh 7d 2c", dealer=1, actor=0, street="flop",
              seats=[{"stack": 94.0, "bet": 0.0, "committed": 3.0},
                     {"stack": 91.0, "bet": 3.0, "committed": 6.0}], pot=9.0)
    return {
        "manual_preflop_rollout": (observe_manual(hu()), ROLLOUT, None),
        "manual_flop_heuristic": (observe_manual(flop), HEURISTIC, None),
        "manual_flop_rollout": (observe_manual(flop), ROLLOUT, None),
        "manual_refusal_hero_stack": (observe_manual(hu(seats=[
            {"stack": None, "bet": 0.5, "committed": 0.5},
            {"stack": 99.0, "bet": 1.0, "committed": 1.0}])), ROLLOUT, None),
        "manual_assumption_actor": (observe_manual(hu(actor=None)), HEURISTIC, None),
    }


def run_case(obs, cfg, solver):
    return digest(analyze(obs, config=cfg, solver=solver), obs.state)


def compare(name, got):
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    got = json.loads(json.dumps(got))            # tuples -> lists
    if UPDATE:
        data[name] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return
    assert name in data, f"no golden entry {name!r}: run with POKERALPHA_UPDATE_GOLDEN=1"
    assert got == data[name]


@pytest.mark.parametrize("name", sorted(manual_cases()))
def test_manual_golden(name):
    obs, cfg, solver = manual_cases()[name]
    compare(name, run_case(obs, cfg, solver))


def test_release_solver_golden():
    from poker_alpha.solver_config import RELEASE_STRATEGY

    solver = load_solver(ROOT / RELEASE_STRATEGY)
    obs = observe_manual(hu(hero_cards="Ah Kd", dealer=0, actor=0))
    compare("release_solver_preflop", run_case(obs, HEURISTIC, solver))


def test_synthetic_screenshot_golden():
    pytest.importorskip("PIL")
    from poker_alpha.observer.pokernow import default_layout
    from poker_alpha.observer.synthetic import SyntheticSeat, SyntheticTable, render_table
    from poker_alpha.pipeline import observe_screenshot

    cal = default_layout(2, 0)
    t = SyntheticTable(seats=[SyntheticSeat("h", 97.0), SyntheticSeat("v", 94.0, bet=6.0)],
                       dealer=1, hero_cards=("Td", "Tc"), board=("9s", "8h", "2d"),
                       pot=6.0, actor=0)
    obs = observe_screenshot(render_table(t, cal), calibration=cal, seats=2)
    compare("synthetic_flop_screenshot", run_case(obs, HEURISTIC, None))


@pytest.mark.vision
def test_pokernow_tuning_frame_golden():
    pytest.importorskip("PIL")
    pytest.importorskip("scipy")
    from PIL import Image

    from poker_alpha.observer.pokernow import pokernow_hu_layout
    from poker_alpha.pipeline import observe_screenshot

    img = Image.open(ROOT / "tests/fixtures/pokernow/raw/hu_preflop_0001.png").convert("RGB")
    obs = observe_screenshot(img, calibration=pokernow_hu_layout(), small_blind=0.25,
                             big_blind=0.5)
    compare("pokernow_tuning_frame_hu_preflop_0001", run_case(obs, HEURISTIC, None))
