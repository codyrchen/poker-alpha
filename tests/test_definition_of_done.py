"""The three end-to-end flows of the project's definition of done."""

from pathlib import Path

import pytest

from poker_alpha.abstraction import ToyHoldemEncoder
from poker_alpha.decision import (DecisionConfig, SolverStrategyProvider,
                                  recommend_action)
from poker_alpha.games import HoldemGame
from poker_alpha.history import load_hands, replay_hand
from poker_alpha.holdem import ManualStateAdapter, validate
from poker_alpha.session import SessionStore, analyze_session, import_hands
from poker_alpha.solvers import MCCFRSolver
from poker_alpha.solvers.holdem_analysis import infoset_visits
from poker_alpha.solvers.serialize import load_checkpoint, save_checkpoint

FIX = Path(__file__).resolve().parent / "fixtures"
CFG = DecisionConfig(equity_simulations=300, rollout_simulations=200, seed=2)


@pytest.mark.vision
def test_flow_screenshot_to_decision_report():
    pytest.importorskip("PIL")
    from PIL import Image

    from poker_alpha.observer.calibration import TableCalibration
    from poker_alpha.observer.fusion import StateTracker
    from poker_alpha.observer.pokernow import PokerNowStyleAdapter

    cal = TableCalibration.load(FIX / "table_calibration.json")
    adapter = PokerNowStyleAdapter(cal)
    tracker = StateTracker(cal, 0.5, 1.0)
    img = Image.open(FIX / "table.png").convert("RGB")
    for _ in range(3):
        tracker.update(adapter.read_frame(img))
    obs = tracker.to_observed_state()
    assert not [i for i in validate(obs) if i.severity == "error"]
    rep = recommend_action(obs, config=CFG)
    assert rep.hero_equity is not None and rep.pot_odds is not None
    assert rep.spr is not None and rep.opponent_ranges
    assert rep.recommended and all(c.ev_bb is not None for c in rep.candidates)


def test_flow_hand_history_to_post_hand_analysis(tmp_path):
    hands = load_hands(FIX / "hands" / "sample.json")
    decision = replay_hand(hands[0]).decisions[1]
    rep = recommend_action(decision.state, config=CFG)
    assert rep.recommended is not None
    with SessionStore(tmp_path / "s.sqlite") as store:
        sid = import_hands(store, hands, config=CFG)
        analysis = analyze_session(store, sid)
    assert analysis.decisions == 5 and analysis.review


def test_flow_abstract_hu_mccfr_checkpoint_resume_lookup(tmp_path):
    game = HoldemGame(encoder=ToyHoldemEncoder())
    solver = MCCFRSolver(game, seed=0)
    solver.train(30)
    path = save_checkpoint(solver, tmp_path / "hu.npz")
    resumed = load_checkpoint(path, HoldemGame(encoder=ToyHoldemEncoder()))
    resumed.train(30)
    assert resumed.iterations == 60
    provider = SolverStrategyProvider(game, resumed.average_strategy(),
                                      infoset_visits(resumed), min_visits=1)
    obs = ManualStateAdapter.from_dict({
        "num_seats": 2, "hero_seat": 0, "dealer": 0, "small_blind": 0.5,
        "big_blind": 1.0, "hero_cards": "As Ks", "actor": 0, "pot": 1.5,
        "seats": [{"stack": 99.5, "bet": 0.5, "committed": 0.5},
                  {"stack": 99.0, "bet": 1.0, "committed": 1.0}]})
    rep = recommend_action(obs, config=DecisionConfig(equity_simulations=200,
                                                      solver=provider))
    assert rep.method == "solver"
    assert sum(rep.recommended_mix.values()) == pytest.approx(1.0)
    assert rep.details["infoset"] == "0|SB|0|"
