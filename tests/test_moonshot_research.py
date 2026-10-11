"""Moonshot research layer: receipts, decision surface, artifact stats,
red team, agents/tournaments, auction game, LLM benchmark scaffold."""

from __future__ import annotations

import json
import random

import pytest

from poker_alpha.decision import DecisionConfig
from poker_alpha.demo_spots import DEMO_SPOTS
from poker_alpha.pipeline import analyze, load_solver, observe_manual
from poker_alpha.solver_config import RELEASE_STRATEGY

CFG = DecisionConfig(equity_simulations=120, rollout_simulations=120, seed=5)


@pytest.fixture(scope="module")
def solver():
    return load_solver(RELEASE_STRATEGY)


# -- decision receipts --------------------------------------------------------

def test_receipt_deterministic_and_verifiable(solver):
    from poker_alpha.analysis.receipt import build_receipt, verify_receipt

    state = DEMO_SPOTS["River, checked to hero — solver accepted"]
    rep1 = analyze(observe_manual(state), config=CFG, solver=solver)
    rep2 = analyze(observe_manual(state), config=CFG, solver=solver)
    r1 = build_receipt(rep1, state, CFG)
    r2 = build_receipt(rep2, state, CFG)
    assert r1["receipt_id"] == r2["receipt_id"]
    assert r1["receipt_id"].startswith("PA-") and len(r1["receipt_id"]) == 11
    assert r1["format"] == "pokeralpha.receipt/v1"
    assert len(r1["strategy_artifact_sha256"]) == 64
    out = verify_receipt(r1, solver=solver)
    assert out["reproduced"], out
    r1["result"]["recommended"] = "fold"
    assert not verify_receipt(r1, solver=solver)["reproduced"]


def test_receipt_privacy(solver):
    """No absolute paths, usernames, or environment leakage."""
    from poker_alpha.analysis.receipt import build_receipt

    state = DEMO_SPOTS["Preflop vs 3-bet — rollout fallback"]
    rep = analyze(observe_manual(state), config=CFG, solver=solver)
    blob = json.dumps(build_receipt(rep, state, CFG))
    assert "/Users/" not in blob and "home/" not in blob
    assert "codychen" not in blob


# -- decision surface ----------------------------------------------------------

def test_surface_cells_are_validated_states(solver):
    from poker_alpha.analysis.surface import generate_surface, surface_to_dict

    base = DEMO_SPOTS["River, facing a bet — solver at low confidence "
                      "(off-tree size)"]
    cells = generate_surface(base, "villain_bet", [1.65, 3.75, -2],
                             "effective_stack", [50, 100],
                             config=CFG, solver=solver)
    assert len(cells) == 2 and len(cells[0]) == 3
    flat = [c for row in cells for c in row]
    invalid = [c for c in flat if not c.valid]
    assert len(invalid) == 2                       # bet -2 at each stack
    assert all("negative" in c.reason for c in invalid)
    for c in flat:
        if c.valid:
            assert c.solver_state in ("ACCEPTED", "LOW CONFIDENCE",
                                      "REJECTED", "OFF")
            assert abs(sum(c.mix.values()) - 1) < 0.05 or c.mix
    d = surface_to_dict(cells, base, "villain_bet", "effective_stack")
    assert "no interpolation" in d["semantics"]
    with pytest.raises(ValueError):
        generate_surface(base, "pot", [1], "pot", [1], config=CFG)


# -- artifact stats -------------------------------------------------------------

def test_artifact_stats_consistency():
    import numpy as np

    from poker_alpha.analysis.artifact_stats import (atlas, complexity_stats,
                                                     entropy_bits, load_table,
                                                     policy_vectors,
                                                     sample_indices)

    t = load_table(RELEASE_STRATEGY)
    assert len(t.keys) == len(t.visits)
    # policies normalize
    idx = sample_indices(t, 200, seed=3)
    vecs = policy_vectors(t, idx)
    assert np.allclose(vecs.sum(axis=1), 1.0, atol=1e-5)
    ent = entropy_bits(vecs)
    assert (ent >= -1e-9).all() and np.isfinite(ent).all()
    # deterministic sampling
    assert (sample_indices(t, 200, seed=3) == idx).all()
    a1 = atlas(t, n=400, seed=9)
    a2 = atlas(t, n=400, seed=9)
    assert a1["x"] == a2["x"] and a1["top"] == a2["top"]
    c = complexity_stats(t)
    tot = (c["overall"]["near_pure_pct"] + c["overall"]["moderately_mixed_pct"]
           + c["overall"]["highly_mixed_pct"])
    assert abs(tot - 100) < 0.1
    assert "not encountered-hand frequency" in c["note"]


def test_microscope_real_series():
    from poker_alpha.analysis.artifact_stats import (load_confidence_stats,
                                                     load_table, microscope,
                                                     microscope_keys)

    t = load_table(RELEASE_STRATEGY)
    picks = microscope_keys(t, load_confidence_stats(), per_bucket=2)
    assert {p["bucket"] for p in picks} == {"low", "medium", "high"}
    m = microscope([picks[0]["key"]])
    series = m["series"][picks[0]["key"]]
    assert len(series) == 9                       # 3 milestones x 3 seeds
    for v in series.values():
        assert abs(sum(v["mix"].values()) - 1) < 1e-3


# -- red team -------------------------------------------------------------------

def test_redteam_search_reproducible(solver):
    from poker_alpha.analysis.redteam import run_search, sample_states

    s1 = sample_states(10, seed=4)
    s2 = sample_states(10, seed=4)
    assert s1 == s2                                # deterministic
    rep = run_search(n_states=12, seed=4, solver=solver,
                     config=CFG)
    assert rep["n_states_scored"] >= 10
    assert "DISCOVERY" in rep["discipline"]
    for lb in rep["leaderboards"].values():
        for s in lb:
            assert s["gate"] in ("ACCEPTED", "LOW CONFIDENCE", "REJECTED",
                                 "OFF")
    for c in rep["clusters"]:
        assert c["count"] >= 3
        assert "not a causal diagnosis" in c["note"]


# -- agents / tournament ----------------------------------------------------------

def test_tournament_seat_swapped_and_reproducible():
    from poker_alpha.agents import (RandomAgent, ScriptedKuhnAgent,
                                    kuhn_policy_agent, round_robin)
    from poker_alpha.games.kuhn import KuhnPoker

    game = KuhnPoker()
    agents = {"cfr": kuhn_policy_agent(5000), "rand": RandomAgent(),
              "tight": ScriptedKuhnAgent("tight")}
    r1 = round_robin(game, agents, hands=800, seed=11)
    r2 = round_robin(game, agents, hands=800, seed=11)
    assert r1["matrix"] == r2["matrix"]
    m = r1["matrix"]
    assert m["cfr"]["rand"]["mean"] == -m["rand"]["cfr"]["mean"]
    # trained policy beats random significantly
    lo, _hi = m["cfr"]["rand"]["ci95"]
    assert lo > 0


def test_best_response_edge_matches_exploitability():
    from poker_alpha.agents import (BestResponseAgent, kuhn_policy_agent,
                                    play_match)
    from poker_alpha.games.kuhn import KuhnPoker
    from poker_alpha.solvers.evaluation import exploitability

    game = KuhnPoker()
    cfr = kuhn_policy_agent(20000)
    expl = exploitability(game, cfr.strategy)
    br = BestResponseAgent(game, cfr.strategy)
    r = play_match(game, br, cfr, hands=20000, seed=3)
    # the oracle's seat-averaged edge ~ exploitability/2 (per-seat mean);
    # allow generous CI slack
    assert r.mean_utility_a < expl + 4 * r.se + 0.01
    assert r.mean_utility_a > -4 * r.se - 1e-6


def test_illegal_agent_action_rejected():
    from poker_alpha.agents.tournament import play_episode
    from poker_alpha.games.kuhn import KuhnPoker

    class Cheater:
        def choose_action(self, obs, legal, rng):
            return "raise_all_in"

        def reset(self):
            pass

        def metadata(self):
            return {}

    with pytest.raises(ValueError):
        play_episode(KuhnPoker(), (Cheater(), Cheater()), random.Random(0))


# -- auction duel ------------------------------------------------------------------

def test_auction_duel_is_solvable_and_sane():
    from poker_alpha.games.auction import AuctionDuel
    from poker_alpha.solvers.cfr import CFRSolver
    from poker_alpha.solvers.evaluation import exploitability

    g = AuctionDuel(n_values=3, n_bids=3)
    s = CFRSolver(g)
    s.train(3000)
    strat = s.average_strategy()
    assert exploitability(g, strat) < 0.01
    # symmetric game: both players' strategies match by value
    for v in range(3):
        for b in strat[f"A{v}"]:
            assert strat[f"A{v}"][b] == pytest.approx(
                strat[f"B{v}"][b], abs=0.05)
    # monotone bidding: higher value never bids lower (in expectation)
    def mean_bid(k):
        return sum(int(a) * p for a, p in strat[k].items())
    assert mean_bid("A0") <= mean_bid("A1") + 1e-6 <= mean_bid("A2") + 2e-6
    assert g.complexity()["infosets"] == 6
    # hidden information: player 1's infoset ignores player 0's bid
    from poker_alpha.games.auction import AuctionState

    st1 = AuctionState((1, 2), (0,))
    st2 = AuctionState((1, 2), (2,))
    assert g.infoset_key(st1) == g.infoset_key(st2)


# -- LLM benchmark scaffold -----------------------------------------------------------

def test_llm_parser_strict():
    from poker_alpha.llm_bench.parser import ParseError, parse_response

    ok = parse_response('{"action": "b", "probabilities": {"p": 0.3, "b": 0.7}}',
                        ["p", "b"])
    assert ok.action == "b" and ok.probabilities["b"] == pytest.approx(0.7)
    with pytest.raises(ParseError) as e:
        parse_response('{"action": "fold"}', ["p", "b"])
    assert e.value.category == "illegal_action"
    with pytest.raises(ParseError) as e:
        parse_response('{"action": "p", "probabilities": {"p": 0.9, "b": 0.4}}',
                       ["p", "b"])
    assert e.value.category == "bad_probabilities"
    with pytest.raises(ParseError):
        parse_response("I would bet here!", ["p", "b"])


def test_llm_harness_with_mock_provider():
    from poker_alpha.llm_bench import (MockProvider, build_dataset,
                                       run_benchmark)

    data = build_dataset()
    assert data["n_items"] == 36                   # 12 infosets x 3 templates
    always_bet = json.dumps({"action": "b",
                             "probabilities": {"p": 0.0, "b": 1.0}})
    res = run_benchmark(MockProvider({}, default=always_bet), data)
    assert res["illegal_action_rate"] == 0.0
    pol = res["policies_by_template"]["plain"]
    # always-bet is maximally deterministic and clearly exploitable
    assert pol["deterministic_infosets"] == 12
    assert pol["exploitability_chips_per_hand"] > 0.3
    # identical answers across templates -> zero paraphrase instability
    for pair in res["paraphrase_instability"]["pairs"]:
        assert pair["mean_tv"] == 0.0


def test_llm_provider_env_gating(monkeypatch):
    from poker_alpha.llm_bench.adapter import provider_from_env

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        provider_from_env("openai")
    assert "BLOCKED on credentials" in str(e.value)
