#!/usr/bin/env python
"""Generate every deterministic website research artifact from the REAL
engine and committed results.

    .venv/bin/python scripts/build_web_research_data.py [--fast]

All policy/analysis numbers on the website come from here (React only
renders). Every file is wrapped in a pokeralpha.web/v1 envelope with the
generating commit, strategy SHA, seed and args. Running twice produces
identical payloads (generation timestamp excluded from determinism).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "apps" / "web" / "src" / "data" / "lab"

from poker_alpha.decision import DecisionConfig                     # noqa: E402
from poker_alpha.pipeline import analyze, load_solver, observe_manual  # noqa: E402
from poker_alpha.solver_config import RELEASE_STRATEGY               # noqa: E402

SEED = 7


def commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def strategy_sha() -> str:
    return hashlib.sha256((ROOT / RELEASE_STRATEGY).read_bytes()).hexdigest()


def envelope(data_type: str, payload, args: dict) -> dict:
    return {
        "format": "pokeralpha.web/v1",
        "data_type": data_type,
        "generated_from_commit": commit(),
        "strategy_sha256": strategy_sha(),
        "seed": SEED,
        "generation_args": args,
        "provenance": "PRECOMPUTED output of the real PokerAlpha engine; "
                      "regenerate with scripts/build_web_research_data.py",
        "payload": payload,
    }


def write(name: str, data_type: str, payload, args: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(envelope(data_type, payload, args),
                            separators=(",", ":"), allow_nan=False))
    print(f"  {name:<28} {p.stat().st_size // 1024:>5} KB")


# ---------------------------------------------------------------------------

def build_kuhn_reference() -> None:
    """Vanilla-CFR learning trace + final policy (browser-validation
    fixture and Arena opponent)."""
    from poker_alpha.games.kuhn import KuhnPoker
    from poker_alpha.solvers.cfr import CFRSolver
    from poker_alpha.solvers.evaluation import best_response, exploitability

    game = KuhnPoker()
    solver = CFRSolver(game)
    checkpoints = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                   10000]
    trace = []
    done = 0
    for cp in checkpoints:
        solver.train(cp - done)
        done = cp
        strat = solver.average_strategy()
        trace.append({
            "iteration": cp,
            "exploitability": round(exploitability(game, strat), 8),
            "strategy": {k: {a: round(p, 6) for a, p in v.items()}
                         for k, v in sorted(strat.items())},
        })
    final = solver.average_strategy()
    br0 = best_response(game, final, 0)
    br1 = best_response(game, final, 1)
    payload = {
        "game": "kuhn",
        "algorithm": "vanilla CFR (deterministic full-tree traversal, "
                     "simultaneous updates, regret matching)",
        "game_value_exact": -1.0 / 18.0,
        "trace": trace,
        "final_policy": trace[-1]["strategy"],
        "final_exploitability": trace[-1]["exploitability"],
        "best_response_values": {"p0": round(br0[0], 8),
                                 "p1": round(br1[0], 8)},
    }
    write("kuhn_reference.json", "kuhn_cfr_reference", payload,
          {"iterations": checkpoints[-1]})


def build_surface(solver, fast: bool) -> None:
    from poker_alpha.analysis.surface import generate_surface, surface_to_dict
    from poker_alpha.demo_spots import DEMO_SPOTS

    base = DEMO_SPOTS["River, facing a bet — solver at low confidence "
                      "(off-tree size)"]
    xs = [1.0, 1.65, 2.5, 3.75, 5.0, 7.5, 12.0, 25.0, 60.0]
    ys = [10, 20, 35, 50, 75, 100, 150, 200, 300]
    cfg = DecisionConfig(equity_simulations=150 if fast else 400,
                         rollout_simulations=150 if fast else 400, seed=SEED)
    cells = generate_surface(base, "villain_bet", xs, "effective_stack", ys,
                             config=cfg, solver=solver)
    write("surface.json", "decision_surface",
          surface_to_dict(cells, base, "villain_bet", "effective_stack"),
          {"grid": f"{len(xs)}x{len(ys)}",
           "equity_simulations": cfg.equity_simulations,
           "rollout_simulations": cfg.rollout_simulations})


def build_trust_map(solver) -> None:
    from poker_alpha.analysis.explorer import (PREFLOP_NODES, matrix_stats,
                                               preflop_matrix)

    nodes = {}
    for node in PREFLOP_NODES:
        m = preflop_matrix(solver, node)
        nodes[node] = {
            "stats": matrix_stats(m),
            "cells": [[{
                "hand": c.hand, "status": c.status, "dominant": c.dominant,
                "freq": round(c.dominant_freq, 4) if c.dominant_freq else None,
                "mix": [[a, round(p, 4)] for a, p in c.mix],
                "visits": c.visits,
                "signals": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in (c.signals or {}).items()},
                "reasons": list(c.reasons)} for c in row]
                for row in m],
        }
    write("trust_map.json", "trust_map",
          {"nodes": nodes,
           "note": "gate state is an evidence gate, not a probability of "
                   "correctness; sampling is the 169 canonical classes at "
                   "fixed nodes, not natural hand frequency"},
          {"nodes": list(PREFLOP_NODES)})


def build_microscope() -> None:
    from poker_alpha.analysis.artifact_stats import (load_confidence_stats,
                                                     load_table, microscope,
                                                     microscope_keys)

    t = load_table(RELEASE_STRATEGY)
    conf = load_confidence_stats()
    picks = microscope_keys(t, conf)
    m = microscope([p["key"] for p in picks])
    write("microscope.json", "solver_microscope",
          {"picks": picks, **m,
           "selection": "deterministic: visits >= 200, entropy >= 0.3 bits, "
                        "top-3000 by visits, then low/median/high cross-seed "
                        "disagreement buckets (4 each)"},
          {"per_bucket": 4})


def build_atlas() -> None:
    import numpy as np

    from poker_alpha.analysis.artifact_stats import (atlas,
                                                     complexity_stats,
                                                     load_confidence_stats,
                                                     load_table)
    from poker_alpha.decision.solver_gate import GateThresholds, KeyStats, gate

    t = load_table(RELEASE_STRATEGY)
    conf = load_confidence_stats()
    a = atlas(t, n=6000, seed=SEED)
    # table-level gate status per sampled key (default thresholds; the
    # live lookup additionally caps flop/turn at LOW CONFIDENCE)
    th = GateThresholds()
    idx = None
    from poker_alpha.analysis.artifact_stats import sample_indices

    idx = sample_indices(t, 6000, SEED)
    gates = []
    for i in idx:
        st = conf.get(str(t.keys[i]))
        if st is None:
            gates.append("NO_DATA")
            continue
        d = gate(KeyStats(float(st[0]), float(st[1]), float(st[2]),
                          float(st[3])), th)
        s = {"SOLVER_ACCEPT": "ACCEPT", "SOLVER_LOW_CONFIDENCE": "LOW",
             "SOLVER_REJECT": "REJECT"}[d.status]
        if s == "ACCEPT" and a["street"][len(gates)] in (1, 2):
            s = "LOW (street cap)"
        gates.append(s)
    a["gate"] = gates
    write("atlas.json", "policy_atlas", a, {"n": 6000})
    write("complexity.json", "policy_complexity", complexity_stats(t), {})


def build_redteam(solver, fast: bool) -> None:
    from poker_alpha.analysis.redteam import (research_backlog, run_search,
                                              save_report)

    n = 120 if fast else 400
    rep = run_search(n_states=n, seed=SEED, solver=solver)
    rep["backlog"] = research_backlog(rep)
    save_report(rep, ROOT / "results" / "research" /
                f"redteam_seed{SEED}.json")
    # trim state payloads for the web (keep the top few per board)
    for name, lst in rep["leaderboards"].items():
        rep["leaderboards"][name] = lst[:6]
    write("redteam.json", "solver_redteam", rep, {"n_states": n})


def build_tournament() -> None:
    from poker_alpha.agents import (BestResponseAgent, RandomAgent,
                                    ScriptedKuhnAgent, kuhn_policy_agent,
                                    round_robin)
    from poker_alpha.games.kuhn import KuhnPoker
    from poker_alpha.solvers.evaluation import exploitability

    game = KuhnPoker()
    cfr = kuhn_policy_agent(20000)
    agents = {
        "CFR (20k iterations)": cfr,
        "Best response to CFR (oracle)": BestResponseAgent(game, cfr.strategy),
        "Random": RandomAgent(),
        "Tight (scripted)": ScriptedKuhnAgent("tight"),
        "Aggressive (scripted)": ScriptedKuhnAgent("aggressive"),
    }
    res = round_robin(game, agents, hands=4000, seed=SEED)
    res["cfr_exploitability"] = round(exploitability(game, cfr.strategy), 6)
    write("tournament.json", "agent_tournament", res, {"hands": 4000})


def build_games() -> None:
    from poker_alpha.games.auction import AuctionDuel
    from poker_alpha.games.kuhn import KuhnPoker
    from poker_alpha.games.leduc import LeducPoker
    from poker_alpha.solvers.cfr import CFRSolver
    from poker_alpha.solvers.evaluation import exploitability

    def enumerate_complexity(game):
        states = infosets = terminals = chance = 0
        seen = set()

        def walk(s):
            nonlocal states, terminals, chance
            states += 1
            if game.is_terminal(s):
                terminals += 1
                return
            if game.is_chance(s):
                chance += 1
                for _, nxt in game.chance_outcomes(s):
                    walk(nxt)
                return
            seen.add(game.infoset_key(s))
            for a in game.legal_actions(s):
                walk(game.next_state(s, a))

        walk(game.root())
        return {"states": states, "infosets": len(seen),
                "terminal_states": terminals, "chance_nodes": chance}

    games = {}
    for name, game, iters in (("kuhn", KuhnPoker(), 10000),
                              ("leduc", LeducPoker(), 1500),
                              ("auction_duel", AuctionDuel(), 5000)):
        t0 = time.time()
        solver = CFRSolver(game)
        solver.train(iters)
        strat = solver.average_strategy()
        entry = {
            "complexity": enumerate_complexity(game),
            "cfr_iterations": iters,
            "exploitability": round(exploitability(game, strat), 6),
            "train_seconds": round(time.time() - t0, 2),
        }
        if name == "auction_duel":
            entry["strategy"] = {k: {a: round(p, 4) for a, p in v.items()
                                     if p > 0.005}
                                 for k, v in sorted(strat.items())}
            entry["semantics"] = ("zero-sum sealed-bid duel: payoff = own "
                                  "surplus minus opponent surplus; NOT the "
                                  "classic general-sum first-price auction")
        games[name] = entry
    write("games.json", "game_lab", games, {})


def build_one_hand(solver, fast: bool) -> None:
    from poker_alpha.analysis.decision_path import decision_path
    from poker_alpha.analysis.ranges_view import range_stages
    from poker_alpha.analysis.receipt import build_receipt
    from poker_alpha.demo_spots import DEMO_SPOTS

    name = "River, checked to hero — solver accepted"
    state = DEMO_SPOTS[name]
    cfg = DecisionConfig(equity_simulations=1000 if not fast else 300,
                         rollout_simulations=800 if not fast else 300,
                         seed=SEED)
    obs = observe_manual(state)
    rep = analyze(obs, config=cfg, solver=solver)
    info = rep.details.get("solver", {})
    stages = range_stages(obs.state, seat=1)
    payload = {
        "name": name,
        "state": state,
        "mix": {k: round(v, 4) for k, v in rep.recommended_mix.items()},
        "recommended": rep.recommended,
        "method": rep.method,
        "gate": {"used": info.get("used"),
                 "confidence": info.get("confidence"),
                 "reasons": list(info.get("reasons", ()))},
        "equity": rep.hero_equity, "equity_se": rep.hero_equity_se,
        "infoset": rep.details.get("infoset"),
        "visits": rep.details.get("visits"),
        "decision_path": [{"label": r.label, "value": r.value,
                           "note": r.note}
                          for r in decision_path(rep, obs.state)],
        "range_stages": [{
            "name": s.name, "entropy_bits": round(s.entropy_bits, 2),
            "effective_combos": round(s.effective_combos, 0),
            "matrix": [[round(x, 5) for x in row] for row in s.matrix]}
            for s in stages],
        "receipt": build_receipt(rep, state, cfg),
    }
    write("one_hand.json", "one_hand_walkthrough", payload,
          {"spot": name, "equity_simulations": cfg.equity_simulations})


def build_break_it(solver, fast: bool) -> None:
    from poker_alpha.demo_spots import DEMO_SPOTS

    cfg = DecisionConfig(equity_simulations=300, rollout_simulations=300,
                         seed=SEED)
    scenarios = []

    def run(title, state, expect):
        try:
            rep = analyze(observe_manual(state), config=cfg, solver=solver)
        except (ValueError, KeyError, TypeError) as exc:
            scenarios.append({"title": title, "expected": expect,
                              "outcome": "STATE REJECTED BEFORE ANALYSIS",
                              "detail": str(exc)})
            return
        info = rep.details.get("solver", {})
        from poker_alpha.analysis.alignment import solver_state

        scenarios.append({
            "title": title, "expected": expect,
            "outcome": {"gate": solver_state(rep), "method": rep.method,
                        "reasons": list(info.get("reasons", ())),
                        "refusal": list(rep.details.get("refusal_codes", ())),
                        "recommended": rep.recommended},
        })

    run("Gate rejection: independently trained seeds disagree",
        DEMO_SPOTS["Preflop open decision — gate rejects (seed disagreement)"],
        "solver withheld; rollout fallback with its own uncertainty")
    run("Off-tree bet size (between trained sizes)",
        DEMO_SPOTS["River, facing a bet — solver at low confidence (off-tree size)"],
        "lookup translated + downgraded to LOW CONFIDENCE")
    run("Known abstraction-risk street (flop)",
        {**DEMO_SPOTS["River, checked to hero — solver accepted"],
         "board": "Qs 7d 2c", "pot": 5.0, "actions":
         DEMO_SPOTS["River, checked to hero — solver accepted"]["actions"][:3]},
        "street abstraction error caps confidence at LOW")
    run("Unsupported state: multiway pot",
        DEMO_SPOTS["6-max flop c-bet — multiway (rollout only)"],
        "outside the trained abstraction; rollout only")
    run("Invalid state: not the hero's turn",
        {**DEMO_SPOTS["River, checked to hero — solver accepted"],
         "actor": 1},
        "refusal — no recommendation at all")
    bad = dict(DEMO_SPOTS["River, checked to hero — solver accepted"])
    bad = {**bad, "hero_cards": "Qs Qs"}
    run("Invalid state: duplicate card", bad,
        "validation error — never analysed")
    write("break_it.json", "break_it", {"scenarios": scenarios}, {})


def build_claims() -> None:
    bench = json.load(open(ROOT / "results/benchmarks/native_mccfr_v1.json"))
    calib = json.load(open(ROOT /
                           "results/validation/confidence_signal_quality.json"))
    # Coverage ratio of the PROMOTED v2 rule (sd<0.2 & recent mv<0.1 &
    # visits>=20 — docs/solver_confidence_v2.md) vs the v1 ACCEPT share.
    # Deliberately NOT a best-of-grid pick: selecting the best candidate
    # post hoc on the same data would overstate the gain.
    cov = None
    try:
        v1 = calib["v1_baseline"]["SOLVER_ACCEPT"]
        v2 = calib["v2_candidates"]["sd<0.2|mv<0.1|v>=20"]
        cov = round(v2["coverage"] / v1["share"], 2)
    except (KeyError, TypeError, ZeroDivisionError):
        cov = None
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    n_tests = 0
    for line in reversed(collected):
        if "test" in line and ("collected" in line or "tests" in line):
            import re

            m = re.search(r"(\d+)\s+tests?\b", line)
            if m:
                n_tests = int(m.group(1))
                break
    claims = {
        "format": "pokeralpha.claims/v1",
        "claims": {
            "native_speedup": {
                "value": bench["speedup_warm"], "units": "x",
                "source": "results/benchmarks/native_mccfr_v1.json",
                "script": "experiments (native benchmark)",
                "qualifier": "warm median of 3 trials, "
                             f"{bench['native_warm_it_per_s_median']:.0f} vs "
                             f"{bench['python_warm_it_per_s_median']:.2f} it/s",
            },
            "native_it_per_s": {"value": bench["native_warm_it_per_s_median"],
                                "units": "it/s",
                                "source": "results/benchmarks/native_mccfr_v1.json"},
            "python_it_per_s": {"value": bench["python_warm_it_per_s_median"],
                                "units": "it/s",
                                "source": "results/benchmarks/native_mccfr_v1.json"},
            "release_iterations": {"value": 2_000_000, "units": "iterations",
                                   "source": "poker_alpha/solver_config.py"},
            "training_seeds": {"value": 3, "units": "independent seeds",
                               "source": "results/strategy/candidates/"},
            "evaluator_hands": {"value": 2_598_960, "units": "5-card hands",
                                "source": "native evaluator parity tests"},
            "calibration_rows": {"value": calib["rows"], "units": "infosets",
                                 "source": "results/validation/"
                                           "confidence_signal_quality.json"},
            "coverage_gain": {"value": cov, "units": "x",
                              "source": "results/validation/"
                                        "confidence_signal_quality.json",
                              "qualifier": "promoted v2 rule "
                                           "(sd<0.2, recent mv<0.1, "
                                           "visits>=20) coverage vs the v1 "
                                           "ACCEPT share, held-out games; "
                                           "accepted wrong-rate 3.0% vs "
                                           "3.3%"},
            "test_count": {"value": n_tests, "units": "tests",
                           "source": "pytest --collect-only -q (this commit)"},
            "stored_infosets": {"value": 144025, "units": "infosets",
                                "source": RELEASE_STRATEGY},
        },
    }
    (ROOT / "results" / "claims.json").write_text(json.dumps(claims, indent=1))
    write("claims.json", "claims_registry", claims, {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="smaller grids/samples (CI smoke)")
    args = ap.parse_args()
    t0 = time.time()
    print("building web research data…")
    solver = load_solver(str(ROOT / RELEASE_STRATEGY))
    build_kuhn_reference()
    build_surface(solver, args.fast)
    build_trust_map(solver)
    build_microscope()
    build_atlas()
    build_redteam(solver, args.fast)
    build_tournament()
    build_games()
    build_one_hand(solver, args.fast)
    build_break_it(solver, args.fast)
    build_claims()
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
