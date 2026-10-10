"""Phase 32C: where do preflop all-ins come from? EVs of BB responses to
small opens under the trained profile, and how the BTN responds to a jam.

Usage: python experiments/phase32_jams.py --ckpt-dir /home/user/pa_ckpt
Appends section ``jam_ev`` to results/validation/solver_quality_v1.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase32_preflop_audit import classes_grid, hole_of  # noqa: E402

from poker_alpha.games.holdem import _tokens  # noqa: E402
from poker_alpha.solver_config import PRIMARY_CONFIG  # noqa: E402
from poker_alpha.solvers.holdem_analysis import spot_state  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402

OUT = ROOT / "results" / "validation" / "solver_quality_v1.json"


def evs(game, table, hero, hero_seat, prefix, actions, deals, seed):
    """Paired MC EV (bb, for the hero) of forced actions after preflop
    ``prefix`` tokens; villain holding uniform, then both follow ``table``."""
    rng = np.random.default_rng(seed)
    live = [c for c in range(52) if c not in hero]
    vals = {a: np.zeros(deals) for a in actions}
    for d in range(deals):
        pick = rng.choice(len(live), size=7, replace=False)
        vill = (live[pick[0]], live[pick[1]])
        board = tuple(live[i] for i in pick[2:7])
        us = rng.random(64)
        holes = (hero, vill) if hero_seat == 0 else (vill, hero)
        for a in actions:
            st = replace(game.root(), holes=holes)
            for t in list(_tokens(prefix)) + [a]:
                st = game.next_state(st, t)
            k = 0
            while not game.is_terminal(st):
                if game.is_chance(st):
                    st = replace(st, board=board[:[0, 3, 4, 5][st.street + 1]], streets=st.streets + ("",))
                    continue
                legal = game.legal_actions(st)
                p = table.get(game.infoset_key(st))
                p = np.full(len(legal), 1 / len(legal)) if p is None else p
                cdf = np.cumsum(p)
                cdf /= cdf[-1]
                st = game.next_state(st, legal[int(np.searchsorted(cdf, us[k % 64], side="right"))])
                k += 1
            u = game.utility(st)
            vals[a][d] = u if hero_seat == 0 else -u
    base = actions[0]
    return {a: {"ev_bb": round(float(vals[a].mean()), 3),
                f"minus_{base}": round(float((vals[a] - vals[base]).mean()), 3),
                "paired_se": round(float((vals[a] - vals[base]).std(ddof=1) / np.sqrt(deals)), 3)}
            for a in actions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--deals", type=int, default=8000)
    a = p.parse_args()
    game = PRIMARY_CONFIG.build_game()
    solver = load_checkpoint(a.ckpt_dir / "locked_seed0_it300000.npz", game)
    table = {k: n.average_strategy() for k, n in solver.infosets.items() if n.strategy_sum.sum() > 0}
    # BTN response to a jam after each open: fold frequency by class
    classes = [c for r in classes_grid() for c in r]
    resp = {}
    for open_tok in ("b33", "b75", "b150"):
        folds, call_cls = [], []
        for cls in classes:
            hole = hole_of(cls)
            vill = tuple(c for c in range(52) if c not in hole)[:2]
            st = spot_state(game, "BTN", hole, (), (open_tok + "a",), villain_hole=vill)
            n = solver.infosets.get(game.infoset_key(st))
            if n is None or n.strategy_sum.sum() == 0:
                continue
            avg = n.average_strategy()
            folds.append(float(avg[game.legal_actions(st).index("f")]))
            if avg[game.legal_actions(st).index("c")] > 0.5:
                call_cls.append(cls)
        resp[open_tok] = {"classes_visited": len(folds),
                          "mean_fold_to_jam_over_classes": round(float(np.mean(folds)), 4),
                          "combo_weighted_note": "class-average, not combo-weighted",
                          "classes_calling_majority": call_cls}
    del solver
    out = {"btn_response_to_bb_jam": resp, "bb_action_ev": {}}
    for name, cls, prefix in (("BB 99 vs open33", "99", "b33"), ("BB A5o vs open33", "A5o", "b33"),
                              ("BB AJo vs open75", "AJo", "b75"), ("BB 72o vs open33", "72o", "b33"),
                              ("BB AKo vs open150", "AKo", "b150")):
        hero = hole_of(cls)
        out["bb_action_ev"][name] = evs(game, table, hero, 1, prefix,
                                        ["c", "b33", "b75", "b150", "a", "f"], a.deals, 7)
        print(name, out["bb_action_ev"][name], flush=True)
    doc = json.loads(OUT.read_text())
    doc["jam_ev"] = out
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps(resp, indent=1))


if __name__ == "__main__":
    main()
