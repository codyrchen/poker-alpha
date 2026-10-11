"""Phase II Program 9: exhaustive reachable-state enumeration of reduced
Hold'em configurations with full invariant checking and graph statistics.

    python experiments/phase2_exhaustive_correctness.py [--quick]

Writes results/phase2/exhaustive_correctness.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.games.holdem import HoldemGame  # noqa: E402
from poker_alpha.poker.cards import codes  # noqa: E402
from poker_alpha.research.exhaustive import (ReducedDeckEnumerator, permute_suits,  # noqa: E402
                                             replay_from_root, sample_states, state_from_dict,
                                             state_to_dict, swap_hole_order)
from poker_alpha.solver_config import make_encoder  # noqa: E402

OUT = ROOT / "results" / "phase2" / "exhaustive_correctness.json"

CONFIGS = [
    {"name": "akq_3suits_stack4_b100_cap1", "ranks": "AKQ", "suits": "shd", "stack": 4.0,
     "bets": {"b100": 1.0}, "raise_cap": 1, "min_raise": False},
    {"name": "akq_3suits_stack6_b50_b100_cap2_minraise", "ranks": "AKQ", "suits": "shd", "stack": 6.0,
     "bets": {"b50": 0.5, "b100": 1.0}, "raise_cap": 2, "min_raise": True},
    {"name": "akqj_2suits_plus_As_stack3_b100_cap1", "ranks": "AKQJ", "suits": "sh", "extra": ["Ad"], "stack": 3.0,
     "bets": {"b100": 1.0}, "raise_cap": 1, "min_raise": False},
]


def build(cfg):
    game = HoldemGame(starting_stack=cfg["stack"], bet_fractions=dict(cfg["bets"]), raise_cap=cfg["raise_cap"],
                      enforce_min_raise=cfg.get("min_raise", False))
    deck = codes([r + s for r in cfg["ranks"] for s in cfg["suits"]] + cfg.get("extra", []))
    return game, deck


def metamorphic(game, enumerator, n=3000, seed=0):
    """Suit permutation, hole-card order, serialization and replay checks on
    sampled reachable states."""
    states = sample_states(enumerator, n, seed)
    enc = make_encoder("compact")
    perms = [(1, 0, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1)]
    out = {"sampled": len(states), "suit_permutation_failures": 0, "hole_order_failures": 0,
           "serialization_failures": 0, "replay_failures": 0, "examples": []}
    for s in states:
        for perm in perms:
            t = permute_suits(s, perm)
            ok = (game.is_terminal(t) == game.is_terminal(s)
                  and (game.is_terminal(s) or game.legal_actions(t) == game.legal_actions(s))
                  and (not game.is_terminal(s) or abs(game.utility(t) - game.utility(s)) < 1e-12)
                  and (game.is_terminal(s) or game.is_chance(s) or enc.encode(game, t) == enc.encode(game, s)))
            if not ok:
                out["suit_permutation_failures"] += 1
                if len(out["examples"]) < 5:
                    out["examples"].append({"kind": "suit", "state": state_to_dict(s), "perm": perm})
        t = swap_hole_order(s)
        if game.is_terminal(s):
            ok = abs(game.utility(t) - game.utility(s)) < 1e-12
        else:
            ok = game.is_chance(t) == game.is_chance(s) and (game.is_chance(s) or game.legal_actions(t) == game.legal_actions(s))
        if not ok:
            out["hole_order_failures"] += 1
        if state_from_dict(json.loads(json.dumps(state_to_dict(s)))) != s:
            out["serialization_failures"] += 1
        r = replay_from_root(game, s)
        if (r.streets != s.streets or any(abs(a - b) > 1e-9 for a, b in zip(r.contrib, s.contrib))
                or r.folded != s.folded or r.all_in != s.all_in or r.board != s.board):
            out["replay_failures"] += 1
            if len(out["examples"]) < 5:
                out["examples"].append({"kind": "replay", "state": state_to_dict(s), "replayed": state_to_dict(r)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    doc = {"format": "pokeralpha.phase2.exhaustive_correctness/v1", "configs": [],
           "invariants": ["contrib cache == replayed contributions", "0 <= contrib <= stack", "pot == sum(contrib)",
                          "board size matches street", "no duplicate cards", "valid current player",
                          "legal actions non-empty; 'c' always; 'f' iff facing", "no raise vs all-in opponent",
                          "no raise without chips beyond the call", "matched all-in closes betting",
                          "sized bets strictly below stack", "folded => terminal", "showdown only on the river",
                          "finite utility bounded by contributions", "depth bound (no cycles)"]}
    for cfg in CONFIGS[: 1 if a.quick else None]:
        game, deck = build(cfg)
        en = ReducedDeckEnumerator(game, deck, max_states=200000 if a.quick else None)
        rep = en.run()
        comp = ReducedDeckEnumerator(game, deck, max_states=200000 if a.quick else None, encoder=make_encoder("compact")).run()
        d = rep.as_dict()
        d["compact_encoder_infosets"] = comp.infosets
        d["raw_to_compact_ratio"] = (rep.infosets / comp.infosets) if comp.infosets else None
        d["metamorphic"] = metamorphic(game, en, n=800 if a.quick else 3000)
        d["config"] = cfg
        d["deck_size"] = len(deck)
        d["game_signature"] = game.signature()
        doc["configs"].append(d)
        print(cfg["name"], {k: d[k] for k in ("states", "decision_nodes", "terminal_nodes", "infosets",
                                             "compact_encoder_infosets", "max_depth", "violation_count")},
              "metamorphic", {k: v for k, v in d["metamorphic"].items() if k != "examples"}, flush=True)
    doc["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    doc["machine"] = {"platform": platform.platform(), "python": platform.python_version()}
    out = OUT if not a.quick else OUT.with_name("exhaustive_correctness_quick.json")
    out.write_text(json.dumps(doc, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
