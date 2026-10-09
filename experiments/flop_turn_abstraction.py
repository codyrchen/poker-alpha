"""Phases 56-57: exact abstraction error of the solver encoders on fixed-flop
and fixed-turn subgames.

Method (same as the river study, Phase 34A, extended to earlier streets):
a subgame of the real ``HoldemGame`` rules starts at the flop (or turn)
with a fixed preflop (and flop) line and hole cards dealt from explicit
finite ranges. To stay exactly solvable in pure Python the remaining board
cards come from a FIXED SAMPLE of turn / river cards (uniform over the
sample minus dead cards) and the bet menu is reduced (one bet size + all-in,
raise cap 1). Raw and abstract keys play IDENTICAL trees, so the comparison
is exact for this game; it measures how each encoder's flop / turn / river
keys merge strategically different situations, not the full 52-card error.

Per subgame and encoder (raw = perfect information sets):
  * CFR+ solution of the encoder's abstract game, lifted to raw information
    sets and evaluated in the raw game: exact exploitability (BB),
    EV error vs the raw solution, reach-weighted L1 to the raw strategy;
  * abstract infoset count;
  * collision severity: for every abstract key, the reach-weighted mean L1
    between the raw-optimal strategies of the raw infosets merged into it and
    their reach-weighted mean (0 = the merged situations want the same play).

    python experiments/flop_turn_abstraction.py flop --iters 300
    python experiments/flop_turn_abstraction.py turn --iters 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import infoset_reach  # noqa: E402

from poker_alpha.games.base import Game  # noqa: E402
from poker_alpha.games.holdem import HoldemGame, HoldemState  # noqa: E402
from poker_alpha.poker.cards import card_str, codes  # noqa: E402
from poker_alpha.poker.evaluator import evaluate_best_codes  # noqa: E402
from poker_alpha.solver_config import make_encoder  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import expected_value, exploitability  # noqa: E402

BETS = {"b75": 0.75}
RAISE_CAP = 1
ENCODERS = ("raw", "bucket", "compact", "compact_river_pct20")
ENCODER_NOTES = {"raw": "perfect information sets (reference)",
                 "bucket": "perfect-recall card buckets (v1 'bucket' encoder)",
                 "compact": "compact imperfect-recall encoder (v1 release)",
                 "compact_river_pct20": "compact + 20 river percentile buckets (release v2)"}


class FixedRunoutSubgame(Game):
    """``HoldemGame`` rules from a fixed board prefix and line; hole cards from
    ranges; later board cards from a fixed sample."""

    def __init__(self, prefix, line, turns, rivers, r0, r1, encoder="raw"):
        enc = None if encoder == "raw" else make_encoder(encoder)
        self.inner = HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS),
                                raise_cap=RAISE_CAP, encoder=enc)
        self.prefix = tuple(codes(prefix))
        self.line = tuple(line) + ("",)
        self.turns = tuple(codes(turns))
        self.rivers = tuple(codes(rivers))
        self.encoder_name = encoder
        self._keys = {}
        deals = [(a, b) for a in r0 for b in r1 if not set(a) & set(b)
                 and not (set(a) | set(b)) & set(self.prefix)]
        self._deals = [(d, 1.0 / len(deals)) for d in deals]
        probe = HoldemState(holes=((0, 1), (2, 3)), board=self.prefix, streets=self.line,
                            contrib=(0.0, 0.0))
        _, total, _, _ = self.inner._replay(probe)
        self._contrib = tuple(total)

    def signature(self):
        return f"FixedRunoutSubgame:{self.encoder_name}"

    def root(self):
        return None

    def is_chance(self, s):
        return s is None or self.inner.is_chance(s)

    def chance_outcomes(self, s):
        if s is None:
            return [(w, HoldemState(holes=d, board=self.prefix, streets=self.line,
                                    contrib=self._contrib)) for d, w in self._deals]
        dead = set(s.board) | {c for h in s.holes for c in h}
        pool = self.turns if len(s.board) == 3 else self.rivers
        live = [c for c in pool if c not in dead]
        return [(1.0 / len(live), replace(s, board=s.board + (c,), streets=s.streets + ("",)))
                for c in live]

    def is_terminal(self, s):
        return s is not None and self.inner.is_terminal(s)

    def utility(self, s):
        return self.inner.utility(s)

    def current_player(self, s):
        return self.inner.current_player(s)

    def infoset_key(self, s):
        # Keys depend only on the (frozen, hashable) state; CFR+ revisits every
        # node each iteration, and card encoders are expensive (suit
        # canonicalization), so cache them per game.
        k = self._keys.get(s)
        if k is None:
            k = self._keys[s] = self.inner.infoset_key(s)
        return k

    def legal_actions(self, s):
        return self.inner.legal_actions(s)

    def next_state(self, s, a):
        return self.inner.next_state(s, a)


def lift_with_map(abstract_game, raw_game, abstract_strategy):
    """Abstract strategy read through raw keys + the raw -> abstract key map."""
    out, keymap = {}, {}

    def walk(s):
        if raw_game.is_terminal(s):
            return
        if raw_game.is_chance(s):
            for _, c in raw_game.chance_outcomes(s):
                walk(c)
            return
        rk = raw_game.infoset_key(s)
        if rk not in out:
            ak = abstract_game.infoset_key(s)
            keymap[rk] = ak
            acts = raw_game.legal_actions(s)
            p = abstract_strategy.get(ak)
            out[rk] = p if p is not None else {a: 1.0 / len(acts) for a in acts}
        for a in raw_game.legal_actions(s):
            walk(raw_game.next_state(s, a))
    walk(raw_game.root())
    return out, keymap


def distance(ref, strat, weights):
    tot = sum(weights.values())
    return sum(w * sum(abs(ref[k][a] - strat[k].get(a, 0.0)) for a in ref[k])
               for k, w in weights.items() if k in ref and k in strat) / tot


def collision_severity(keymap, ref, weights):
    """Per abstract key: reach-weighted mean L1 of the merged raw-optimal
    strategies to their weighted mean. Returns summary statistics."""
    groups = {}
    for rk, ak in keymap.items():
        if weights.get(rk, 0.0) > 0:
            groups.setdefault(ak, []).append(rk)
    sev, mass = [], []
    for ak, rks in groups.items():
        w = np.array([weights[k] for k in rks])
        acts = list(ref[rks[0]])
        P = np.array([[ref[k].get(a, 0.0) for a in acts] for k in rks])
        mean = (w[:, None] * P).sum(0) / w.sum()
        d = float((w * np.abs(P - mean).sum(1)).sum() / w.sum())
        sev.append(d)
        mass.append(float(w.sum()))
    sev, mass = np.array(sev), np.array(mass)
    tot = mass.sum()
    return {"abstract_keys_reached": int(len(sev)),
            "merged_keys": int(sum(len(v) > 1 for v in groups.values())),
            "reach_weighted_severity": float((sev * mass).sum() / tot) if tot else 0.0,
            "max_severity": float(sev.max()) if len(sev) else 0.0,
            "reach_share_severity_gt_0_5": float(mass[sev > 0.5].sum() / tot) if tot else 0.0}


def strength_range(board, bands, n):
    """Combos whose strength percentile on ``board`` lies in the bands;
    deterministically thinned to ``n`` (as phase33.percentile_range)."""
    b = codes(board)
    combos = [(a, c) for a in range(52) for c in range(a + 1, 52) if a not in b and c not in b]
    vals = sorted((evaluate_best_codes([a, c] + b), (a, c)) for a, c in combos)
    m = len(vals)
    chosen = [h for lo, hi in bands for i, (_, h) in enumerate(vals) if lo <= i / m < hi]
    if len(chosen) > n:
        idx = np.linspace(0, len(chosen) - 1, n).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return [tuple(sorted(h)) for h in chosen]


def runout_sample(prefix, k, seed, avoid=()):
    rng = np.random.default_rng(seed)
    dead = set(codes(prefix)) | set(codes(list(avoid)))
    live = [c for c in range(52) if c not in dead]
    return [card_str(int(c)) for c in rng.choice(live, size=k, replace=False)]


FLOPS = [  # name, flop, p0 bands (polarized), p1 bands (bluff-catchers / medium)
    ("A72 rainbow", ["As", "7d", "2c"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("K83 two-tone", ["Kh", "8h", "3c"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("987 two-tone", ["9s", "8s", "7d"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("762 rainbow", ["7c", "6d", "2h"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("paired KK5", ["Ks", "Kd", "5c"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("monotone J84", ["Jh", "8h", "4h"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("connected QJT", ["Qc", "Jd", "Th"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
]

TURNS = [  # name, flop + turn, p0 bands, p1 bands
    ("blank", ["Ks", "7d", "2c", "3h"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("overcard", ["9s", "7d", "2c", "Ah"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("paired turn", ["Ks", "7d", "2c", "7h"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("flush completes", ["Kh", "8h", "3c", "2h"], [(0.0, 0.15), (0.88, 1.0)], [(0.40, 0.85)]),
    ("straight completes", ["9s", "8d", "2c", "Tc"], [(0.0, 0.15), (0.88, 1.0)],
     [(0.40, 0.85)]),
    ("four-straight board", ["9s", "8d", "7c", "6h"], [(0.0, 0.15), (0.88, 1.0)],
     [(0.40, 0.85)]),
    ("four-flush board", ["Qh", "9h", "5h", "2h"], [(0.0, 0.15), (0.88, 1.0)],
     [(0.40, 0.85)]),
]


def run_subgame(name, prefix, line, b0, b1, n_range, n_turn, n_river, iters, encoders, seed):
    r0 = strength_range(prefix, b0, n_range)
    r1 = strength_range(prefix, b1, n_range)
    turns = runout_sample(prefix, n_turn, seed) if len(prefix) == 3 else []
    rivers = runout_sample(prefix, n_river, seed + 1, avoid=turns)
    raw = FixedRunoutSubgame(prefix, line, turns, rivers, r0, r1, "raw")
    res = {"name": name, "board": prefix, "line": list(line), "turn_sample": turns,
           "river_sample": rivers, "range_sizes": [len(r0), len(r1)],
           "ranges": [[card_str(a) + card_str(b) for a, b in r] for r in (r0, r1)],
           "pot_bb": round(sum(raw._contrib), 3), "encoders": {}}
    ref = weights = value = None
    for enc in encoders:
        g = raw if enc == "raw" else FixedRunoutSubgame(prefix, line, turns, rivers, r0, r1, enc)
        if enc != "raw" and "raw" in res["encoders"]:
            _, km = lift_with_map(g, raw, {})
            if len(set(km.values())) == len(km):
                # Same information partition as raw = the same abstract game:
                # deterministic CFR+ gives the raw solution exactly.
                row = dict(res["encoders"]["raw"], identical_partition_to_raw=True,
                           seconds=0.0)
                res["encoders"][enc] = row
                print(name, enc, "identical information partition to raw (no merges)",
                      flush=True)
                continue
        t = time.time()
        s = CFRPlusSolver(g)
        s.train(iters)
        secs = time.time() - t
        strat, keymap = lift_with_map(g, raw, s.average_strategy())
        if enc == "raw":
            ref = strat
            weights = infoset_reach(raw, ref)
            value = expected_value(raw, ref)
        ev = expected_value(raw, strat)
        row = {"abstract_infosets": len(s.infosets), "seconds": round(secs, 1),
               "exploitability_in_raw_game_bb": exploitability(raw, strat),
               "ev_p0_bb": ev, "ev_error_vs_raw_bb": abs(ev - value),
               "weighted_l1_vs_raw": distance(ref, strat, weights),
               "collisions": collision_severity(keymap, ref, weights)}
        res["encoders"][enc] = row
        print(name, enc, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in row.items() if k != "collisions"}),
              json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in row["collisions"].items()}), flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("street", choices=("flop", "turn"))
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--range", type=int, default=6)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--rivers", type=int, default=3)
    p.add_argument("--only", default=None, help="comma-separated subgame names")
    p.add_argument("--encoders", default=",".join(ENCODERS))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    games = FLOPS if a.street == "flop" else TURNS
    line = ("b75c",) if a.street == "flop" else ("b75c", "cc")
    if a.only:
        keep = set(a.only.split(","))
        games = [g for g in games if g[0] in keep]
    out = a.out or ROOT / "results" / "validation" / f"{a.street}_abstraction_v1.json"
    rows = []
    t0 = time.time()
    for i, (name, board, b0, b1) in enumerate(games):
        rows.append(run_subgame(name, board, line, b0, b1, a.range, a.turns, a.rivers,
                                a.iters, a.encoders.split(","), a.seed + 10 * i))
        out.write_text(json.dumps({
            "format": f"pokeralpha.{a.street}_abstraction/v1",
            "method": __doc__.split("\n\n")[1].replace("\n", " "),
            "settings": {"iters": a.iters, "range_combos": a.range, "turn_sample": a.turns,
                         "river_sample": a.rivers, "bets": BETS, "raise_cap": RAISE_CAP,
                         "stack_bb": 100, "line": list(line), "seed": a.seed},
            "encoders": {e: ENCODER_NOTES[e] for e in a.encoders.split(",")},
            "rows": rows, "seconds": round(time.time() - t0, 1)}, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
