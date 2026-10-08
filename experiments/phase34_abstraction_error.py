"""Phase 34: abstraction error of the solver encoders on exact subgames.

34A  Identical game trees: the river decision of the real ``HoldemGame``
     (fixed board, fixed earlier line ``b75c / cc / cc`` -> pot 5 BB, 97.5 BB
     behind, bet menu 33/75/150% + all-in, raise cap 2 for tractability),
     hands dealt from explicit ranges. Each encoder (raw, bucket, transition,
     compact) defines an abstract game on that tree; CFR+ solves it exactly;
     the abstract strategy is then evaluated in the RAW game (one strategy
     per raw information set) by exact exploitability, EV and reach-weighted
     L1 distance to the raw solution. Raw = perfect information sets, so its
     exploitability is the solver error and the others' excess is
     abstraction error.

34B  Collision attribution for suspicious keys of the trained strategy
     (highest seed disagreement in the 300k canonical matrix): member states
     from a large seeded corpus with equity range, made-hand / draw / nut
     mix, texture, SPR, position, betting context and preflop lines; a
     severity score from measured dispersion.

Writes results/validation/abstraction_error_v1.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase33_reduced_games import infoset_reach, percentile_range  # noqa: E402

from poker_alpha.games.base import Game  # noqa: E402
from poker_alpha.games.holdem import HoldemGame, HoldemState  # noqa: E402
from poker_alpha.poker.cards import codes  # noqa: E402
from poker_alpha.solver_config import make_encoder  # noqa: E402
from poker_alpha.solvers.cfr_plus import CFRPlusSolver  # noqa: E402
from poker_alpha.solvers.evaluation import expected_value, exploitability  # noqa: E402

OUT = ROOT / "results" / "validation" / "abstraction_error_v1.json"
LINE = ("b75c", "cc", "cc", "")
BETS = {"b33": 0.33, "b75": 0.75, "b150": 1.5}


class HoldemRiverSubgame(Game):
    """River decision of HoldemGame with dealt ranges; keys from ``encoder``
    (``None`` = raw perfect information sets)."""

    def __init__(self, board, r0, r1, encoder_name="raw"):
        self.inner = HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS), raise_cap=2,
                                encoder=None if encoder_name == "raw" else make_encoder(encoder_name))
        self.raw = HoldemGame(starting_stack=100.0, bet_fractions=dict(BETS), raise_cap=2)
        self.board = tuple(codes(board))
        self.encoder_name = encoder_name
        deals = [((a, b), 1.0) for a in r0 for b in r1 if not set(a) & set(b)]
        self._deals = [(d, w / len(deals)) for d, w in deals]
        probe = HoldemState(holes=((0, 1), (2, 3)), board=self.board, streets=LINE, contrib=(0.0, 0.0))
        _, total, _, _ = self.inner._replay(probe)
        self._contrib = tuple(total)

    def signature(self):
        return f"HoldemRiverSubgame:{self.encoder_name}"

    def root(self):
        return None

    def is_chance(self, s):
        return s is None

    def chance_outcomes(self, s):
        return [(w, HoldemState(holes=d, board=self.board, streets=LINE, contrib=self._contrib))
                for d, w in self._deals]

    def is_terminal(self, s):
        return s is not None and self.inner.is_terminal(s)

    def utility(self, s):
        return self.inner.utility(s)

    def current_player(self, s):
        return self.inner.current_player(s)

    def infoset_key(self, s):
        return self.inner.infoset_key(s)

    def legal_actions(self, s):
        return self.inner.legal_actions(s)

    def next_state(self, s, a):
        return self.inner.next_state(s, a)


class _Lifted(dict):
    """Abstract strategy read through raw keys: raw key -> abstract probs."""


def lift(abstract_game, raw_game, abstract_strategy):
    out = {}

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
            acts = raw_game.legal_actions(s)
            p = abstract_strategy.get(ak)
            out[rk] = p if p is not None else {a: 1.0 / len(acts) for a in acts}
        for a in raw_game.legal_actions(s):
            walk(raw_game.next_state(s, a))
    walk(raw_game.root())
    return out


def distance(ref, strat, weights):
    tot = sum(weights.values())
    return sum(w * sum(abs(ref[k][a] - strat[k].get(a, 0.0)) for a in ref[k])
               for k, w in weights.items() if k in ref and k in strat) / tot


def part_a(boards, iters):
    rows = []
    for name, board, b0, b1 in boards:
        r0 = [tuple(sorted(codes([h[:2], h[2:]]))) for h in percentile_range(board, b0, 14)]
        r1 = [tuple(sorted(codes([h[:2], h[2:]]))) for h in percentile_range(board, b1, 14)]
        raw = HoldemRiverSubgame(board, r0, r1, "raw")
        res = {"name": name, "board": board, "range_sizes": [len(r0), len(r1)], "encoders": {}}
        ref = None
        for enc in ("raw", "bucket", "transition", "compact"):
            g = raw if enc == "raw" else HoldemRiverSubgame(board, r0, r1, enc)
            t = time.time()
            s = CFRPlusSolver(g)
            s.train(iters)
            secs = time.time() - t
            table_bytes = sum(sys.getsizeof(k) + 2 * n.regret_sum.nbytes + 300 for k, n in s.infosets.items())
            strat = lift(g, raw, s.average_strategy())
            if enc == "raw":
                ref = strat
                weights = infoset_reach(raw, ref)
                value = expected_value(raw, ref)
            res["encoders"][enc] = {
                "abstract_infosets": len(s.infosets), "seconds": round(secs, 1),
                "infoset_table_kb_estimate": round(table_bytes / 1e3, 1),
                "exploitability_in_raw_game": exploitability(raw, strat),
                "ev_p0": expected_value(raw, strat),
                "ev_error_vs_raw": abs(expected_value(raw, strat) - value),
                "weighted_l1_vs_raw": distance(ref, strat, weights)}
            print(name, enc, {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in res["encoders"][enc].items()}, flush=True)
        res["pot_bb"] = sum(raw._contrib)
        rows.append(res)
    return rows


# -- 34B collision attribution -------------------------------------------------

def part_b(top_keys, hands, seed):
    from poker_alpha.abstraction.betting_history import betting_context
    from poker_alpha.abstraction.features import card_features
    from poker_alpha.poker.ranges import COMBO_CLASS, COMBO_INDEX
    from poker_alpha.solver_config import PRIMARY_CONFIG
    from poker_alpha.validation.abstraction_audit import generate_corpus

    game = PRIMARY_CONFIG.build_game()
    wanted = {k for k, _ in top_keys}
    members = defaultdict(list)
    corpus = generate_corpus(game, hands, seed)
    for st in corpus:
        k = game.infoset_key(st)
        if k in wanted and len(members[k]) < 400:
            members[k].append(st)
    from poker_alpha.abstraction.cards import hand_equity
    rows = []
    for k, seed_l1 in top_keys:
        ms = members.get(k, [])
        if not ms:
            rows.append({"key": k, "seed_disagreement_l1": seed_l1, "members_found": 0})
            continue
        eqs, made, draws, nuts, tex, sprs, lines, pre = [], Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
        for st in ms[:120]:
            p = game.current_player(st)
            hole = st.holes[p]
            f = card_features(hole, st.board)
            made[f.made] += 1
            draws[f.draw] += 1
            nuts[f.nut] += 1
            tex[f.texture] += 1
            ctx = betting_context(game, st)
            sprs[ctx.spr] += 1
            lines["/".join(st.streets)] += 1
            a, b = sorted(hole)
            pre[COMBO_CLASS[COMBO_INDEX[(a, b)]]] += 1
            eqs.append(hand_equity(hole, st.board, samples=200, seed=0) if st.board else None)
        eqs = [e for e in eqs if e is not None]
        n = sum(made.values())
        eq_range = (max(eqs) - min(eqs)) if eqs else 0.0
        top_made = made.most_common(1)[0][1] / n
        top_line = lines.most_common(1)[0][1] / n
        severity = 0.5 * eq_range + 0.25 * (1 - top_made) + 0.25 * (1 - top_line)
        rows.append({"key": k, "seed_disagreement_l1": seed_l1, "members_found": len(ms),
                     "sampled": n, "equity_min": round(min(eqs), 3) if eqs else None,
                     "equity_max": round(max(eqs), 3) if eqs else None,
                     "equity_std": round(float(np.std(eqs)), 3) if eqs else None,
                     "made_classes": made.most_common(6), "draw_classes": dict(draws),
                     "nut_status": dict(nuts), "textures": tex.most_common(4),
                     "spr_buckets": dict(sprs), "distinct_lines": len(lines),
                     "lines": lines.most_common(4), "distinct_preflop_classes": len(pre),
                     "top_preflop_classes": pre.most_common(5),
                     "severity": round(severity, 3)})
    return rows


def suspicious_keys(n=20):
    d = json.loads((ROOT / "results" / "validation" / "holdem_training_v1_300k.json").read_text())
    mats = d["canonical_matrix"]["final"]
    by_key = defaultdict(dict)
    for s, rows in mats.items():
        for r in rows:
            if r["status"] == "visited" and r["visits"] >= 20 and r["street"] != "preflop":
                by_key[r["key"]][s] = r["policy"]
    out = []
    for k, pols in by_key.items():
        if len(pols) < 3:
            continue
        ps = list(pols.values())
        l1 = np.mean([sum(abs(a[x] - b[x]) for x in a) for i, a in enumerate(ps) for b in ps[i + 1:]])
        out.append((k, round(float(l1), 3)))
    return sorted(out, key=lambda x: -x[1])[:n]


BOARDS = [
    ("dry_high", ["Ks", "7d", "2c", "Qh", "4s"], [(0.0, 0.12), (0.88, 1.0)], [(0.40, 0.85)]),
    ("four_flush", ["Qh", "9h", "5h", "2h", "Kc"], [(0.55, 0.9)], [(0.35, 0.8)]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--corpus-hands", type=int, default=30000)
    a = p.parse_args()
    t = time.time()
    doc = {"format": "pokeralpha.abstraction_error/v1",
           "part_a_setup": {"tree": "HoldemGame river, line b75c/cc/cc (pot 5 BB, 97.5 BB behind), "
                                    "bets 33/75/150% + all-in, raise cap 2",
                            "solver": f"CFR+ {a.iters} iterations per encoder",
                            "ranges": "14 combos per player from river-strength percentile bands"}}
    doc["part_a"] = part_a(BOARDS, a.iters)
    keys = suspicious_keys()
    doc["part_b"] = {"selection": "canonical-matrix keys (postflop, >= 20 visits on all seeds) with the "
                                  "highest seed disagreement at 300k",
                     "corpus": {"hands": a.corpus_hands, "seed": 3},
                     "severity": "0.5 * within-key equity range + 0.25 * (1 - top made-class share) "
                                 "+ 0.25 * (1 - top action-line share); heuristic, from measured dispersion",
                     "rows": part_b(keys, a.corpus_hands, 3)}
    doc["seconds"] = round(time.time() - t)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
