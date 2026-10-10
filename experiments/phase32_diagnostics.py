"""Phase 32B/32C: diagnose the AA-limp and frequent-jam preflop outputs.

Independent checks, each recorded in ``results/validation/solver_quality_v1.json``:

rules      heads-up order, blinds, pot, effective stack, all-in amounts
geometry   concrete chips of every abstract action; how often abstract bets /
           raises are below a legal NLHE minimum bet or minimum raise
utility    exact chip results of simple terminal lines
trace      instrumented external-sampling MCCFR updates at the BTN-AA
           infoset (strategy, child values, node value, regret delta)
averaging  full-history average vs recent-window averages vs current
           strategy, from checkpoint strategy sums (10k / 100k / 300k)
action_ev  EV of each first action for BTN AA (and other classes) under the
           trained profile, paired Monte Carlo (common deals) with SE
recall     which preflop holdings / lines share the flop keys AA reaches
jams       all-in frequency summary from the preflop audit

Usage::

    python experiments/phase32_diagnostics.py --ckpt-dir /home/user/pa_ckpt
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

from poker_alpha.games.holdem import HoldemState  # noqa: E402
from poker_alpha.poker.cards import codes  # noqa: E402
from poker_alpha.poker.ranges import COMBO_CLASS, COMBO_INDEX  # noqa: E402
from poker_alpha.solver_config import PRIMARY_CONFIG  # noqa: E402
from poker_alpha.solvers.mccfr import MCCFRSolver  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402
from poker_alpha.validation.abstraction_audit import generate_corpus  # noqa: E402

OUT = ROOT / "results" / "validation" / "solver_quality_v1.json"


def cls_of(hole):
    a, b = sorted(hole)
    return COMBO_CLASS[COMBO_INDEX[(a, b)]]


# -- rules -------------------------------------------------------------------

def rules_checks(game):
    out = {}
    root = replace(game.root(), holes=((0, 1), (2, 3)))
    sp, tot, me, nr = game._replay(root)
    out["btn_acts_first_preflop"] = me == 0
    out["blinds_sb_bb"] = list(tot) == [0.5, 1.0]
    s = game.next_state(root, "c")          # limp
    out["bb_acts_after_limp"] = game.current_player(s) == 1
    s = game.next_state(s, "c")             # BB checks: street closes
    out["street_closes_after_limp_check"] = game.is_chance(s)
    s = replace(s, board=(10, 11, 12), streets=s.streets + ("",))
    out["bb_acts_first_postflop"] = game.current_player(s) == 1
    out["pot_after_limp_check_bb"] = s.pot
    out["pot_ok"] = abs(s.pot - 2.0) < 1e-9
    a = game.next_state(root, "a")
    out["btn_allin_total"] = a.contrib[0]
    out["allin_ok"] = abs(a.contrib[0] - game.starting_stack) < 1e-9
    c = game.next_state(a, "c")
    out["call_allin_total"] = c.contrib[1]
    out["call_allin_ok"] = abs(c.contrib[1] - game.starting_stack) < 1e-9
    out["runout_after_allin_call_is_chance"] = game.is_chance(c)
    r = game.next_state(root, "b75")
    rr = game.next_state(r, "b75")
    _, tot2, me2, _ = game._replay(rr)
    out["effective_stack_after_3bet"] = game.starting_stack - max(tot2)
    out["all_pass"] = all(v for k, v in out.items() if k.endswith(("_ok", "first_preflop", "after_limp",
                                                                    "postflop", "_sb_bb", "is_chance",
                                                                    "limp_check")))
    return out


# -- geometry ------------------------------------------------------------------

def geometry(game, hands=1500, seed=5):
    """Fraction of abstract bets/raises below NLHE minimums on a corpus."""
    corpus = generate_corpus(game, hands, seed)
    tally = Counter()
    examples = defaultdict(list)
    for st in corpus:
        street_paid, total, me, n_raises = game._replay(st)
        owe = street_paid[1 - me] - street_paid[me]
        from poker_alpha.games.holdem import _tokens
        # replay current street to get increments
        totals = [0.5, 1.0]
        for si, sa in enumerate(st.streets):
            paid = [0.5, 1.0] if si == 0 else [0.0, 0.0]
            to_act = 0 if si == 0 else 1
            incs = [1.0] if si == 0 else []
            for tok in _tokens(sa):
                m, o = to_act, 1 - to_act
                ow = paid[o] - paid[m]
                stack = game.starting_stack - totals[m]
                if tok == "c":
                    add = min(ow, stack)
                elif tok == "a":
                    add = stack
                elif tok == "f":
                    add = 0.0
                else:
                    add = ow + game.bet_fractions[tok] * (totals[0] + totals[1] + ow)
                if tok not in ("c", "f") and add - ow > 1e-9:
                    incs.append(add - ow)
                paid[m] += add
                totals[m] += add
                to_act = o
        min_inc = max([1.0] + incs[-1:]) if incs else 1.0
        for tok in game.legal_actions(st):
            if tok in ("f", "c", "a"):
                continue
            nxt = game.next_state(st, tok)
            add = nxt.contrib[me] - st.contrib[me]
            inc = add - owe
            kind = ("pre" if st.street == 0 else "post") + ("_raise" if owe > 1e-9 or st.street == 0 else "_bet")
            tally[(kind, tok, "total")] += 1
            if inc < min_inc - 1e-9:
                tally[(kind, tok, "below_min")] += 1
                if len(examples[(kind, tok)]) < 3:
                    examples[(kind, tok)].append({"streets": list(st.streets), "increment_bb": round(inc, 3),
                                                  "min_increment_bb": round(min_inc, 3)})
    rows = {}
    for (kind, tok, what), n in tally.items():
        if what != "total":
            continue
        below = tally[(kind, tok, "below_min")]
        rows[f"{kind}:{tok}"] = {"offered": n, "below_nlhe_minimum": below,
                                 "share_below": round(below / n, 4),
                                 "examples": examples[(kind, tok)]}
    return rows


# -- utility -------------------------------------------------------------------

def utility_checks(game):
    holes = (tuple(codes(["As", "Ah"])), tuple(codes(["7c", "2d"])))
    board = tuple(codes(["Kd", "9s", "4h", "3c", "Jd"]))
    root = replace(game.root(), holes=holes)
    out = {}
    s = game.next_state(root, "f")
    out["sb_folds"] = {"u0": game.utility(s), "expected": -0.5}
    s = game.next_state(game.next_state(root, "b75"), "f")
    out["bb_folds_to_open"] = {"u0": game.utility(s), "expected": 1.0}

    def run(seq_by_street):
        st = root
        for si, acts in enumerate(seq_by_street):
            if si > 0:
                st = replace(st, board=board[:[0, 3, 4, 5][si]], streets=st.streets + ("",))
            for a in acts:
                st = game.next_state(st, a)
        while game.is_chance(st):
            si = st.street + 1
            st = replace(st, board=board[:[0, 3, 4, 5][si]], streets=st.streets + ("",))
        return st
    s = run([["b75", "c"], ["c", "c"], ["c", "c"], ["c", "c"]])
    out["raise_call_checkdown_AA_wins"] = {"u0": game.utility(s), "expected": 2.5}
    s = run([["a", "c"]])
    out["allin_call_AA_wins"] = {"u0": game.utility(s), "expected": 100.0}
    flipped = replace(s, holes=(holes[1], holes[0]))
    out["allin_call_72_loses"] = {"u0": game.utility(flipped), "expected": -100.0}
    tie_holes = (tuple(codes(["Ac", "Qd"])), tuple(codes(["As", "Qh"])))
    s = run([["a", "c"]])
    s = replace(s, holes=tie_holes)
    out["allin_split"] = {"u0": game.utility(s), "expected": 0.0}
    for v in out.values():
        v["ok"] = abs(v["u0"] - v["expected"]) < 1e-9
    out["all_pass"] = all(v["ok"] for v in out.values())
    return out


# -- MCCFR trace ---------------------------------------------------------------

class _FixedDealGame:
    """Wraps the game so the root deal gives BTN a fixed hand."""

    def __init__(self, game, btn_hole, rng):
        self.g, self.hole, self.rng = game, tuple(btn_hole), rng

    def __getattr__(self, name):
        return getattr(self.g, name)

    def sample_chance(self, state, rng):
        if state.holes is None:
            live = [c for c in range(52) if c not in self.hole]
            v = rng.choice(len(live), size=2, replace=False)
            return replace(state, holes=(self.hole, (live[v[0]], live[v[1]])))
        return self.g.sample_chance(state, rng)


class _TracingSolver(MCCFRSolver):
    def __init__(self, game, key, seed):
        super().__init__(game, seed)
        self.key, self.log, self.raw = key, [], []

    def _traverse(self, state, update_player):
        game = self.game
        if (not game.is_terminal(state) and not game.is_chance(state)
                and game.current_player(state) == update_player
                and game.infoset_key(state) == self.key):
            node = self._get_infoset(self.key, game.legal_actions(state))
            before_r, before_s = node.regret_sum.copy(), node.strategy_sum.copy()
            strat = node.current_strategy()
            v = super()._traverse(state, update_player)
            delta = node.regret_sum - before_r
            child = delta + v                     # regret delta = child - node value
            self.raw.append((strat.copy(), (delta + v).copy(), float(v), delta.copy()))
            self.log.append({"iteration": self.iterations, "player": update_player,
                             "villain": cls_of(state.holes[1 - update_player]),
                             "strategy": [round(float(x), 4) for x in strat],
                             "child_values": [round(float(x), 4) for x in child],
                             "node_value": round(float(v), 4),
                             "node_value_check": round(float(np.dot(strat, child)), 4),
                             "regret_delta": [round(float(x), 4) for x in delta],
                             "strategy_sum_delta": [round(float(x), 4)
                                                    for x in node.strategy_sum - before_s]})
            return v
        return super()._traverse(state, update_player)


def mccfr_trace(game, iterations=6, seed=3):
    hole = tuple(codes(["As", "Ah"]))
    rng = np.random.default_rng(seed)
    g = _FixedDealGame(game, hole, rng)
    from poker_alpha.solvers.holdem_analysis import spot_state
    key = game.infoset_key(spot_state(game, "BTN", hole, (), ("",), villain_hole=("7c", "2d")))
    s = _TracingSolver(g, key, seed)
    for _ in range(iterations):
        s.iterate()
    # identities are checked on the unrounded values kept in _raw
    checks = {
        "node_value_equals_strategy_dot_children": all(
            abs(v - float(np.dot(st, ch))) < 1e-9 for st, ch, v, _ in s.raw),
        "regret_delta_sums_to_zero_under_strategy": all(
            abs(float(np.dot(st, d))) < 1e-9 for st, ch, v, d in s.raw),
        "strategy_sum_unchanged_on_traverser_visit": all(
            max(abs(x) for x in e["strategy_sum_delta"]) < 1e-12 for e in s.log),
        "fold_child_value_is_minus_small_blind": all(
            abs(e["child_values"][0] + 0.5) < 1e-9 for e in s.log if e["player"] == 0),
    }
    return {"key": key, "log": s.log[:12], "checks": checks, "all_pass": all(checks.values())}


# -- averaging -----------------------------------------------------------------

def averaging(audit):
    out = {}
    for name in ("BTN_first|AA", "BTN_first|KK", "BTN_first|72o", "BB_vs_limp|AA",
                 "BB_vs_open75|AKo", "BB_vs_open33|99"):
        rec = audit["records"][name]
        per = {}
        for s in (0, 1, 2):
            cps = {it: rec["checkpoints"].get(f"s{s}@{it}") for it in (10000, 100000, 300000)}
            if not all(cps.values()):
                continue

            def win(a, b):
                d = np.array(cps[b]["strategy_sum"]) - np.array(cps[a]["strategy_sum"])
                return [round(float(x), 4) for x in d / d.sum()] if d.sum() > 0 else None
            per[str(s)] = {"full_average_300k": cps[300000]["avg"],
                           "window_100k_300k": win(100000, 300000),
                           "window_10k_100k": win(10000, 100000),
                           "current_300k": cps[300000]["current"],
                           "visits_by_window": [cps[10000]["visits"],
                                                round(cps[100000]["visits"] - cps[10000]["visits"]),
                                                round(cps[300000]["visits"] - cps[100000]["visits"])]}
        out[name] = {"legal": rec["legal"], "per_seed": per}
    return out


# -- action EVs under the trained profile ---------------------------------------

def _table(solver):
    return {k: n.average_strategy() for k, n in solver.infosets.items() if n.strategy_sum.sum() > 0}


def action_evs(game, table, cls_hole, first_actions, deals, seed):
    """Paired MC EV for BTN of each forced first action, both players then
    following ``table`` (uniform if unvisited). Same deals and action
    uniforms for every first action (common random numbers)."""
    hole = tuple(cls_hole)
    rng = np.random.default_rng(seed)
    live = [c for c in range(52) if c not in hole]
    vals = {a: np.zeros(deals) for a in first_actions}
    for d in range(deals):
        pick = rng.choice(len(live), size=7, replace=False)
        vill = (live[pick[0]], live[pick[1]])
        board = tuple(live[i] for i in pick[2:7])
        us = rng.random(64)
        for a in first_actions:
            st = replace(game.root(), holes=(hole, vill))
            st = game.next_state(st, a)
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
            vals[a][d] = game.utility(st)
    base = first_actions[0]
    out = {}
    for a in first_actions:
        diff = vals[a] - vals[base]
        out[a] = {"ev_bb": round(float(vals[a].mean()), 4),
                  "se_bb": round(float(vals[a].std(ddof=1) / np.sqrt(deals)), 4),
                  f"minus_{base}_bb": round(float(diff.mean()), 4),
                  "paired_se_bb": round(float(diff.std(ddof=1) / np.sqrt(deals)), 4)}
    return out


# -- imperfect recall at the flop -------------------------------------------------

def recall_contamination(game, hands=4000, seed=11):
    """For flop keys reached by BTN AA, which other preflop classes and which
    preflop lines share the key?"""
    corpus = generate_corpus(game, hands, seed)
    members = defaultdict(lambda: {"classes": Counter(), "lines": Counter()})
    for st in corpus:
        if st.street != 1:
            continue
        p = game.current_player(st)
        k = game.infoset_key(st)
        members[k]["classes"][cls_of(st.holes[p])] += 1
        members[k]["lines"][st.streets[0]] += 1
    aa_keys = [k for k, m in members.items() if m["classes"].get("AA")]
    rows = []
    for k in sorted(aa_keys, key=lambda k: -sum(members[k]["classes"].values()))[:15]:
        m = members[k]
        rows.append({"key": k, "states": sum(m["classes"].values()),
                     "distinct_preflop_classes": len(m["classes"]),
                     "distinct_preflop_lines": len(m["lines"]),
                     "AA_share": round(m["classes"]["AA"] / sum(m["classes"].values()), 4),
                     "top_classes": m["classes"].most_common(6),
                     "lines": m["lines"].most_common(6)})
    return {"flop_keys_with_AA": len(aa_keys), "examples": rows,
            "note": "the preflop key of a first action is the exact 169 class (no collision); "
                    "from the flop on the compact key forgets the preflop class and line"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--deals", type=int, default=20000)
    a = p.parse_args()
    game = PRIMARY_CONFIG.build_game()
    t = time.time()
    doc = {"format": "pokeralpha.solver_quality/v1", "config_signature": PRIMARY_CONFIG.signature()}
    doc["rules"] = rules_checks(game)
    doc["geometry"] = geometry(game)
    doc["utility"] = utility_checks(game)
    doc["mccfr_trace"] = mccfr_trace(game)
    audit = json.loads((ROOT / "results" / "validation" / "preflop_audit_v1.json").read_text())
    doc["averaging"] = averaging(audit)
    doc["jams"] = {k: {kk: vv for kk, vv in v.items() if kk != "top"} | {"top5": v["top"][:5]}
                   for k, v in audit["jam_distribution"].items()}
    print("static parts", round(time.time() - t), "s", flush=True)
    evs = {}
    for seed in (0, 1):
        solver = load_checkpoint(a.ckpt_dir / f"locked_seed{seed}_it300000.npz", game)
        table = _table(solver)
        del solver
        for cls, cards in (("AA", ["As", "Ah"]), ("KK", ["Ks", "Kh"]), ("72o", ["7c", "2h"])):
            evs[f"seed{seed}@300k|BTN_first|{cls}"] = action_evs(
                game, table, codes(cards), ["c", "b33", "b75", "b150", "a", "f"], a.deals, 100 + seed)
            print(cls, seed, evs[f"seed{seed}@300k|BTN_first|{cls}"], flush=True)
    doc["action_ev"] = {"deals": a.deals, "note": "EV for BTN (bb) of each forced first action, both "
                        "players then follow the seed's 300k average strategy; paired differences vs limp",
                        "results": evs}
    doc["recall"] = recall_contamination(game)
    doc["seconds"] = round(time.time() - t)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
