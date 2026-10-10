"""Phases 8-11 (final-trust): policy disagreement vs VALUE disagreement.

For canonical preflop states of the full release game, estimate each legal
action's EV by Monte Carlo playouts: hero is forced into the action, then
BOTH players follow a fixed artifact strategy (uniform at unseen keys) to
showdown on pre-dealt duplicate boards — the same deals for every action
and every seed-environment, so EV *differences* are low-variance.

Per state and per seed-environment s:  q_s(a) = EV(forced a | self-play of
seed s).  Reported per state:

* per-seed policies at the key and pairwise policy L1;
* per-environment q vectors, best action, action margin (best − second);
* cross-seed policy regret: regret of seed j's policy measured under seed
  i's environment, max over i != j — the EV consequence of the observed
  policy disagreement;
* classification A/B/C/D: high/low policy L1 (>= 0.3) x high/low EV
  consequence (cross-regret >= 0.5% of pot, i.e. 0.0075 bb at the 1.5 bb
  root pot — scaled by the state's pot).

Modes:
  --canonical   12 classes x 7 situations, 3000 deals/action (deep, Phase 8)
  --btn169      all 169 classes, BTN unopened, 800 deals/action (broad, Phase 11)

EV here is self-play value inside the abstract strategy family measured in
the REAL game (real dealing, real evaluator) — a decision-consequence
proxy, not exploitability.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.abstraction.cards import all_preflop_classes, combos_for_class  # noqa: E402
from poker_alpha.solver_config import V2_CONFIG  # noqa: E402

CANONICAL_CLASSES = ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs",
                     "A5s", "76s", "22", "72o"]
SITUATIONS = [
    ("BTN unopened", [], 0),
    ("BB vs x200 open", ["x200"], 1),
    ("BB vs x250 open", ["x250"], 1),
    ("BB vs x350 open", ["x350"], 1),
    ("BTN vs 3-bet (x200,x250)", ["x200", "x250"], 0),
    ("BTN vs jam (x200,a)", ["x200", "a"], 0),
    ("BB vs jam (x200,x250,a)", ["x200", "x250", "a"], 1),
]

_WORKER = {}


def _init_worker(artifact_paths):
    from poker_alpha.solvers.strategy_artifact import load_artifact

    game = V2_CONFIG.build_game()
    _WORKER["game"] = game
    _WORKER["arts"] = {name: load_artifact(ROOT / p, game)
                       for name, p in artifact_paths.items()}


def _playout(game, art, state, board, rng) -> float:
    """Finish the hand from `state` with both players on `art` (uniform at
    unseen keys), the board fixed. Returns utility for player 0."""
    n_board = {1: 3, 2: 4, 3: 5}
    while not game.is_terminal(state):
        if game.is_chance(state):
            state = replace(state, board=board[: n_board[state.street + 1]],
                            streets=state.streets + ("",))
            continue
        legal = game.legal_actions(state)
        hit = art.lookup(game.infoset_key(state))
        if hit is None:
            probs = np.full(len(legal), 1.0 / len(legal))
        else:
            probs = np.array([hit[0].get(a, 0.0) for a in legal])
            s = probs.sum()
            probs = probs / s if s > 0 else np.full(len(legal), 1.0 / len(legal))
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]
        state = game.next_state(state, legal[int(np.searchsorted(cdf, rng.random(),
                                                                side="right"))])
    return game.utility(state)


def eval_state(job):
    """One canonical state: returns per-env q vectors and policies."""
    (sid, cls, tokens, actor, deals_per_action, seed_base) = job
    game = _WORKER["game"]
    arts = _WORKER["arts"]
    combos = combos_for_class(cls)
    deal_rng = np.random.default_rng(seed_base)

    # Pre-deal: hero combo, opponent holes, full board — shared across
    # actions and environments.
    deals = []
    for _ in range(deals_per_action):
        hero = combos[int(deal_rng.integers(len(combos)))]
        dead = set(hero)
        live = [c for c in range(52) if c not in dead]
        pick = deal_rng.choice(len(live), size=7, replace=False)
        opp = (live[pick[0]], live[pick[1]])
        board = tuple(live[pick[i]] for i in range(2, 7))
        deals.append((hero, opp, board))

    # Build the pre-action state once per deal (cheap).
    def pre_state(hero, opp):
        holes = (tuple(hero), tuple(opp)) if actor == 0 else (tuple(opp), tuple(hero))
        s = replace(game.root(), holes=holes)
        for t in tokens:
            s = game.next_state(s, t)
        return s

    probe = pre_state(*deals[0][:2])
    assert game.current_player(probe) == actor
    legal = game.legal_actions(probe)
    key = game.infoset_key(probe)   # NOTE: key depends on hero's hole class
    # keys differ per combo only through the hole class — same class => same key
    pot = probe.pot

    out = {"id": sid, "class": cls, "tokens": tokens, "actor": actor,
           "legal": legal, "pot": pot, "deals_per_action": deals_per_action,
           "envs": {}}
    for name, art in arts.items():
        hit = art.lookup(key)
        policy = ({a: round(float(hit[0].get(a, 0.0)), 4) for a in legal}
                  if hit else None)
        visits = float(hit[1]) if hit else 0.0
        q = {}
        se = {}
        for ai, action in enumerate(legal):
            vals = np.empty(len(deals))
            for di, (hero, opp, board) in enumerate(deals):
                s = game.next_state(pre_state(hero, opp), action)
                rng = np.random.default_rng(
                    (seed_base * 1_000_003 + di * 97 + ai) & 0x7FFFFFFF)
                u0 = _playout(game, art, s, board, rng)
                vals[di] = u0 if actor == 0 else -u0
            q[action] = float(vals.mean())
            se[action] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        out["envs"][name] = {"policy": policy, "visits": visits,
                             "q": {a: round(v, 4) for a, v in q.items()},
                             "q_se": {a: round(v, 4) for a, v in se.items()}}
    return out


def analyze(rows, l1_high=0.3, regret_frac_of_pot=0.005):
    """Classify every state A/B/C/D and compute summary stats."""
    for row in rows:
        envs = row["envs"]
        names = [n for n, e in envs.items() if e["policy"] is not None]
        # pairwise policy L1
        l1s = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pa, pb = envs[a]["policy"], envs[b]["policy"]
                l1s.append(sum(abs(pa[x] - pb[x]) for x in row["legal"]))
        row["policy_l1_mean"] = round(float(np.mean(l1s)), 4) if l1s else None
        # margins and cross-seed regret
        margins = []
        best = {}
        for n in envs:
            q = envs[n]["q"]
            vals = sorted(q.values(), reverse=True)
            margins.append(vals[0] - vals[1] if len(vals) > 1 else 0.0)
            best[n] = max(q, key=q.get)
        row["action_margin_mean"] = round(float(np.mean(margins)), 4)
        row["best_actions"] = best
        row["best_action_agreement"] = len(set(best.values())) == 1
        cross = []
        for i_env in envs:
            q = envs[i_env]["q"]
            mx = max(q.values())
            for j_env in names:
                if j_env == i_env:
                    continue
                pol = envs[j_env]["policy"]
                tot = sum(pol.values())
                mix = (sum(pol[a] * q[a] for a in q) / tot) if tot > 0 else mx
                cross.append(mx - mix)
        row["cross_regret_max"] = round(float(max(cross)), 4) if cross else None
        row["cross_regret_mean"] = round(float(np.mean(cross)), 4) if cross else None
        hi_l1 = row["policy_l1_mean"] is not None and row["policy_l1_mean"] >= l1_high
        thresh = regret_frac_of_pot * row["pot"]
        hi_ev = row["cross_regret_max"] is not None and row["cross_regret_max"] >= max(thresh, 0.05)
        row["class_quadrant"] = ("A_highL1_highEV" if hi_l1 and hi_ev else
                                 "B_highL1_lowEV" if hi_l1 else
                                 "C_lowL1_highEV" if hi_ev else
                                 "D_lowL1_lowEV")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("canonical", "btn169"), default="canonical")
    ap.add_argument("--deals", type=int, default=0, help="0 = mode default")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--artifacts", default="native_1m",
                    choices=("native_1m", "native_300k", "release_200k", "native_2m"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    art_sets = {
        "native_1m": {f"seed{s}": f"results/strategy/candidates/native_1m/holdem_v2_native_seed{s}_1000k.npz"
                      for s in (0, 1, 2)},
        "native_300k": {f"seed{s}": f"results/strategy/candidates/native_300k/holdem_v2_native_seed{s}_300k.npz"
                        for s in (0, 1, 2)},
        "native_2m": {f"seed{s}": f"results/strategy/candidates/native_2m/holdem_v2_native_seed{s}_2000k.npz"
                      for s in (0, 1, 2)},
        "release_200k": {"seed0": "results/strategy/holdem_v2_seed0_200k.npz",
                         "seed1": "results/strategy/candidates/v2_extension/holdem_v2_seed1_200k.npz",
                         "seed2": "results/strategy/candidates/v2_extension/holdem_v2_seed2_200k.npz"},
    }[args.artifacts]

    if args.mode == "canonical":
        deals = args.deals or 3000
        jobs = []
        sid = 0
        for sit, tokens, actor in SITUATIONS:
            for cls in CANONICAL_CLASSES:
                jobs.append((f"{sit}::{cls}", cls, tokens, actor, deals,
                             100_000 + sid))
                sid += 1
        default_out = ROOT / f"results/validation/policy_vs_value_disagreement_{args.artifacts}.json"
    else:
        deals = args.deals or 800
        jobs = [(f"BTN unopened::{cls}", cls, [], 0, deals, 200_000 + i)
                for i, cls in enumerate(all_preflop_classes())]
        default_out = ROOT / f"results/validation/preflop169_value_{args.artifacts}.json"
    out_path = args.out or default_out

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                             initargs=(art_sets,)) as ex:
        for i, row in enumerate(ex.map(eval_state, jobs, chunksize=2)):
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)} states, {time.time()-t0:.0f}s", flush=True)
    rows = analyze(rows)

    from collections import Counter

    quad = Counter(r["class_quadrant"] for r in rows)
    doc = {"format": "pokeralpha.policy_vs_value_disagreement/v1",
           "mode": args.mode, "artifacts": args.artifacts,
           "artifact_paths": art_sets, "deals_per_action": deals,
           "ev_semantics": ("self-play value of the artifact family in the real "
                            "game; duplicate pre-dealt boards shared across "
                            "actions and environments; NOT exploitability"),
           "quadrants": dict(quad),
           "seconds": round(time.time() - t0, 1),
           "states": rows}
    out_path.write_text(json.dumps(doc, indent=1))
    print("quadrants:", dict(quad))
    print("wrote", out_path, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
