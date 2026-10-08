"""Phase 32A/32C: full 169-class preflop audit of the trained locked config.

For every preflop class and situation (BTN first action; BB vs limp, vs
33 / 75 / 150% opens, vs all-in; BTN vs BB raise after a limp; BTN vs a 75%
3-bet after a 75% open) and every available checkpoint (seeds x
iterations), records the infoset key, visits, average strategy, current
(regret-matching) strategy, regret sums and strategy sums, plus
100k->300k movement and seed disagreement. Writes

* ``results/validation/preflop_audit_v1.json`` (committed; it is the
  persistent extract, so later phases do not need the checkpoints),
* heatmaps ``results/figures/phase32/<situation>_<actionclass>.png``.

Also summarizes the action geometry (concrete chips of every abstract
preflop action) and all-in frequencies (32C).

Usage::

    python experiments/phase32_preflop_audit.py --ckpt-dir /home/user/pa_ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.poker.ranges import CLASS_MEMBERS, COMBOS  # noqa: E402
from poker_alpha.solver_config import PRIMARY_CONFIG  # noqa: E402
from poker_alpha.solvers.holdem_analysis import action_label, spot_state  # noqa: E402
from poker_alpha.solvers.serialize import load_checkpoint  # noqa: E402

RANKS = "AKQJT98765432"
SITUATIONS = {                   # name -> (position, streets)
    "BTN_first": ("BTN", ("",)),
    "BB_vs_limp": ("BB", ("c",)),
    "BB_vs_open33": ("BB", ("b33",)),
    "BB_vs_open75": ("BB", ("b75",)),
    "BB_vs_open150": ("BB", ("b150",)),
    "BB_vs_allin": ("BB", ("a",)),
    "BTN_limp_vs_raise33": ("BTN", ("cb33",)),
    "BTN_limp_vs_raise75": ("BTN", ("cb75",)),
    "BTN_limp_vs_raise150": ("BTN", ("cb150",)),
    "BTN_open75_vs_3bet75": ("BTN", ("b75b75",)),
}
ACTION_CLASSES = {"f": "fold", "c": "call_or_limp", "b33": "small_raise",
                  "b75": "medium_raise", "b150": "large_raise", "a": "all_in"}


def classes_grid():
    """13x13 grid: pairs on the diagonal, suited above, offsuit below."""
    grid = []
    for i, r1 in enumerate(RANKS):
        row = []
        for j, r2 in enumerate(RANKS):
            if i == j:
                row.append(r1 + r2)
            elif i < j:
                row.append(r1 + r2 + "s")
            else:
                row.append(r2 + r1 + "o")
        grid.append(row)
    return grid


def hole_of(cls):
    a, b = COMBOS[CLASS_MEMBERS[cls][0]]
    return int(a), int(b)


def villain_for(hole):
    return tuple(c for c in range(52) if c not in hole)[:2]


def node_record(node):
    if node is None:
        return None
    v = float(node.strategy_sum.sum())
    return {"visits": round(v, 3),
            "avg": [round(float(x), 5) for x in node.average_strategy()],
            "current": [round(float(x), 5) for x in node.current_strategy()],
            "regret_sum": [round(float(x), 3) for x in node.regret_sum],
            "strategy_sum": [round(float(x), 3) for x in node.strategy_sum]}


def l1(a, b):
    return float(np.abs(np.array(a) - np.array(b)).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--iterations", default="10000,100000,300000")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "validation" / "preflop_audit_v1.json")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "results" / "figures" / "phase32")
    a = p.parse_args()
    game = PRIMARY_CONFIG.build_game()
    seeds = [int(x) for x in a.seeds.split(",")]
    its = [int(x) for x in a.iterations.split(",")]
    grid = classes_grid()
    classes = [c for row in grid for c in row]

    # Static part: keys, legal actions, concrete geometry.
    states, geometry = {}, {}
    for sit, (pos, streets) in SITUATIONS.items():
        for cls in classes:
            hole = hole_of(cls)
            s = spot_state(game, pos, hole, (), streets, villain_hole=villain_for(hole))
            states[(sit, cls)] = (game.infoset_key(s), game.legal_actions(s))
        s = states[(sit, "AA")]
        hole = hole_of("AA")
        st = spot_state(game, pos, hole, (), streets, villain_hole=villain_for(hole))
        street_paid, total, me, _ = game._replay(st)
        geometry[sit] = {
            "streets": list(streets), "pot_before_bb": round(sum(total), 4),
            "to_call_bb": round(street_paid[1 - me] - street_paid[me], 4),
            "actions": {tok: {"label": action_label(game, st, tok),
                              "raise_to_bb": round(game.next_state(st, tok).contrib[me], 4)}
                        for tok in game.legal_actions(st)}}

    records = {f"{sit}|{cls}": {"situation": sit, "class": cls, "key": states[(sit, cls)][0],
                                "legal": states[(sit, cls)][1], "checkpoints": {}}
               for (sit, cls) in states}
    for seed in seeds:
        for it in its:
            path = a.ckpt_dir / f"locked_seed{seed}_it{it}.npz"
            if not path.exists():
                print("missing", path)
                continue
            solver = load_checkpoint(path, game)
            for rec in records.values():
                rec["checkpoints"][f"s{seed}@{it}"] = node_record(solver.infosets.get(rec["key"]))
            print("loaded", path.name, flush=True)
            del solver

    final, prev = its[-1], its[-2] if len(its) > 1 else None
    surprises = []
    for rec in records.values():
        cps = rec["checkpoints"]
        fin = [cps.get(f"s{s}@{final}") for s in seeds]
        fin = [x for x in fin if x]
        if fin:
            mean = np.mean([x["avg"] for x in fin], axis=0)
            rec["mean_avg_final"] = [round(float(x), 4) for x in mean]
            rec["visits_final"] = [x["visits"] for x in fin]
            rec["seed_disagreement_l1"] = round(float(np.mean(
                [l1(x["avg"], y["avg"]) for i, x in enumerate(fin) for y in fin[i + 1:]])), 4) \
                if len(fin) > 1 else None
            rec["avg_vs_current_l1"] = round(float(np.mean([l1(x["avg"], x["current"]) for x in fin])), 4)
            mv = [l1(cps[f"s{s}@{final}"]["avg"], cps[f"s{s}@{prev}"]["avg"])
                  for s in seeds if cps.get(f"s{s}@{final}") and cps.get(f"s{s}@{prev}")]
            rec["movement_prev_to_final_l1"] = round(float(np.mean(mv)), 4) if mv else None
            legal = rec["legal"]
            rec["mean_classes"] = {ACTION_CLASSES[t]: round(float(mean[i]), 4) for i, t in enumerate(legal)}
    # Surprises (flagged, not "fixed"): premium hands folding or passive first-in,
    # trash jamming, high seed disagreement, average far from current.
    premium = {"AA", "KK", "QQ", "AKs", "AKo", "JJ"}
    for rec in records.values():
        mc = rec.get("mean_classes")
        if not mc:
            continue
        flags = []
        if rec["class"] in premium and mc.get("fold", 0) > 0.05:
            flags.append("premium folds > 5%")
        if rec["class"] in premium and rec["situation"] == "BTN_first" and mc.get("call_or_limp", 0) > 0.3:
            flags.append("premium limps > 30%")
        if mc.get("all_in", 0) > 0.3 and rec["situation"] in ("BTN_first", "BB_vs_limp"):
            flags.append("jams > 30% as first raise")
        if (rec.get("seed_disagreement_l1") or 0) > 0.8:
            flags.append("seed disagreement L1 > 0.8")
        if (rec.get("avg_vs_current_l1") or 0) > 1.0:
            flags.append("average vs current strategy L1 > 1.0")
        if flags:
            surprises.append({"state": f"{rec['situation']}|{rec['class']}", "flags": flags,
                              "mean_classes": mc, "visits": rec["visits_final"]})

    # All-in frequency distribution (32C).
    jam = {}
    for sit in SITUATIONS:
        rows = [(r["class"], r["mean_classes"].get("all_in", 0.0), r["visits_final"],
                 r.get("seed_disagreement_l1")) for r in records.values()
                if r["situation"] == sit and r.get("mean_classes")]
        if rows:
            f = np.array([x[1] for x in rows])
            jam[sit] = {"mean_jam_freq": round(float(f.mean()), 4),
                        "classes_jam_gt_50pct": int((f > 0.5).sum()),
                        "classes_jam_gt_20pct": int((f > 0.2).sum()),
                        "top": sorted(rows, key=lambda x: -x[1])[:12]}

    _heatmaps(records, grid, a.fig_dir, final)
    doc = {"format": "pokeralpha.preflop_audit/v1", "config_signature": PRIMARY_CONFIG.signature(),
           "seeds": seeds, "iterations": its, "action_geometry": geometry,
           "action_classes": ACTION_CLASSES, "surprises": surprises, "jam_distribution": jam,
           "records": records}
    a.out.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote", a.out, "surprises", len(surprises))


def _heatmaps(records, grid, fig_dir, final):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for sit in SITUATIONS:
        for tok, cname in ACTION_CLASSES.items():
            m = np.full((13, 13), np.nan)
            for i, row in enumerate(grid):
                for j, cls in enumerate(row):
                    r = records.get(f"{sit}|{cls}")
                    if r and r.get("mean_avg_final") and tok in r["legal"]:
                        m[i, j] = r["mean_avg_final"][r["legal"].index(tok)]
            if np.all(np.isnan(m)):
                continue
            fig, ax = plt.subplots(figsize=(6.4, 6))
            im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
            for i, row in enumerate(grid):
                for j, cls in enumerate(row):
                    if not np.isnan(m[i, j]):
                        ax.text(j, i, f"{cls}\n{m[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                                color="white" if m[i, j] < 0.6 else "black")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{sit}: P({cname}), mean of seeds @ {final}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{sit}_{cname}.png", dpi=72)
            plt.close(fig)


if __name__ == "__main__":
    main()
