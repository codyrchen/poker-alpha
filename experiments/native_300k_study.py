"""Native 300k multi-seed study (Phases 61-63, 97-98).

Inputs: the native training run produced by train_native_multiseed.py
(three seeds, milestones to 300k, native checkpoints + JSONL logs).

Produces:
* per-seed strategy artifacts at the final milestone;
* a confidence table matching the release construction (seeds 0-2 at the
  final milestone, movement vs the earlier milestone) via
  phase37_build_confidence.py;
* gate acceptance on that table via gate_acceptance.py;
* canonical preflop strategies (AA..72o in the standard situations) with
  per-seed disagreement;
* duplicate-deal cross-play with seat swap: new seeds pairwise, and each
  new seed vs the current release artifact (and vs the old python-lineage
  300k candidate);
* a summary JSON (training curves from the JSONL logs included).

Cross-play measures head-to-head strength inside the abstract game only —
it is NOT an exploitability measurement.

    python experiments/native_300k_study.py \
        --train-dir results/native_training --final 300000 --earlier 10000 \
        --out-dir results/strategy/candidates/native_300k \
        --report results/validation/native_training_300k.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.abstraction.cards import combos_for_class  # noqa: E402
from poker_alpha.solver_config import V2_CONFIG, RELEASE_STRATEGY  # noqa: E402
from poker_alpha.solvers.strategy_artifact import load_artifact  # noqa: E402
from poker_alpha.validation.crossplay import duplicate_match, table_policy  # noqa: E402

SEEDS = (0, 1, 2)

CANONICAL_CLASSES = ["AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "72o",
                     "T9s", "66"]
# (name, prior preflop tokens, player to act)
CANONICAL_SITUATIONS = [
    ("BTN unopened", [], 0),
    ("BB vs x200 open", ["x200"], 1),
    ("BB vs x250 open", ["x250"], 1),
    ("BB vs x350 open", ["x350"], 1),
    ("BTN vs 3-bet (x200,x250)", ["x200", "x250"], 0),
    ("BTN vs jam (x200,a)", ["x200", "a"], 0),
]


def canonical_state(game, cls: str, tokens, actor: int):
    """A reachable preflop state with `cls` in the actor's hand."""
    from dataclasses import replace

    hole = combos_for_class(cls)[0]
    used = set(hole)
    filler = [c for c in range(52) if c not in used][:2]
    holes = (tuple(hole), tuple(filler)) if actor == 0 else (tuple(filler), tuple(hole))
    state = replace(game.root(), holes=holes)
    for tok in tokens:
        state = game.next_state(state, tok)
    assert game.current_player(state) == actor
    return state


def seed_artifact_path(out_dir: Path, seed: int, final: int) -> Path:
    return out_dir / f"holdem_v2_native_seed{seed}_{final // 1000}k.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=ROOT / "results/native_training")
    ap.add_argument("--final", type=int, default=300_000)
    ap.add_argument("--earlier", type=int, default=10_000)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results/strategy/candidates/native_300k")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "results/validation/native_training_300k.json")
    ap.add_argument("--deals", type=int, default=4_000)
    ap.add_argument("--skip-crossplay", action="store_true")
    args = ap.parse_args()

    from poker_alpha.native import NativeMCCFRSolver

    args.out_dir.mkdir(parents=True, exist_ok=True)
    game = V2_CONFIG.build_game()
    report: dict = {"format": "pokeralpha.native_300k_study/v1",
                    "config": V2_CONFIG.signature(),
                    "final": args.final, "earlier": args.earlier,
                    "seeds": list(SEEDS), "backend": "native"}

    def ckpt(seed, it):
        return args.train_dir / f"seed{seed}" / f"v2_native_seed{seed}_it{it}.npz"

    # 1. per-seed artifacts ------------------------------------------------
    artifacts = {}
    for seed in SEEDS:
        path = seed_artifact_path(args.out_dir, seed, args.final)
        if not path.exists():
            solver = NativeMCCFRSolver.load_checkpoint(ckpt(seed, args.final),
                                                       V2_CONFIG)
            solver.export_strategy(path, meta={
                "seed": seed, "iterations": args.final,
                "lineage": "native fresh 0->%d (Plan B)" % args.final,
                "source_checkpoint": str(ckpt(seed, args.final))})
            print(f"exported {path}", flush=True)
        artifacts[seed] = load_artifact(path, game)
    report["artifacts"] = {
        str(s): {"path": str(seed_artifact_path(args.out_dir, s, args.final)),
                 "infosets": len(a), "iterations": a.meta["iterations"]}
        for s, a in artifacts.items()}

    # 2. confidence table ---------------------------------------------------
    conf_path = args.out_dir / f"holdem_v2_native_seed0_{args.final // 1000}k_confidence.npz"
    if not conf_path.exists():
        with tempfile.TemporaryDirectory() as td:
            flat = Path(td)
            for seed in SEEDS:
                for it in (args.final, args.earlier):
                    src = ckpt(seed, it)
                    (flat / src.name).symlink_to(src.resolve())
            subprocess.run([sys.executable,
                            str(ROOT / "experiments/phase37_build_confidence.py"),
                            "--ckpt-dir", str(flat), "--config", "v2",
                            "--prefix", "v2_native",
                            "--final", str(args.final),
                            "--earlier", str(args.earlier),
                            "--out", str(conf_path)], check=True)
    report["confidence_table"] = str(conf_path)

    # 3. gate acceptance ----------------------------------------------------
    gate_out = ROOT / f"results/validation/gate_acceptance_v2_native_{args.final // 1000}k.json"
    subprocess.run([sys.executable, str(ROOT / "experiments/gate_acceptance.py"),
                    str(conf_path), "--out", str(gate_out)], check=True)
    report["gate_acceptance"] = json.loads(gate_out.read_text())

    # 4. canonical preflop ---------------------------------------------------
    canon = {}
    for sit_name, tokens, actor in CANONICAL_SITUATIONS:
        per_class = {}
        for cls in CANONICAL_CLASSES:
            state = canonical_state(game, cls, tokens, actor)
            key = game.infoset_key(state)
            per_seed = {}
            for seed in SEEDS:
                hit = artifacts[seed].lookup(key)
                if hit is not None:
                    probs, visits = hit
                    per_seed[str(seed)] = {"probs": {k: round(v, 4) for k, v in probs.items()},
                                           "visits": visits}
            l1 = []
            seeds_present = [s for s in SEEDS if str(s) in per_seed]
            for i, a in enumerate(seeds_present):
                for b in seeds_present[i + 1:]:
                    pa = per_seed[str(a)]["probs"]
                    pb = per_seed[str(b)]["probs"]
                    acts = set(pa) | set(pb)
                    l1.append(sum(abs(pa.get(x, 0) - pb.get(x, 0)) for x in acts))
            per_class[cls] = {"key": key, "per_seed": per_seed,
                              "seed_l1_mean": round(float(np.mean(l1)), 4) if l1 else None}
        canon[sit_name] = per_class
    report["canonical_preflop"] = canon

    # 5. cross-play -----------------------------------------------------------
    if not args.skip_crossplay:
        def lookup_policy(art):
            return table_policy(lambda key: art.lookup(key))

        pols = {f"native{s}_{args.final // 1000}k": lookup_policy(artifacts[s])
                for s in SEEDS}
        release = load_artifact(ROOT / RELEASE_STRATEGY, game)
        pols["release_200k"] = lookup_policy(release)
        old300 = ROOT / "results/strategy/candidates/v2_extension/holdem_v2_seed0_300k.npz"
        if old300.exists():
            pols["python_seed0_300k"] = lookup_policy(load_artifact(old300, game))

        names = list(pols)
        pairings = []
        new_names = [n for n in names if n.startswith("native")]
        for i, a in enumerate(new_names):
            for b in new_names[i + 1:]:
                pairings.append((a, b))
        for a in new_names:
            pairings.append((a, "release_200k"))
        if "python_seed0_300k" in pols:
            pairings.append((new_names[0], "python_seed0_300k"))

        matches = {}
        for i, (a, b) in enumerate(pairings):
            r = duplicate_match(game, pols[a], pols[b], args.deals, seed=7_000 + i)
            matches[f"{a} vs {b}"] = r.to_dict()
            print(f"crossplay {a} vs {b}: {r.bb_per_100:+.1f} bb/100 "
                  f"(95% CI {r.ci95_bb_per_100[0]:+.1f}..{r.ci95_bb_per_100[1]:+.1f})",
                  flush=True)
        report["crossplay"] = {"deals_per_match": args.deals,
                               "note": "duplicate deals, seat swap; abstract game only",
                               "matches": matches}

    # 6. training curves from the JSONL logs ---------------------------------
    curves = {}
    for seed in SEEDS:
        rows = []
        log = args.train_dir / f"v2_native_seed{seed}.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                r = json.loads(line)
                rows.append({k: r.get(k) for k in
                             ("iterations", "iters_per_sec_segment", "infosets",
                              "new_infosets", "top_n_prev_mean_l1",
                              "top_n_prev_median_l1",
                              "entropy_bits_visit_weighted", "max_rss_mb")})
        curves[str(seed)] = rows
    report["training_curves"] = curves

    args.report.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
