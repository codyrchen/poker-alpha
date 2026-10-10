"""Phase 5 (final-trust): quantify the movement signal's horizon dependence.

For the native seed-0 lineage at "current" milestones 200k / 300k / 1M,
compute candidate movement features per infoset key:

A  historical        L1(10k -> current)            (the production v1 signal)
B  medium            L1(100k -> current)
C  recent            L1(previous milestone -> current)
D  recent normalized recent / interval * 100k      (movement per 100k iterations)
E  slope             OLS slope of L1(m_i -> m_{i+1}) over the last 3 segments
F  ewma              exponentially weighted (0.5) mean of recent segment L1s
G  max_recent        max of the last 3 segment L1s
H  median_recent     median of the last 3 segment L1s

and report distributions, the share of keys each feature would push over the
v1 movement thresholds (0.1 low / 0.5 reject), and rank correlation with
seed disagreement at the same milestone (the best available in-vivo
stability reference; true-error calibration happens in Phase 6/7 on exact
games). Writes results/validation/movement_signal_study.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from phase37_build_confidence import policies  # noqa: E402
from poker_alpha.solver_config import V2_CONFIG  # noqa: E402

CK = ROOT / "results/native_training"
MILESTONES = [10_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000,
              400_000, 500_000, 750_000, 1_000_000]
CURRENTS = {200_000: [10_000, 100_000, 150_000],
            300_000: [10_000, 100_000, 250_000],
            1_000_000: [10_000, 100_000, 750_000]}
RECENT_SEGMENTS = {200_000: [(100_000, 150_000), (150_000, 200_000)],
                   300_000: [(200_000, 250_000), (250_000, 300_000)],
                   1_000_000: [(400_000, 500_000), (500_000, 750_000),
                               (750_000, 1_000_000)]}


def ck(seed, it):
    return CK / f"seed{seed}" / f"v2_native_seed{seed}_it{it}.npz"


def l1_map(pa, pb):
    out = {}
    for k, (_, p) in pa.items():
        hit = pb.get(k)
        if hit is not None and len(hit[1]) == len(p):
            out[k] = float(np.abs(p - hit[1]).sum())
    return out


def main() -> None:
    game = V2_CONFIG.build_game()
    needed = sorted({m for cur, lst in CURRENTS.items() for m in lst + [cur]}
                    | {m for segs in RECENT_SEGMENTS.values() for ab in segs for m in ab})
    print("loading", len(needed), "checkpoints (seed 0)...", flush=True)
    pol = {m: policies(ck(0, m), game) for m in needed}

    # Seed disagreement at each current milestone (seeds 0-2).
    def seed_dis(milestone):
        ps = [policies(ck(s, milestone), game) for s in (0, 1, 2)]
        out = {}
        for k, (_, p0) in ps[0].items():
            pols = [p0] + [ps[i][k][1] for i in (1, 2)
                           if k in ps[i] and len(ps[i][k][1]) == len(p0)]
            if len(pols) > 1:
                out[k] = float(np.mean([np.abs(a - b).sum()
                                        for i, a in enumerate(pols)
                                        for b in pols[i + 1:]]))
        return out

    doc = {"format": "pokeralpha.movement_signal_study/v1",
           "lineage": "native seed 0 (seeds 0-2 for disagreement)",
           "v1_thresholds": {"low": 0.1, "reject": 0.5},
           "milestones": {}}

    for current, baselines in CURRENTS.items():
        print("milestone", current, flush=True)
        cur = pol[current]
        feats = {}
        feats["A_hist_10k"] = l1_map(cur, pol[10_000])
        feats["B_med_100k"] = l1_map(cur, pol[100_000])
        prev = baselines[-1]
        feats["C_recent_prev"] = l1_map(cur, pol[prev])
        interval = current - prev
        feats["D_recent_per_100k"] = {k: v * 100_000 / interval
                                      for k, v in feats["C_recent_prev"].items()}
        # segment L1s for slope / ewma / max / median
        segs = RECENT_SEGMENTS[current]
        seg_l1 = [l1_map(pol[b], pol[a]) for a, b in segs]
        common = set(seg_l1[0])
        for s in seg_l1[1:]:
            common &= set(s)
        xs = np.arange(len(segs), dtype=float)
        E, F, G, H = {}, {}, {}, {}
        w = np.array([0.5 ** (len(segs) - 1 - i) for i in range(len(segs))])
        w = w / w.sum()
        for k in common:
            ys = np.array([s[k] for s in seg_l1])
            E[k] = float(np.polyfit(xs, ys, 1)[0]) if len(segs) > 1 else 0.0
            F[k] = float((w * ys).sum())
            G[k] = float(ys.max())
            H[k] = float(np.median(ys))
        feats.update({"E_slope": E, "F_ewma": F, "G_max_recent": G,
                      "H_median_recent": H})

        dis = seed_dis(current)
        row = {"keys": len(cur), "features": {}}
        for name, f in feats.items():
            vals = np.array(list(f.values()))
            common_keys = [k for k in f if k in dis]
            if common_keys:
                a = np.array([f[k] for k in common_keys])
                b = np.array([dis[k] for k in common_keys])
                from scipy.stats import spearmanr

                rho = float(spearmanr(a, b).statistic)
            else:
                rho = None
            row["features"][name] = {
                "n": len(vals),
                "median": float(np.median(vals)),
                "p90": float(np.quantile(vals, 0.9)),
                "share_over_low_0.1": float(np.mean(vals >= 0.1)),
                "share_over_reject_0.5": float(np.mean(vals >= 0.5)),
                "spearman_vs_seed_disagreement": rho,
            }
        doc["milestones"][str(current)] = row

    out = ROOT / "results/validation/movement_signal_study.json"
    out.write_text(json.dumps(doc, indent=1))
    for m, row in doc["milestones"].items():
        print("===", m)
        for name, st in row["features"].items():
            print(f" {name:>18}: med {st['median']:.4f} p90 {st['p90']:.4f} "
                  f">=0.1 {st['share_over_low_0.1']:.2f} >=0.5 {st['share_over_reject_0.5']:.3f} "
                  f"rho(sd) {st['spearman_vs_seed_disagreement']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
