"""Phase II Programs 18 / 51 / 52 / 57: policy geometry and release-strategy
forensics over the 144,025 stored infosets of the release and its sibling
seeds (native, 3 seeds x {1M, 2M} iterations).

Questions (each measured, none assumed):
  * entropy distribution by street; share of pure / mixed policies
  * visit-count distribution: heavy tail? coverage of rare keys?
  * are low-visit keys more mixed?  (visits vs entropy)
  * is seed disagreement concentrated in high-entropy (near-indifferent) keys?
  * is seed disagreement training-limited? Under pure sampling noise the
    per-key disagreement between seeds should shrink ~1/sqrt(2) from 1M to
    2M iterations (ratio ~0.71); a ratio near 1 indicates a structural cause
    (indifference between actions, or imperfect-recall abstraction) that
    more training will not remove.
  * 3-action simplex geometry: vertices / edges / interior

    python experiments/phase2_policy_geometry.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "results" / "strategy" / "candidates"
OUT = ROOT / "results" / "phase2" / "policy_geometry.json"
STREETS = {"0": "preflop", "1": "flop", "2": "turn", "3": "river"}


def load(path):
    z = np.load(path, allow_pickle=False)
    keys = z["keys"]
    off = z["action_offsets"]
    probs = z["probs"].astype(np.float64)
    acts = z["actions"]
    pol = {}
    for i, k in enumerate(keys):
        a, b = off[i], off[i + 1]
        p = probs[a:b]
        s = p.sum()
        pol[str(k)] = (tuple(acts[a:b]), p / s if s > 0 else np.full(b - a, 1.0 / (b - a)))
    return pol, dict(zip(map(str, keys), z["visits"].astype(np.float64)))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def entropy(p):
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def mean_pairwise_tv(pols, k):
    vs = [P[k][1] for P in pols if k in P]
    if len(vs) < 2:
        return None
    return float(np.mean([tv(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]))


def hill_tail_index(x, top_frac=0.05):
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(10, int(len(x) * top_frac))
    xs = x[: k + 1]
    return float(1.0 / np.mean(np.log(xs[:k] / xs[k])))


def main():
    t0 = time.time()
    p2 = [load(CAND / "native_2m" / f"holdem_v2_native_seed{s}_2000k.npz") for s in range(3)]
    p1 = [load(CAND / "native_1m" / f"holdem_v2_native_seed{s}_1000k.npz") for s in range(3)]
    pol2 = [p for p, _ in p2]
    pol1 = [p for p, _ in p1]
    visits = p2[0][1]
    keys = sorted(pol2[0])
    rows = []
    for k in keys:
        acts, p = pol2[0][k]
        sd2 = mean_pairwise_tv(pol2, k)
        sd1 = mean_pairwise_tv(pol1, k)
        mv = tv(p, pol1[0][k][1]) if k in pol1[0] else None
        rows.append({"street": STREETS.get(k.split("|")[0], "?"), "n": len(acts), "visits": visits[k],
                     "entropy": entropy(p), "norm_entropy": entropy(p) / np.log(len(acts)) if len(acts) > 1 else 0.0,
                     "maxp": float(p.max()), "sd2": sd2, "sd1": sd1, "move_1m_2m": mv})
    V = np.array([r["visits"] for r in rows])
    H = np.array([r["norm_entropy"] for r in rows])
    st = np.array([r["street"] for r in rows])
    doc = {"format": "pokeralpha.phase2.policy_geometry/v1", "infosets": len(rows),
           "sources": {"2m": "results/strategy/candidates/native_2m seeds 0-2", "1m": "results/strategy/candidates/native_1m seeds 0-2"},
           "definitions": {"tv": "total variation distance 0.5*L1", "norm_entropy": "entropy / log(#actions)",
                           "seed_disagreement": "mean pairwise TV across the 3 seeds at the same key",
                           "pure": "max action prob >= 0.95"}}
    # entropy by street
    by_street = {}
    for s in ("preflop", "flop", "turn", "river"):
        m = st == s
        if not m.any():
            continue
        mp = np.array([r["maxp"] for r, k in zip(rows, m) if k])
        by_street[s] = {"keys": int(m.sum()), "visit_share": float(V[m].sum() / V.sum()),
                        "norm_entropy_mean": float(H[m].mean()), "norm_entropy_median": float(np.median(H[m])),
                        "pure_share": float((mp >= 0.95).mean()), "visits_median": float(np.median(V[m]))}
    doc["by_street"] = by_street
    # visits distribution
    srt = np.sort(V)[::-1]
    doc["visits"] = {"min": float(V.min()), "median": float(np.median(V)), "mean": float(V.mean()), "max": float(V.max()),
                     "share_below_20": float((V < 20).mean()), "share_below_100": float((V < 100).mean()),
                     "share_of_visits_in_top_1pct_keys": float(srt[: len(srt) // 100].sum() / srt.sum()),
                     "share_of_visits_in_bottom_50pct_keys": float(srt[len(srt) // 2:].sum() / srt.sum()),
                     "hill_tail_index_top5pct": hill_tail_index(V),
                     "deciles": [float(x) for x in np.percentile(V, np.arange(0, 101, 10))]}
    # visits vs entropy
    lv = np.log10(np.maximum(V, 1))
    doc["visits_vs_entropy"] = {"spearman": float(spearmanr(V, H).statistic),
                                "norm_entropy_by_visit_decile": [float(H[(lv >= a) & (lv < b)].mean()) if ((lv >= a) & (lv < b)).any() else None
                                                                 for a, b in zip(np.percentile(lv, np.arange(0, 100, 10)),
                                                                                 list(np.percentile(lv, np.arange(10, 100, 10))) + [99])]}
    # disagreement vs entropy and visits
    ok = np.array([r["sd2"] is not None for r in rows])
    SD = np.array([r["sd2"] if r["sd2"] is not None else np.nan for r in rows])
    doc["disagreement"] = {"keys_in_all_seeds": int(ok.sum()), "mean": float(np.nanmean(SD)),
                           "spearman_vs_entropy": float(spearmanr(SD[ok], H[ok]).statistic),
                           "spearman_vs_visits": float(spearmanr(SD[ok], V[ok]).statistic),
                           "by_street": {s: float(np.nanmean(SD[(st == s) & ok])) for s in by_street}}
    # partial: among well-visited keys (>= 1000 visits), does entropy still predict disagreement?
    hv = ok & (V >= 1000)
    doc["disagreement"]["spearman_vs_entropy_visits_ge_1000"] = float(spearmanr(SD[hv], H[hv]).statistic)
    doc["disagreement"]["mean_visits_ge_1000"] = float(np.nanmean(SD[hv]))
    # training-limited? ratio sd2/sd1 per visit bucket
    both = np.array([r["sd1"] is not None and r["sd2"] is not None for r in rows])
    SD1 = np.array([r["sd1"] if r["sd1"] is not None else np.nan for r in rows])
    ratio = {}
    for lo, hi in ((0, 20), (20, 100), (100, 1000), (1000, 1e4), (1e4, 1e9)):
        m = both & (V >= lo) & (V < hi)
        if m.sum() < 30:
            continue
        a, b = SD[m], SD1[m]
        ratio[f"{int(lo)}-{int(hi) if hi < 1e9 else 'inf'}"] = {
            "keys": int(m.sum()), "sd_1m_mean": float(b.mean()), "sd_2m_mean": float(a.mean()),
            "ratio_of_means": float(a.mean() / b.mean()) if b.mean() > 0 else None,
            "median_per_key_ratio": float(np.median(a[b > 0.02] / b[b > 0.02])) if (b > 0.02).sum() > 10 else None}
    for s in by_street:
        m = both & (st == s)
        ratio[f"street:{s}"] = {"keys": int(m.sum()), "sd_1m_mean": float(SD1[m].mean()), "sd_2m_mean": float(SD[m].mean()),
                                "ratio_of_means": float(SD[m].mean() / SD1[m].mean())}
    doc["training_limited_test"] = {"pure_noise_expectation": 1 / np.sqrt(2), "ratio_2m_over_1m": ratio,
                                    "note": "visits are the 2M run's; the 1M run visited each key ~half as often"}
    # simplex geometry for 3-action keys
    tri = [(pol2[0][k][1]) for k in keys if len(pol2[0][k][0]) == 3]
    tri = np.array(tri)
    if len(tri):
        sorted_p = np.sort(tri, axis=1)
        vertex = sorted_p[:, 2] >= 0.95
        edge = (~vertex) & (sorted_p[:, 0] < 0.02)
        interior = ~(vertex | edge)
        doc["simplex_3action"] = {"keys": int(len(tri)), "vertex_share": float(vertex.mean()), "edge_share": float(edge.mean()),
                                  "interior_share": float(interior.mean()),
                                  "mean_min_action_prob_interior": float(sorted_p[interior, 0].mean()) if interior.any() else None}
    doc["seconds"] = round(time.time() - t0, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("definitions", "sources")}, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
