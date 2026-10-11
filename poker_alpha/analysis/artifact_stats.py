"""Read-only inspection of strategy artifacts: policies, complexity,
microscope (checkpoints × seeds), atlas projection, trust aggregation.

Everything here reads committed ``.npz`` artifacts; nothing is trained,
modified or invented. Deterministic seeds everywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

#: Canonical action-token order used to build fixed-length policy vectors
#: (tokens come from the release tree: fold/call, preflop raise-to
#: multiples, postflop pot fractions, all-in).
ACTION_SLOTS = ("f", "c", "x200", "x250", "x350", "b33", "b75", "b150", "a")

STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


@dataclass
class StrategyTable:
    path: str
    sha256: str
    keys: np.ndarray            # (n,) unicode
    offsets: np.ndarray         # (n+1,)
    actions: np.ndarray         # flat tokens
    probs: np.ndarray           # flat probabilities
    visits: np.ndarray          # (n,)
    index: Dict[str, int]

    def policy(self, key: str) -> Optional[Dict[str, float]]:
        i = self.index.get(key)
        if i is None:
            return None
        s, e = self.offsets[i], self.offsets[i + 1]
        return {str(a): float(p) for a, p in
                zip(self.actions[s:e], self.probs[s:e])}

    def visit_count(self, key: str) -> Optional[float]:
        i = self.index.get(key)
        return None if i is None else float(self.visits[i])


def load_table(path) -> StrategyTable:
    p = _ROOT / path if not Path(path).is_absolute() else Path(path)
    with np.load(p, allow_pickle=False) as z:
        t = StrategyTable(
            path=str(path),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            keys=z["keys"], offsets=z["action_offsets"],
            actions=z["actions"], probs=z["probs"], visits=z["visits"],
            index={})
    t.index = {k: i for i, k in enumerate(t.keys)}
    return t


def street_of(key: str) -> int:
    return int(key.split("|", 1)[0])


# ---------------------------------------------------------------------------
# policy complexity
# ---------------------------------------------------------------------------

def policy_vectors(table: StrategyTable,
                   idx: Sequence[int]) -> np.ndarray:
    """Fixed-length probability vectors over ACTION_SLOTS for the given
    infoset indices (absent actions are 0)."""
    slot = {a: j for j, a in enumerate(ACTION_SLOTS)}
    out = np.zeros((len(idx), len(ACTION_SLOTS)), dtype=np.float64)
    for row, i in enumerate(idx):
        s, e = table.offsets[i], table.offsets[i + 1]
        for a, p in zip(table.actions[s:e], table.probs[s:e]):
            j = slot.get(str(a))
            if j is not None:
                out[row, j] = p
    return out


def entropy_bits(vec: np.ndarray) -> np.ndarray:
    q = np.where(vec > 0, vec, 1.0)
    return -(np.where(vec > 0, vec, 0.0) * np.log2(q)).sum(axis=1)


def complexity_stats(table: StrategyTable, material: float = 0.05) -> dict:
    """Distribution of mixing across ALL stored infosets. NOTE: stored
    abstract infosets are not weighted by how often real hands reach
    them — this describes the table, not play frequency."""
    idx = np.arange(len(table.keys))
    vecs = policy_vectors(table, idx)
    ent = entropy_bits(vecs)
    dom = vecs.max(axis=1)
    used = (vecs >= material).sum(axis=1)
    streets = np.array([street_of(k) for k in table.keys])
    def bucket(mask):
        return {
            "infosets": int(mask.sum()),
            "near_pure_pct": float((dom[mask] >= 0.95).mean() * 100),
            "moderately_mixed_pct": float(
                ((dom[mask] < 0.95) & (dom[mask] >= 0.7)).mean() * 100),
            "highly_mixed_pct": float((dom[mask] < 0.7).mean() * 100),
            "mean_entropy_bits": float(ent[mask].mean()),
            "mean_actions_used": float(used[mask].mean()),
        }
    return {
        "overall": bucket(np.ones(len(idx), bool)),
        "by_street": {STREET_NAMES[s]: bucket(streets == s)
                      for s in sorted(set(streets.tolist()))},
        "material_threshold": material,
        "note": ("stored abstract infosets, not encountered-hand "
                 "frequency"),
    }


# ---------------------------------------------------------------------------
# atlas projection (deterministic PCA via SVD)
# ---------------------------------------------------------------------------

def sample_indices(table: StrategyTable, n: int, seed: int = 0,
                   min_visits: float = 10.0) -> np.ndarray:
    """Deterministic sample, stratified by street, restricted to infosets
    with at least ``min_visits`` training visits (so points describe
    learned behavior, not initialisation noise)."""
    rng = np.random.default_rng(seed)
    streets = np.array([street_of(k) for k in table.keys])
    eligible = np.where(table.visits >= min_visits)[0]
    out = []
    per = max(n // 4, 1)
    for s in (0, 1, 2, 3):
        pool = eligible[streets[eligible] == s]
        take = min(per, len(pool))
        if take:
            out.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=int)


def atlas(table: StrategyTable, confidence=None, n: int = 6000,
          seed: int = 0) -> dict:
    """2D PCA projection of sampled policy vectors.

    Distance in the plot is similarity of ACTION DISTRIBUTIONS under the
    fixed slot basis — not poker-state similarity.
    """
    idx = sample_indices(table, n, seed)
    vecs = policy_vectors(table, idx)
    mean = vecs.mean(axis=0)
    centered = vecs - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    ent = entropy_bits(vecs)
    tops = [ACTION_SLOTS[j] for j in vecs.argmax(axis=1)]
    streets = [street_of(table.keys[i]) for i in idx]
    gate = None
    if confidence is not None:
        gate = [confidence.get(str(table.keys[i]), None) for i in idx]
    return {
        "n": len(idx), "seed": seed, "sample": "stratified by street, "
        f"visits >= 10, rng seed {seed}",
        "explained": "2 leading right-singular vectors of centered policy "
                     "vectors over the fixed action-slot basis",
        "x": np.round(xy[:, 0], 4).tolist(),
        "y": np.round(xy[:, 1], 4).tolist(),
        "top": tops,
        "street": streets,
        "entropy": np.round(ent, 3).tolist(),
        "visits": np.round(table.visits[idx], 0).tolist(),
        "gate": gate,
    }


# ---------------------------------------------------------------------------
# microscope: same infoset across checkpoints × seeds
# ---------------------------------------------------------------------------

MILESTONES = {
    "300k": "results/strategy/candidates/native_300k/holdem_v2_native_seed{seed}_300k.npz",
    "1M": "results/strategy/candidates/native_1m/holdem_v2_native_seed{seed}_1000k.npz",
    "2M": "results/strategy/candidates/native_2m/holdem_v2_native_seed{seed}_2000k.npz",
}
SEEDS = (0, 1, 2)


def microscope_keys(table: StrategyTable, conf_stats: Dict[str, np.ndarray],
                    per_bucket: int = 4) -> List[dict]:
    """Deterministic selection of interesting infosets: well-visited, mixed,
    and spanning low/medium/high cross-seed disagreement."""
    ent = entropy_bits(policy_vectors(table, np.arange(len(table.keys))))
    cands = []
    for i, k in enumerate(table.keys):
        st = conf_stats.get(str(k))
        if st is None or table.visits[i] < 200 or ent[i] < 0.3:
            continue
        sd = float(st[2])
        if np.isnan(sd):
            continue
        cands.append((str(k), float(table.visits[i]), float(ent[i]), sd))
    cands.sort(key=lambda t: (-t[1], t[0]))
    cands = cands[:3000]
    cands.sort(key=lambda t: t[3])
    n = len(cands)
    picks = []
    for name, sl in (("low", slice(0, per_bucket)),
                     ("medium", slice(n // 2 - per_bucket // 2,
                                      n // 2 - per_bucket // 2 + per_bucket)),
                     ("high", slice(n - per_bucket, n))):
        for key, v, e, sd in cands[sl]:
            picks.append({"key": key, "visits": v, "entropy": round(e, 3),
                          "seed_disagreement": round(sd, 4),
                          "bucket": name})
    return picks


def microscope(keys: Sequence[str]) -> dict:
    """Policy of each key across every available milestone × seed."""
    out: Dict[str, dict] = {k: {} for k in keys}
    artifacts = {}
    for label, pattern in MILESTONES.items():
        for seed in SEEDS:
            p = _ROOT / pattern.format(seed=seed)
            if not p.exists():
                continue
            t = load_table(pattern.format(seed=seed))
            artifacts[f"{label}/seed{seed}"] = t.sha256[:12]
            for k in keys:
                pol = t.policy(k)
                if pol is not None:
                    out[k][f"{label}/seed{seed}"] = {
                        "mix": {a: round(p, 4) for a, p in pol.items()},
                        "visits": t.visit_count(k)}
    return {"series": out, "artifacts": artifacts,
            "milestones": list(MILESTONES), "seeds": list(SEEDS)}


def load_confidence_stats(path="results/strategy/holdem_v2_native_seed0_2m_confidence.npz"):
    p = _ROOT / path
    with np.load(p, allow_pickle=False) as z:
        keys = z["keys"]
        stats = z["stats"]   # columns: visits, movement, seed_disagreement, collision
    return {str(k): stats[i] for i, k in enumerate(keys)}


__all__ = ["ACTION_SLOTS", "StrategyTable", "load_table", "street_of",
           "policy_vectors", "entropy_bits", "complexity_stats",
           "sample_indices", "atlas", "microscope", "microscope_keys",
           "load_confidence_stats", "MILESTONES", "SEEDS"]
