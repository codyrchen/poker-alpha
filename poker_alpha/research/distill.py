"""Strategy compiler: shallow, interpretable approximations of a tabular policy
(Phase II, Program 17). APPROXIMATION ONLY - never a replacement for the
release strategy.

A weighted CART regression tree whose leaves hold action *distributions*:
splits minimize the weighted sum of squared errors of the probability
vectors (sum over actions), weights are information-set visit counts.
Features are small ordinal encodings parsed from the information-set key.
Pure numpy; no new dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RANKS = "23456789TJQKA"
FACING = {"none": 0, "small": 1, "medium": 2, "large": 3, "over": 4, "allin": 5}
PRIOR = {"n": 0, "c": 1, "a": 2, "f": 3}
INIT = {"n": 0, "o": 1, "p": 2}   # none / own / opp (first letter)
TEXTURE_PAIR = {"u": 0, "p": 1}
TEXTURE_SUIT = {"r": 0, "t": 1, "f": 2}
TEXTURE_CONN = {"d": 0, "c": 1}
_BET = re.compile(r"^i([nop])r(\d)f([a-z]+)a([ncaf])s(\d)$")

HOLDEM_FEATURES = ["street", "player", "hi_rank", "lo_rank", "suited", "pair", "strength", "pct", "draw", "nut",
                   "blocker", "tex_pair", "tex_suit", "tex_conn", "initiative", "raises", "facing", "own_prior", "spr"]


def holdem_features(key: str) -> Optional[List[float]]:
    """Ordinal features of a compact-v2 release key; None if unparseable.
    Missing features (e.g. strength preflop) are -1."""
    parts = key.split("|")
    try:
        street, player = int(parts[0]), int(parts[1])
        bet = _BET.match(parts[-2])
        if bet is None:
            return None
        f = dict.fromkeys(HOLDEM_FEATURES, -1.0)
        f["street"], f["player"] = street, player
        if street == 0:
            c = parts[2]
            f["hi_rank"], f["lo_rank"] = RANKS.index(c[0]), RANKS.index(c[1])
            f["pair"] = float(c[0] == c[1])
            f["suited"] = float(c.endswith("s"))
        else:
            card, tex = parts[2], parts[3]
            if street == 3:
                m = re.match(r"^p(\d+)(\d)(\d)b(\d)$", card)
                f["pct"], f["draw"], f["nut"], f["blocker"] = map(float, m.groups())
            else:
                f["strength"], f["draw"], f["nut"] = float(card[0]), float(card[1]), float(card[2])
            f["tex_pair"], f["tex_suit"], f["tex_conn"] = TEXTURE_PAIR[tex[0]], TEXTURE_SUIT[tex[1]], TEXTURE_CONN[tex[2]]
        i, r, fac, ap, spr = bet.groups()
        f["initiative"], f["raises"], f["facing"], f["own_prior"], f["spr"] = INIT[i], int(r), FACING[fac], PRIOR[ap], int(spr)
        return [float(f[n]) for n in HOLDEM_FEATURES]
    except (ValueError, KeyError, IndexError, AttributeError):
        return None


@dataclass
class Node:
    value: np.ndarray
    weight: float
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


@dataclass
class DistributionTree:
    max_depth: int = 4
    min_leaf_weight: float = 1.0
    feature_names: Sequence[str] = ()
    root: Optional[Node] = None
    leaves: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> "DistributionTree":
        self.leaves = 0
        self.root = self._grow(X, Y, w, 0)
        return self

    def _grow(self, X, Y, w, depth) -> Node:
        W = w.sum()
        mean = (w[:, None] * Y).sum(0) / W
        node = Node(mean, float(W))
        if depth >= self.max_depth or len(w) < 2:
            self.leaves += 1
            return node
        best = None
        base = float((w[:, None] * (Y - mean) ** 2).sum())
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            ws, ys = w[order], Y[order]
            cw = np.cumsum(ws)
            cy = np.cumsum(ws[:, None] * ys, axis=0)
            cyy = np.cumsum(ws[:, None] * ys * ys, axis=0)
            # candidate split after position i where xs[i] < xs[i+1]
            idx = np.flatnonzero(xs[:-1] < xs[1:])
            if len(idx) == 0:
                continue
            lw = cw[idx]
            rw = W - lw
            ok = (lw >= self.min_leaf_weight) & (rw >= self.min_leaf_weight)
            if not ok.any():
                continue
            ly, lyy = cy[idx], cyy[idx]
            ry, ryy = cy[-1] - ly, cyy[-1] - lyy
            sse = (lyy - ly ** 2 / lw[:, None]).sum(1) + (ryy - ry ** 2 / rw[:, None]).sum(1)
            sse[~ok] = np.inf
            k = int(np.argmin(sse))
            if best is None or sse[k] < best[0]:
                best = (float(sse[k]), j, float((xs[idx[k]] + xs[idx[k] + 1]) / 2))
        if best is None or best[0] >= base - 1e-12:
            self.leaves += 1
            return node
        _, j, thr = best
        m = X[:, j] <= thr
        node.feature, node.threshold = j, thr
        node.left = self._grow(X[m], Y[m], w[m], depth + 1)
        node.right = self._grow(X[~m], Y[~m], w[~m], depth + 1)
        return node

    def predict_one(self, x) -> np.ndarray:
        n = self.root
        while not n.is_leaf:
            n = n.left if x[n.feature] <= n.threshold else n.right
        return n.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.predict_one(x) for x in X])

    def rules(self, actions: Sequence[str], max_rules: int = 12) -> List[str]:
        """Human-readable leaf rules, heaviest first."""
        out: List[Tuple[float, str]] = []

        def walk(n, conds):
            if n.is_leaf:
                top = sorted(zip(actions, n.value), key=lambda t: -t[1])[:3]
                mix = ", ".join(f"{a} {p:.2f}" for a, p in top)
                out.append((n.weight, f"IF {' AND '.join(conds) or 'always'} THEN {mix}"))
                return
            name = self.feature_names[n.feature] if self.feature_names else f"x{n.feature}"
            walk(n.left, conds + [f"{name} <= {n.threshold:g}"])
            walk(n.right, conds + [f"{name} > {n.threshold:g}"])
        walk(self.root, [])
        tot = sum(w for w, _ in out)
        return [f"[{100 * w / tot:4.1f}% of weight] {r}" for w, r in sorted(out, key=lambda t: -t[0])[:max_rules]]
