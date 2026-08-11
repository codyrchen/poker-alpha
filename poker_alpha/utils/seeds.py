"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python's ``random`` and NumPy's legacy global RNG, and return a fresh
    :class:`numpy.random.Generator` seeded the same way.

    Prefer the returned generator for new code; the globals are seeded too so any
    library relying on them is reproducible under the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
