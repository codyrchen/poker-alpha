"""Canonical, platform-stable digests of strategy profiles.

A strategy digest is a SHA-256 over an explicit binary encoding of a strategy
``{infoset_key: {action: probability}}``. It exists so reproducibility tests
can pin "this exact training run produced this exact strategy" without
depending on Python's ``repr``, dictionary iteration order, or float printing.

Encoding (version 1)
--------------------
::

    b"PASD" + u8 version
    u64 number_of_infosets
    for each infoset, sorted by its UTF-8 key bytes:
        u32 len(key) + key bytes
        u32 number_of_actions
        for each action, sorted by its UTF-8 bytes:
            u32 len(action) + action bytes
            i64 quantized probability   (quantized mode), or
            f64 probability             (exact mode)

All integers are little-endian. ``-0.0`` is normalized to ``0.0``; NaN and
infinities are rejected.

Two modes:

* ``quantum_bits=None`` — *exact*: the raw IEEE-754 bits of each probability.
  Bit-exact reproducibility on one platform; may differ across BLAS/SIMD
  builds by an ULP.
* ``quantum_bits=k`` (default 40) — each probability is rounded to the nearest
  multiple of ``2**-k`` before hashing. This absorbs last-ULP noise while still
  detecting any meaningful (> ~1e-12) change in a strategy.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Mapping, Optional

DIGEST_FORMAT_VERSION = 1
DEFAULT_QUANTUM_BITS = 40
_MAGIC = b"PASD"


def _encode_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _encode_prob(p: float, quantum_bits: Optional[int]) -> bytes:
    p = float(p)
    if not math.isfinite(p):
        raise ValueError(f"non-finite probability in strategy: {p!r}")
    if p == 0.0:
        p = 0.0  # normalizes -0.0
    if quantum_bits is None:
        return struct.pack("<d", p)
    return struct.pack("<q", int(round(p * (1 << quantum_bits))))


def canonical_strategy_bytes(
    strategy: Mapping[str, Mapping[str, float]],
    quantum_bits: Optional[int] = DEFAULT_QUANTUM_BITS,
) -> bytes:
    """The canonical binary encoding described in the module docstring."""
    if quantum_bits is not None and not 0 < quantum_bits <= 52:
        raise ValueError("quantum_bits must be in 1..52 or None")
    parts = [_MAGIC, struct.pack("<B", DIGEST_FORMAT_VERSION),
             struct.pack("<Q", len(strategy))]
    for key in sorted(strategy, key=lambda k: k.encode("utf-8")):
        probs = strategy[key]
        parts.append(_encode_str(key))
        parts.append(struct.pack("<I", len(probs)))
        for action in sorted(probs, key=lambda a: a.encode("utf-8")):
            parts.append(_encode_str(action))
            parts.append(_encode_prob(probs[action], quantum_bits))
    return b"".join(parts)


def strategy_digest(
    strategy: Mapping[str, Mapping[str, float]],
    quantum_bits: Optional[int] = DEFAULT_QUANTUM_BITS,
) -> str:
    """Hex SHA-256 of :func:`canonical_strategy_bytes`."""
    return hashlib.sha256(
        canonical_strategy_bytes(strategy, quantum_bits)).hexdigest()
