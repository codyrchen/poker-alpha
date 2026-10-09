"""Compact, versioned export of a trained average strategy (Phase 29).

A strategy artifact holds only what decision support needs — the average
strategy and visit count of each information set — not the regret tables
needed to resume training, so it is several times smaller than a
checkpoint. It is an uncompressed-header, zlib-compressed ``.npz`` read with
``allow_pickle=False``.

==================  =======================================================
``format``          ``"pokeralpha.strategy_artifact/v1"``
``config_sig``      ``HoldemSolverConfig`` signature it was trained under
``game_signature``  :meth:`HoldemGame.signature`
``encoder_sig``     encoder signature
``meta``            JSON: seeds, iterations, source checkpoints, notes
``keys``            infoset keys (sorted by UTF-8 bytes)
``action_offsets``  int64, len = n + 1
``actions``         action tokens, concatenated
``probs``           float32 average-strategy probabilities, concatenated
``visits``          float32 per infoset (non-updating-player visits)
==================  =======================================================

Format v2 (written since Phase 55) adds

==================  =======================================================
``config_json``     the full ``HoldemSolverConfig.to_dict()`` (version, encoder,
                    action abstraction, stack, blinds, raise cap, averaging,
                    sampler) — its signature must equal ``config_sig``
``commit``          PokerAlpha commit that generated the artifact ("" if unknown)
``content_sha256``  SHA-256 over keys / offsets / actions / probs / visits
==================  =======================================================

Loading checks the format, the content checksum (v2), and that the game it
is used with has the same solver config, game and encoder signatures. v1
artifacts stay loadable; a ``<name>.manifest.json`` sidecar (written for the
committed v1 artifacts) supplies their missing metadata, and if present the
file must match the SHA-256 it records. There is no fallback to a
mismatched strategy: every mismatch raises :class:`StrategyArtifactError`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

FORMAT_V1 = "pokeralpha.strategy_artifact/v1"
FORMAT = "pokeralpha.strategy_artifact/v2"
FORMATS = (FORMAT_V1, FORMAT)
MANIFEST_FORMAT = "pokeralpha.strategy_artifact_manifest/v1"
PathLike = Union[str, os.PathLike]


class StrategyArtifactError(Exception):
    """Malformed or incompatible strategy artifact."""


@dataclass
class StrategyArtifact:
    config_signature: str
    game_signature: str
    encoder_signature: str
    meta: dict
    strategy: Dict[str, Dict[str, float]]
    visits: Dict[str, float]

    def lookup(self, key: str) -> Optional[Tuple[Dict[str, float], float]]:
        p = self.strategy.get(key)
        return None if p is None else (p, self.visits.get(key, 0.0))

    def __len__(self) -> int:
        return len(self.strategy)

    format: str = FORMAT
    config: Optional[dict] = None          # full config (v2 / manifest)
    commit: Optional[str] = None
    content_sha256: Optional[str] = None
    manifest: Optional[dict] = None


def _content_hash(keys, offsets, actions, probs, visits) -> str:
    h = hashlib.sha256()
    for part in ("\x00".join(keys).encode(), np.asarray(offsets, np.int64).tobytes(),
                 "\x00".join(actions).encode(), np.asarray(probs, np.float32).tobytes(),
                 np.asarray(visits, np.float32).tobytes()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()


def manifest_path(path: PathLike) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ".manifest.json")


def export_solver(solver, path: PathLike, meta: Optional[dict] = None,
                  min_visits: float = 0.0) -> Path:
    """Write ``solver``'s average strategy (infosets with visits > ``min_visits``
    or, when it is 0, every infoset with any visit) to ``path``."""
    game = solver.game
    keys = []
    for k, n in solver.infosets.items():
        v = float(n.strategy_sum.sum())
        if v > 0 and v >= min_visits:
            keys.append(k)
    keys.sort(key=lambda k: k.encode("utf-8"))
    offsets, actions, probs, visits = [0], [], [], []
    for k in keys:
        n = solver.infosets[k]
        actions.extend(n.actions)
        probs.append(n.average_strategy().astype(np.float32))
        visits.append(float(n.strategy_sum.sum()))
        offsets.append(offsets[-1] + len(n.actions))
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    m = dict(meta or {})
    m.setdefault("iterations", int(solver.iterations))
    m.setdefault("sampler", type(solver).__name__)
    m["min_visits_exported"] = min_visits
    cfg = getattr(game, "solver_config", None)
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True) if cfg is not None else ""
    from ..utils.provenance import git_commit

    probs_a = np.concatenate(probs) if probs else np.zeros(0, np.float32)
    visits_a = np.array(visits, dtype=np.float32)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, format=np.array(FORMAT), config_sig=np.array(game.solver_config_signature()),
            game_signature=np.array(game.signature()),
            encoder_sig=np.array(game.encoder_signature() or ""),
            meta=np.array(json.dumps(m, sort_keys=True)),
            config_json=np.array(cfg_json), commit=np.array(git_commit() or ""),
            content_sha256=np.array(_content_hash(keys, offsets, actions, probs_a, visits_a)),
            keys=np.array(keys, dtype=np.str_),
            action_offsets=np.array(offsets, dtype=np.int64),
            actions=np.array(actions, dtype=np.str_),
            probs=probs_a, visits=visits_a)
    os.replace(tmp, path)
    return path


def load_artifact(path: PathLike, game=None) -> StrategyArtifact:
    """Read an artifact; if ``game`` is given, require matching signatures."""
    manifest = None
    mp = manifest_path(path)
    if mp.exists():
        from ..utils.provenance import file_sha256

        try:
            manifest = json.loads(mp.read_text())
        except ValueError as exc:
            raise StrategyArtifactError(f"unreadable manifest {mp}: {exc}") from exc
        if manifest.get("format") != MANIFEST_FORMAT:
            raise StrategyArtifactError(f"{mp}: not a strategy artifact manifest")
        if manifest.get("sha256") != file_sha256(path):
            raise StrategyArtifactError(
                f"{path} does not match its manifest checksum (corrupt, truncated or "
                "replaced): refusing to use it")
    try:
        with np.load(Path(path), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError) as exc:
        raise StrategyArtifactError(f"cannot read {path}: {exc}") from exc
    except Exception as exc:  # zipfile.BadZipFile / zlib.error on truncated files
        raise StrategyArtifactError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    fmt = str(d.get("format", np.array(""))[()])
    if fmt not in FORMATS:
        raise StrategyArtifactError("not a pokeralpha strategy artifact (v1 / v2)")
    art_cfg = str(d["config_sig"][()])
    art_game = str(d["game_signature"][()])
    art_enc = str(d["encoder_sig"][()])
    if game is not None:
        for what, mine, theirs in (("solver config", art_cfg, game.solver_config_signature()),
                                   ("game", art_game, game.signature()),
                                   ("encoder", art_enc, game.encoder_signature() or "")):
            if mine != theirs:
                raise StrategyArtifactError(
                    f"{what} signature mismatch: artifact {mine!r} vs game {theirs!r}")
    keys = [str(k) for k in d["keys"]]
    off = d["action_offsets"]
    acts = [str(a) for a in d["actions"]]
    probs, visits = d["probs"], d["visits"]
    if len(off) != len(keys) + 1 or int(off[-1]) != len(acts) or len(probs) != len(acts) \
            or len(visits) != len(keys):
        raise StrategyArtifactError("inconsistent artifact layout")
    config, commit, content = None, None, None
    if fmt == FORMAT:
        content = str(d["content_sha256"][()])
        if _content_hash(keys, off, acts, probs, visits) != content:
            raise StrategyArtifactError("artifact content checksum mismatch (corrupt)")
        cj = str(d["config_json"][()])
        if cj:
            config = json.loads(cj)
            # HoldemSolverConfig.signature() hashes exactly this dict
            blob = json.dumps(config, sort_keys=True).encode()
            sig = (f"HoldemSolverConfig:v{config.get('config_version')}:"
                   f"{hashlib.sha256(blob).hexdigest()[:20]}")
            if sig != art_cfg:
                raise StrategyArtifactError("artifact config does not match its signature")
        commit = str(d["commit"][()]) or None
    elif manifest is not None:
        config, commit = manifest.get("config"), manifest.get("generation_commit")
    strategy, vis = {}, {}
    for i, k in enumerate(keys):
        lo, hi = int(off[i]), int(off[i + 1])
        strategy[k] = {a: float(p) for a, p in zip(acts[lo:hi], probs[lo:hi])}
        vis[k] = float(visits[i])
    return StrategyArtifact(art_cfg, art_game, art_enc, json.loads(str(d["meta"][()])),
                            strategy, vis, fmt, config, commit, content, manifest)
