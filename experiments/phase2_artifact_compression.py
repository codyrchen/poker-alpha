"""Phase II Program 8 (+17): strategy-artifact engineering on the release.

Measures, on results/strategy/holdem_v2_native_seed0_2m.npz (never modified):
  baseline   file size, load time (load_artifact, median of 5), RSS delta,
             lookup throughput (dict lookups of random stored keys)
  float16 / uint16 / uint8 quantized probabilities (renormalized per infoset):
             size (compressed npz), max and mean TV vs the float32 release,
             top-action flips
  sparse     share of action probabilities below 1e-3 / 1e-2, and the TV cost
             of dropping them
  mmap       feasibility of an uncompressed, memory-mappable layout: size and
             time-to-first-lookup
  distill    interpretable approximations (approximation only, no replacement):
             (a) betting-context prototype: mean policy over all card buckets
                 sharing street/player/betting/legal-mask (card-blind),
             (b) card-strength-only prototype per street and legal mask,
             reach(visit)-weighted TV and top-action agreement vs release.
  value      seed-to-seed TV (training noise floor) for scale; exact EV impact
             of quantization is not computable in the full game (documented)

    python experiments/phase2_artifact_compression.py
"""

from __future__ import annotations

import io
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poker_alpha.solvers.strategy_artifact import load_artifact  # noqa: E402

REL = ROOT / "results" / "strategy" / "holdem_v2_native_seed0_2m.npz"
SIB = ROOT / "results" / "strategy" / "candidates" / "native_2m" / "holdem_v2_native_seed1_2000k.npz"
OUT = ROOT / "results" / "phase2" / "artifact_compression.json"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def segments(off):
    return np.repeat(np.arange(len(off) - 1), np.diff(off))


def seg_sum(x, seg, n):
    return np.bincount(seg, weights=x, minlength=n)


def tv_per_infoset(p, q, seg, n):
    return 0.5 * seg_sum(np.abs(p - q), seg, n)


def top_action(p, off):
    return np.array([int(np.argmax(p[off[i]:off[i + 1]])) for i in range(len(off) - 1)])


def renorm(q, seg, n):
    s = seg_sum(q, seg, n)
    return q / np.where(s[seg] > 0, s[seg], 1.0)


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.tell()


def main():
    t_all = time.time()
    z = np.load(REL, allow_pickle=False)
    keys, off, acts = z["keys"], z["action_offsets"], z["actions"]
    p = z["probs"].astype(np.float64)
    visits = z["visits"].astype(np.float64)
    n = len(keys)
    seg = segments(off)
    p = renorm(p, seg, n)
    w = visits / visits.sum()
    doc = {"format": "pokeralpha.phase2.artifact_compression/v1", "artifact": str(REL.relative_to(ROOT)),
           "infosets": int(n), "action_entries": int(len(p))}
    # ---- baseline -------------------------------------------------------
    times = []
    r0 = rss_mb()
    for _ in range(5):
        t = time.perf_counter()
        art = load_artifact(REL)
        times.append(time.perf_counter() - t)
    rss_after = rss_mb()
    rng = np.random.default_rng(0)
    sample = [str(k) for k in keys[rng.integers(0, n, 200000)]]
    t = time.perf_counter()
    hits = sum(1 for k in sample if art.lookup(k) is not None)
    lk = time.perf_counter() - t
    doc["baseline"] = {"file_bytes": REL.stat().st_size, "load_seconds_median": statistics.median(times),
                       "load_seconds_all": times, "peak_rss_mb_before": r0, "peak_rss_mb_after_load": rss_after,
                       "lookups_per_second": len(sample) / lk, "hits": hits,
                       "probs_dtype": str(z["probs"].dtype)}
    base_arrays = {k: z[k] for k in z.files}
    # ---- quantization ---------------------------------------------------
    top0 = top_action(p, off)
    quant = {}
    for name, enc in (("float16", lambda x: x.astype(np.float16)),
                      ("uint16", lambda x: np.round(x * 65535).astype(np.uint16)),
                      ("uint8", lambda x: np.round(x * 255).astype(np.uint8))):
        q_raw = enc(p)
        q = q_raw.astype(np.float64)
        if name.startswith("uint"):
            q = q / (65535.0 if name == "uint16" else 255.0)
        q = renorm(np.maximum(q, 0), seg, n)
        tvs = tv_per_infoset(p, q, seg, n)
        flips = int((top_action(q, off) != top0).sum())
        arrays = dict(base_arrays, probs=q_raw)
        quant[name] = {"npz_compressed_bytes": npz_bytes(**arrays),
                       "probs_raw_bytes": int(q_raw.nbytes),
                       "tv_max": float(tvs.max()), "tv_mean": float(tvs.mean()), "tv_visit_weighted": float((w * tvs).sum()),
                       "top_action_flips": flips}
    quant["float32_release"] = {"npz_compressed_bytes": npz_bytes(**base_arrays), "probs_raw_bytes": int(z["probs"].nbytes)}
    doc["quantization"] = quant
    # ---- sparsity -------------------------------------------------------
    sp = {}
    for thr in (1e-3, 1e-2):
        small = p < thr
        q = renorm(np.where(small, 0.0, p), seg, n)
        tvs = tv_per_infoset(p, q, seg, n)
        sp[str(thr)] = {"share_entries_below": float(small.mean()), "tv_max": float(tvs.max()),
                        "tv_visit_weighted": float((w * tvs).sum()), "top_action_flips": int((top_action(q, off) != top0).sum())}
    doc["sparsity"] = sp
    # ---- mmap feasibility -----------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "flat.npz"
        np.savez(path, probs=z["probs"], action_offsets=off, visits=z["visits"])
        t = time.perf_counter()
        mm = np.load(path, mmap_mode="r")
        first = float(mm["probs"][off[n // 2]])
        tfirst = time.perf_counter() - t
        doc["mmap"] = {"uncompressed_numeric_bytes": path.stat().st_size, "time_to_first_lookup_seconds": tfirst,
                       "note": ("np.load(mmap_mode) does not memory-map members of an .npz archive (they are read on access); "
                                "true mmap needs raw .npy files or a custom binary layout. The 144k string keys still need a "
                                "hash index (dict build dominates load time); a sorted fixed-width key array with binary "
                                "search would make the whole artifact mmap-able."),
                       "probe": first}
    # dict build cost vs numeric load
    t = time.perf_counter()
    idx = {str(k): i for i, k in enumerate(keys)}
    doc["mmap"]["key_index_build_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    sk = np.sort(keys)
    pos = np.searchsorted(sk, np.array(sample[:20000]))
    doc["mmap"]["sorted_key_binary_search_lookups_per_second"] = 20000 / (time.perf_counter() - t)
    del idx, pos
    # ---- distillation prototypes ------------------------------------------
    parts = [str(k).split("|") for k in keys]

    def proto(group_of):
        groups = defaultdict(list)
        for i, pr in enumerate(parts):
            groups[group_of(pr)].append(i)
        q = np.empty_like(p)
        for g, members in groups.items():
            members = np.array(members)
            ww = visits[members] + 1e-9
            L = off[members[0] + 1] - off[members[0]]
            mat = np.stack([p[off[i]:off[i] + L] for i in members])
            mean = (ww[:, None] * mat).sum(0) / ww.sum()
            for i in members:
                q[off[i]:off[i] + L] = mean
        tvs = tv_per_infoset(p, q, seg, n)
        agree = top_action(q, off) == top0
        return {"prototypes": len(groups), "compression_ratio_infosets": n / len(groups),
                "tv_visit_weighted": float((w * tvs).sum()), "tv_mean": float(tvs.mean()),
                "top_action_agreement_visit_weighted": float((w * agree).sum()),
                "top_action_agreement": float(agree.mean())}
    # key = street|player|card|betting|legal (card part may contain '|texture')
    doc["distillation"] = {
        "card_blind_betting_prototype": proto(lambda pr: (pr[0], pr[1], pr[-2], pr[-1])),
        "strength_only_prototype": proto(lambda pr: (pr[0], pr[1], pr[2][:2] if pr[0] != "0" else pr[2], pr[-1])),
        "card_without_texture_prototype": proto(lambda pr: (pr[0], pr[1], pr[2], pr[-2], pr[-1])),
        "label": "approximation only; prototypes are visit-weighted mean policies of the release itself"}
    # ---- noise floor: seed-to-seed TV ------------------------------------
    zs = np.load(SIB, allow_pickle=False)
    sib = {str(k): i for i, k in enumerate(zs["keys"])}
    so, sprob = zs["action_offsets"], zs["probs"].astype(np.float64)
    tvs, ws = [], []
    for i, k in enumerate(keys):
        j = sib.get(str(k))
        if j is None:
            continue
        a = p[off[i]:off[i + 1]]
        b = sprob[so[j]:so[j + 1]]
        b = b / b.sum() if b.sum() > 0 else b
        if len(a) == len(b):
            tvs.append(0.5 * np.abs(a - b).sum())
            ws.append(visits[i])
    tvs, ws = np.array(tvs), np.array(ws)
    doc["noise_floor_seed0_vs_seed1"] = {"tv_visit_weighted": float((ws * tvs).sum() / ws.sum()), "tv_mean": float(tvs.mean())}
    doc["value_effect"] = ("exact EV/exploitability effect of quantization is not computable for the full game "
                           "(no exact best response); TV is compared with the seed-to-seed training noise floor instead")
    doc["seconds"] = round(time.time() - t_all, 1)
    doc["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps(doc, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
