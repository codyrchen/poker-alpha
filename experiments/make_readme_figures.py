"""Regenerate the README headline figures from committed result data.

    python experiments/make_readme_figures.py

Outputs (committed):
  results/figures/native_speedup.png       <- results/benchmarks/native_mccfr_v1.json
  results/figures/confidence_risk_coverage.png
                                           <- results/validation/confidence_signal_quality.json

No number is hand-entered; both figures are rebuilt entirely from the
committed benchmark / calibration JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"

# Validated palette (dataviz reference instance; see commit message).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6e6d66"
SURFACE = "#ffffff"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="x", color="#e7e6e2", linewidth=0.8)
    ax.set_axisbelow(True)


def native_speedup():
    data = json.loads((ROOT / "results/benchmarks/native_mccfr_v1.json").read_text())
    py = data["python_warm_it_per_s_median"]
    nat = data["native_warm_it_per_s_median"]
    speedup = data["speedup_warm"]

    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)

    labels = ["Python reference", "Native C++ (pybind11)"]
    values = [py, nat]
    colors = [ORANGE, BLUE]
    bars = ax.barh(labels, values, color=colors, height=0.52, zorder=3)
    texts = [f"{py:,.0f} it/s", f"{nat:,.0f} it/s  ({speedup:.1f}× faster)"]
    for bar, v, t in zip(bars, values, texts):
        ax.text(v + 12, bar.get_y() + bar.get_height() / 2, t,
                va="center", ha="left", fontsize=10, color=INK,
                fontweight="bold")
    ax.set_xlim(0, nat * 1.32)
    ax.invert_yaxis()
    ax.set_xlabel("external-sampling MCCFR iterations / second "
                  "(release config, warm tables, median of 3 trials)",
                  fontsize=8.5, color=MUTED)
    ax.set_title("Hold'em MCCFR training throughput — Python vs native backend",
                 fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    out = FIG / "native_speedup.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out, f"(py={py}, native={nat}, {speedup}x)")


def risk_coverage():
    data = json.loads((ROOT / "results/validation/confidence_signal_quality.json").read_text())
    rc = data["risk_coverage"]
    base = rc["_all_rows"]["mean_regret"]
    v1 = data["v1_baseline"]["SOLVER_ACCEPT"]
    v2 = data["v2_candidates"]["sd<0.2|mv<0.1|v>=20"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)

    for sig, color, label in (("seed_disagreement", BLUE, "rank by seed disagreement"),
                              ("visits", ORANGE, "rank by visit count")):
        pts = sorted((float(c), v["mean_regret"]) for c, v in rc[sig].items())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                zorder=3)
        li = 3 if sig == "seed_disagreement" else 4
        ax.annotate(label, (xs[li], ys[li]),
                    textcoords="offset points",
                    xytext=(8, -18 if sig == "visits" else 8),
                    ha="left", fontsize=9, color=color, fontweight="bold")

    ax.axhline(base, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"accept everything: {base:.2f} bb mean regret",
                (0.015, base), textcoords="offset points", xytext=(0, -14),
                fontsize=8.5, color=MUTED)

    for cov, reg in ((v1["share"], v1["mean_regret"]),
                     (v2["coverage"], v2["mean_regret"])):
        ax.scatter([cov], [reg], s=110, marker="D", color=INK, zorder=4,
                   edgecolors=SURFACE, linewidths=1.5)
    ax.annotate(f"v1 gate (10k→final movement): "
                f"{v1['share']:.0%} coverage, {v1['mean_regret']:.2f} bb",
                (v1["share"], v1["mean_regret"]),
                xytext=(0.012, 0.035), textcoords=("data", "data"),
                ha="left", va="top", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=4))
    ax.annotate(f"v2 gate (recent movement): "
                f"{v2['coverage']:.0%} coverage, {v2['mean_regret']:.2f} bb"
                f" — same risk, {v2['coverage'] / v1['share']:.1f}× the coverage",
                (v2["coverage"], v2["mean_regret"]),
                xytext=(0.28, 0.12), textcoords=("data", "data"),
                ha="left", va="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=4))

    ax.set_xlabel("coverage — fraction of infosets the gate ACCEPTs", fontsize=9.5, color=INK)
    ax.set_ylabel("mean EV regret of accepted infosets (bb)", fontsize=9.5, color=INK)
    ax.set_title("Abstention calibrated against exact-game EV regret\n"
                 "(held-out: 8,880 infosets from 9 exactly solved games)",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(-0.04, base * 1.3)
    fig.tight_layout()
    out = FIG / "confidence_risk_coverage.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    native_speedup()
    risk_coverage()
