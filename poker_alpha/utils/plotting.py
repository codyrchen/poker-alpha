"""Small matplotlib helpers so experiments produce consistent figures.

Uses the non-interactive ``Agg`` backend so figures render in headless runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_convergence_plot(
    series: Iterable[Tuple[str, Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: "str | Path",
    logx: bool = False,
    logy: bool = False,
) -> Path:
    """Save a labelled line plot with one or more ``(label, xs, ys)`` series."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, xs, ys in series:
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
