"""Small matplotlib helpers so experiments produce consistent figures.

Uses the non-interactive ``Agg`` backend so figures render in headless runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
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


def save_grouped_bar_chart(
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    ylabel: str,
    title: str,
    path: "str | Path",
    errors: Optional[Dict[str, Sequence[Tuple[float, float]]]] = None,
) -> Path:
    """Save a grouped bar chart: one bar per (category, series) pair.

    ``series`` maps a series label to one value per category. ``errors``, if
    given, maps the same labels to per-category ``(lo, hi)`` interval bounds
    (absolute, not offsets) rendered as asymmetric error bars.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(series)
    n_series = len(labels)
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.8 / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * n_cat), 4.5))
    for i, label in enumerate(labels):
        values = np.asarray(series[label], dtype=float)
        offset = (i - (n_series - 1) / 2.0) * width
        yerr = None
        if errors is not None and label in errors:
            lo, hi = zip(*errors[label])
            yerr = np.abs(np.vstack([values - np.asarray(lo),
                                     np.asarray(hi) - values]))
        ax.bar(x + offset, values, width=width, label=label, yerr=yerr,
               capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
