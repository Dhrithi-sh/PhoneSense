"""Report figures: precision-recall curves and parameter heatmaps (T11, T05).

Figures are written as PNG (200 dpi) and PDF to ``paths.figures_dir`` (``report/figures/``)
with one shared style, so every figure in the report looks the same.

Adapted from our midterm project Dhvani-lingual (``dhvani/eval/experiments.py``:
``plot_pr_curves``); the heatmap is new.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phonesense.eval.metrics import (  # noqa: E402
    RECALL_LEVELS,
    interpolated_precision_11pt,
    relevant_set,
)

STYLE = {
    "figure.figsize": (5.5, 4.0),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _out_dir(out_dir: str | Path | None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    from phonesense.config import get_path

    return get_path("figures_dir")


def save(fig: plt.Figure, name: str, out_dir: str | Path | None = None) -> list[Path]:
    """Save ``fig`` as ``<out_dir>/<name>.png`` and ``.pdf``; close it; return the paths."""
    d = _out_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{name}.png", d / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def mean_11pt(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    qids: Sequence[str] | None = None,
    threshold: int | None = None,
) -> list[float]:
    """11-point interpolated precision averaged over queries that have a relevant doc."""
    qs = [
        q
        for q in (qids if qids is not None else qrels)
        if relevant_set(qrels.get(q, {}), threshold)
    ]
    if not qs:
        return [0.0] * len(RECALL_LEVELS)
    curves = np.array(
        [interpolated_precision_11pt(list(rankings.get(q, [])), qrels[q], threshold) for q in qs]
    )
    return curves.mean(axis=0).tolist()


def pr_curve(
    runs: Mapping[str, Mapping[str, Sequence[str]]],
    qrels: Mapping[str, Mapping[str, int]],
    name: str = "pr_curve",
    qids: Sequence[str] | None = None,
    title: str | None = None,
    out_dir: str | Path | None = None,
    threshold: int | None = None,
) -> list[Path]:
    """Plot 11-point interpolated P-R curves for ``runs`` = ``{label: {qid: [doc_id, ...]}}``."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, (label, rankings) in enumerate(runs.items()):
            ax.plot(
                RECALL_LEVELS,
                mean_11pt(rankings, qrels, qids, threshold),
                marker=MARKERS[i % len(MARKERS)],
                markersize=4,
                linewidth=1.5,
                label=label,
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Interpolated precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title or "11-point interpolated precision-recall")
        ax.legend(loc="upper right", fontsize=8)
        return save(fig, name, out_dir)


def heatmap(
    values: Sequence[Sequence[float]],
    row_labels: Sequence[object],
    col_labels: Sequence[object],
    name: str,
    row_title: str,
    col_title: str,
    title: str,
    cbar_label: str = "",
    out_dir: str | Path | None = None,
    best: tuple[int, int] | None = None,
) -> list[Path]:
    """Annotated heatmap (rows x cols); ``best`` = (row, col) cell to outline."""
    arr = np.asarray(values, dtype=float)
    with plt.rc_context(
        {
            **STYLE,
            "axes.grid": False,
            "figure.figsize": (1.0 + 0.6 * arr.shape[1], 1.0 + 0.45 * arr.shape[0]),
        }
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(arr, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(arr.shape[1]), [str(c) for c in col_labels])
        ax.set_yticks(range(arr.shape[0]), [str(r) for r in row_labels])
        ax.set_xlabel(col_title)
        ax.set_ylabel(row_title)
        ax.set_title(title)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                light = hi > lo and (arr[i, j] - lo) / (hi - lo) > 0.6
                ax.text(
                    j,
                    i,
                    f"{arr[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if light else "white",
                )
        if best is not None:
            ax.add_patch(
                plt.Rectangle(
                    (best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2
                )
            )
        fig.colorbar(im, ax=ax, label=cbar_label)
        return save(fig, name, out_dir)
