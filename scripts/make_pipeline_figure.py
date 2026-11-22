"""Draw the PhoneSense pipeline diagram used in the report (R-B2).

    python scripts/make_pipeline_figure.py            # -> report/figures/pipeline.png

Pure matplotlib, no data needed, so the figure can always be regenerated. Edit BOXES / ARROWS
below when a component is added, renamed or removed, then re-run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from phonesense.config import get_path  # noqa: E402

# colours by who owns the stage (also stated in the legend)
COLORS = {"A": "#DCE9FB", "B": "#DDF3E4", "C": "#FDE8D3", "io": "#EDEDED"}
EDGE = "#4A4A4A"

# key: (x, y, width, height, label, owner)
BOXES = {
    # ---- offline lane
    "data": (0.2, 5.0, 1.7, 0.9, "Amazon Reviews 2023\nCell Phones", "io"),
    "prep": (2.2, 5.0, 2.0, 0.9, "Phone filter +\nspec parsing", "A"),
    "index": (4.5, 5.0, 1.9, 0.9, "Positional\ninverted index", "A"),
    "emb": (6.7, 5.0, 1.9, 0.9, "Sentence\nembeddings (cache)", "B"),
    "graph": (9.0, 5.0, 2.1, 0.9, "Phone graph\n(co-review)", "C"),
    "trustb": (11.5, 5.0, 2.0, 0.9, "Reviewer trust\n(HITS-style)", "C"),
    # ---- online lane
    "query": (0.2, 2.9, 1.5, 0.9, "User query", "io"),
    "lex": (2.1, 3.75, 2.1, 0.6, "TF-IDF lnc.ltc, BM25", "A"),
    "hyde": (2.1, 2.95, 2.1, 0.6, "HyDE (LLM listing)", "B"),
    "dense": (4.6, 2.15, 2.0, 0.6, "Dense retrieval", "B"),
    "prf": (4.6, 3.75, 2.0, 0.6, "Rocchio feedback", "B"),
    "fusion": (6.9, 2.9, 1.6, 0.9, "Rank fusion\nRRF / weighted", "B"),
    "filters": (8.8, 2.9, 1.6, 0.9, "Spec filters\n(Boolean)", "A"),
    "prior": (10.7, 2.9, 1.8, 0.9, "Prior rerank\nPageRank, trust", "C"),
    "out": (12.8, 2.9, 1.7, 0.9, "Ranked phones\n+ score breakdown", "io"),
    "attack": (10.7, 1.6, 1.8, 0.7, "Attack simulator\n(fake reviews)", "C"),
    "guide": (12.8, 1.6, 1.7, 0.7, "Buying guide\n(LLM, optional)", "B"),
    # ---- evaluation lane
    "qs": (0.2, 0.2, 1.9, 0.8, "45 queries\n(27 dev / 18 test)", "B"),
    "pool": (2.6, 0.2, 1.9, 0.8, "Pool top-k of\nevery system", "B"),
    "judge": (5.0, 0.2, 1.9, 0.8, "LLM judge 0-3\n+ human sample", "B"),
    "qrels": (7.4, 0.2, 1.7, 0.8, "Qrels +\nkappa", "B"),
    "metrics": (9.6, 0.2, 4.9, 0.8,
                "P@5, P@10, R@10, R@100, F1, MAP, nDCG@10\n+ paired test, ablation", "A"),
}

# (from, to, from side, to side[, waypoints]); sides: l r t b. Waypoints route an arrow around
# boxes: the line goes anchor -> waypoints -> anchor, with the arrowhead on the last segment.
ARROWS = [
    ("data", "prep", "r", "l"),
    ("prep", "index", "r", "l"),
    ("index", "emb", "r", "l"),
    ("emb", "graph", "r", "l"),
    ("graph", "trustb", "r", "l"),
    ("query", "lex", "r", "l"),
    ("query", "hyde", "r", "l"),
    ("query", "dense", "b", "l", [(0.95, 2.45)]),
    ("lex", "prf", "r", "l"),
    ("hyde", "prf", "r", "l"),
    ("prf", "fusion", "r", "l"),
    ("dense", "fusion", "r", "l"),
    ("fusion", "filters", "r", "l"),
    ("filters", "prior", "r", "l"),
    ("prior", "out", "r", "l"),
    ("out", "guide", "b", "t"),
    ("attack", "prior", "t", "b"),
    ("index", "lex", "b", "t"),
    ("emb", "dense", "b", "l", [(7.65, 4.5), (4.4, 4.5), (4.4, 2.45)]),
    ("graph", "prior", "b", "t"),
    ("trustb", "prior", "b", "t"),
    ("qs", "pool", "r", "l"),
    ("pool", "judge", "r", "l"),
    ("judge", "qrels", "r", "l"),
    ("qrels", "metrics", "r", "l"),
]


def _anchor(box: tuple, side: str) -> tuple[float, float]:
    x, y, w, h = box[:4]
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


def draw(path: Path, dpi: int = 200) -> Path:
    fig, ax = plt.subplots(figsize=(14.4, 6.6))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(-0.1, 6.5)
    ax.axis("off")

    ax.text(0.2, 6.15, "OFFLINE: build once", fontsize=9, fontweight="bold", color="#555555")
    ax.text(0.2, 4.55, "ONLINE: per query", fontsize=9, fontweight="bold", color="#555555",
            va="top")
    ax.text(0.2, 1.22, "EVALUATION", fontsize=9, fontweight="bold", color="#555555")
    for y in (4.65, 1.15):
        ax.plot([0.15, 14.55], [y, y], color="#BBBBBB", lw=0.8, ls="--")

    for x, y, w, h, text, owner in BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=COLORS[owner], ec=EDGE, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, zorder=3)

    arrow_style = {"arrowstyle": "-|>", "color": EDGE, "lw": 1.0, "shrinkA": 0, "shrinkB": 1}
    for a, b, sa, sb, *route in ARROWS:
        points = [_anchor(BOXES[a], sa), *(route[0] if route else []), _anchor(BOXES[b], sb)]
        for p0, p1 in zip(points[:-2], points[1:-1], strict=True):  # plain segments
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=EDGE, lw=1.0, zorder=1)
        ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops=arrow_style, zorder=1)

    handles = [FancyBboxPatch((0, 0), 1, 1, fc=COLORS[k], ec=EDGE) for k in ("A", "B", "C", "io")]
    ax.legend(handles, ["Member A: data, core IR, evaluation",
                        "Member B: dense, expansion, fusion, LLM, UI",
                        "Member C: graph, trust, attacks", "Input / output"],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)
    ax.set_title("PhoneSense pipeline", fontsize=12, fontweight="bold", loc="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="default: report/figures/pipeline.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    out = draw(args.out or get_path("figures_dir") / "pipeline.png", args.dpi)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
