"""Graphs for Rishit's results, from the final run files and the judgments.

    .venv/bin/python scripts/rishit_figures.py

Run the experiment runner first (python -m dhvani.eval.experiments --out
data/eval/full). Writes to documentation/figures/:

- rishit-pr-curves.png: 11-point interpolated precision-recall, one line per
  ranker (no stemming)
- rishit-rankers.png: P@10, MAP and nDCG@10 per ranker (no stemming)
- rishit-stemming.png: P@10 and MAP per stemming mode, for BM25 and the net score
- rishit-translation.png: English queries with translation off vs on
- rishit-ltr.png: learned ranker vs the hand-tuned net score
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dhvani.eval.experiments import build_query, build_query_no_xling, read_needs, run_one  # noqa: E402
from dhvani.eval.judge import all_judgments  # noqa: E402
from dhvani.eval.metrics import average_11_point, evaluate, read_run  # noqa: E402

RUNS = ROOT / "data" / "eval" / "full" / "runs"
OUT = ROOT / "documentation" / "figures"
RANKERS = [("lnc", "lnc.ltc"), ("bm25", "BM25"), ("rrf", "Fusion"), ("net", "Net score")]
MODES = [("none", "None"), ("light", "Light"), ("aggr", "Aggressive"), ("auto", "Auto")]


def means(run, queries, qrels):
    by_query = {qid: qrels.get(need, {}) for qid, need, _f, _t in queries if qrels.get(need)}
    _, m = evaluate({q: run.get(q, []) for q in by_query}, by_query, k=10)
    return m


def bars(ax, labels, series, title):
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series):
        xs = [x + i * width for x in range(len(labels))]
        b = ax.bar(xs, values, width, label=name)
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks([x + width * (len(series) - 1) / 2 for x in range(len(labels))], labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(OUT / name)


def main():
    queries, qrels = read_needs(), all_judgments()
    runs = {p.stem: read_run(p) for p in RUNS.glob("*.txt")}
    judged = {qid: qrels[need] for qid, need, _f, _t in queries if qrels.get(need)}
    n_needs = len({need for _q, need, _f, _t in queries if qrels.get(need)})
    note = f"{len(judged)} queries, {n_needs} needs, frozen corpus of 5,000 articles"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    levels = [i / 10 for i in range(11)]
    for key, label in RANKERS:
        ax.plot(levels, average_11_point(runs[f"none_{key}"], judged), marker="o", label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title(f"Precision-recall, no stemming\n{note}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "rishit-pr-curves.png")

    m = [means(runs[f"none_{k}"], queries, qrels) for k, _ in RANKERS]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, [l for _, l in RANKERS], [("P@10", [x["P@10"] for x in m]), ("MAP", [x["MAP"] for x in m]),
                                        ("nDCG@10", [x["nDCG@10"] for x in m])], f"Rankers, no stemming ({note})")
    save(fig, "rishit-rankers.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ranker, label in ((axes[0], "bm25", "BM25"), (axes[1], "net", "Net score")):
        mm = [means(runs[f"{mode}_{ranker}"], queries, qrels) for mode, _ in MODES]
        bars(ax, [l for _, l in MODES], [("P@10", [x["P@10"] for x in mm]), ("MAP", [x["MAP"] for x in mm])],
             f"Stemming, {label}")
    save(fig, "rishit-stemming.png")

    from dhvani.rank.real_index import load_index
    english = [q for q in queries if q[2] == "english" and qrels.get(q[1])]
    idx = load_index("none")
    rows = []
    for builder in (build_query_no_xling, build_query):
        run = run_one(idx, english, "net", query_builder=builder, mode="none")
        rows.append(means({q: [d for d, _ in r] for q, r in run.items()}, english, qrels))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars(ax, ["Translation off", "Dictionary translation"],
         [("P@10", [r["P@10"] for r in rows]), ("MAP", [r["MAP"] for r in rows]), ("nDCG@10", [r["nDCG@10"] for r in rows])],
         f"English queries ({len(english)}), net score")
    save(fig, "rishit-translation.png")

    from dhvani.eval.ltr import build_table, leave_one_need_out
    from dhvani.rank.authority import static_scores
    _g, parts = static_scores(idx)
    dense = None
    try:
        from dhvani.rank.dense import DenseIndex, SentenceEncoder, dense_available
        dense = DenseIndex(idx, SentenceEncoder()) if dense_available() else None
    except Exception:
        dense = None
    net, learned, weights = leave_one_need_out(build_table(queries, qrels, idx, build_query, parts, dense=dense))
    k = len(net)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars(axes[0], ["Net score", "Learned"],
         [("MAP", [sum(v[0] for v in net.values()) / k, sum(v[0] for v in learned.values()) / k]),
          ("P@10", [sum(v[1] for v in net.values()) / k, sum(v[1] for v in learned.values()) / k])],
         f"Learning to rank, leave-one-need-out ({k} queries)")
    names = sorted(weights, key=weights.get)
    axes[1].barh(names, [weights[n] for n in names], color=["tab:red" if weights[n] < 0 else "tab:green" for n in names])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Learned weights (standardised features)")
    axes[1].grid(axis="x", alpha=0.3)
    save(fig, "rishit-ltr.png")


if __name__ == "__main__":
    main()
