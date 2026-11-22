"""How stop words and idf behave on our Hindi corpus (Track 5 requirement).

    python -m dhvani.eval.corpus_stats
    python -m dhvani.eval.corpus_stats --top 40 --stop 25

Prints and saves:

- term statistics: df, collection frequency and idf = log10(N / df) for every term
- the most frequent terms with their idf. Common Hindi function words
  (का, की, के, में, है) should come out at the top with idf near 0, so a stop
  word list falls out of the data instead of being typed in by hand
- Zipf's law: collection frequency against rank on a log-log scale, with the
  fitted slope (close to -1 for natural text)
- a stop word experiment on the judged queries, comparing
    no idf          lnc.lnc: every word counts, stop words included
    idf             lnc.ltc: stop words stay but idf pushes them towards 0
    removed         lnc.ltc with stop words dropped from the query

Plots go to data/eval/ and are only drawn if matplotlib is installed.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

from dhvani.eval.experiments import OUT_DIR, SAMPLE_DIR, read_queries
from dhvani.eval.metrics import evaluate, read_qrels
from dhvani.rank.real_index import make_query
from dhvani.rank.vsm import ZONES, cosine_scores


def term_stats(index):
    """{term: {"df", "cf", "idf"}} for every term in the index."""
    stats = {}
    for term in index.vocab:
        cf = sum(tf for zone in ZONES for _d, tf, _p in index.postings(term, zone))
        df = index.df(term)
        stats[term] = {"df": df, "cf": cf, "idf": math.log10(index.N / df) if df else 0.0}
    return stats


def top_terms(stats, n=30, by="df"):
    """The n most frequent terms, as [(term, df, cf, idf)]."""
    ordered = sorted(stats.items(), key=lambda kv: (-kv[1][by], kv[0]))[:n]
    return [(t, s["df"], s["cf"], s["idf"]) for t, s in ordered]


def stop_words(stats, n=25):
    """The n highest-df terms. Their idf is lowest, so they carry the least signal."""
    return [t for t, *_ in top_terms(stats, n=n, by="df")]


def zipf_fit(stats):
    """Least-squares slope and intercept of log10(cf) against log10(rank)."""
    freqs = sorted((s["cf"] for s in stats.values() if s["cf"] > 0), reverse=True)
    if len(freqs) < 2:
        return 0.0, 0.0
    xs = [math.log10(r) for r in range(1, len(freqs) + 1)]
    ys = [math.log10(f) for f in freqs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return slope, my - slope * mx


def _without(query, words):
    tokens = [t for t in query["tokens"] if t["surface"] not in words]
    return {"raw": query["raw"], "tokens": tokens}


def stopword_experiment(index, queries, qrels, stop, k=10):
    """Mean P@k, MAP and nDCG@k for no idf / idf / stop words removed."""
    by_query = {qid: qrels.get(need, {}) for qid, need, _f, _ in queries}
    setups = {
        "no idf (lnc.lnc)": lambda q: cosine_scores(q, index, use_idf=False)[0],
        "idf (lnc.ltc)": lambda q: cosine_scores(q, index)[0],
        "stop words removed": lambda q: cosine_scores(_without(q, set(stop)), index)[0],
    }
    out = {}
    for name, score in setups.items():
        rankings = {}
        for qid, _need, _form, text in queries:
            scores = score(make_query(text, "none"))
            rankings[qid] = [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
        _, means = evaluate(rankings, by_query, k=k)
        out[name] = means
    return out


def plot(stats, out_dir=OUT_DIR):
    """Zipf plot and idf histogram. Returns the saved paths, or [] without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    freqs = sorted((s["cf"] for s in stats.values() if s["cf"] > 0), reverse=True)
    slope, intercept = zipf_fit(stats)
    ranks = range(1, len(freqs) + 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(ranks, freqs, marker=".", linestyle="none", label="terms")
    ax.loglog(ranks, [10 ** (intercept + slope * math.log10(r)) for r in ranks], label=f"fit, slope {slope:.2f}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Collection frequency")
    ax.set_title("Zipf's law on the Hindi corpus")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "zipf.png", dpi=150)
    plt.close(fig)
    saved.append(out_dir / "zipf.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist([s["idf"] for s in stats.values()], bins=20)
    ax.set_xlabel("idf = log10(N / df)")
    ax.set_ylabel("Number of terms")
    ax.set_title("idf across the vocabulary")
    fig.tight_layout()
    fig.savefig(out_dir / "idf_histogram.png", dpi=150)
    plt.close(fig)
    saved.append(out_dir / "idf_histogram.png")
    return saved


def load_index():
    """Dhrithi's no-stemming index if it's built, else the sample index."""
    from dhvani.rank.real_index import load_index as load_real
    return load_real("none")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stop words, idf and Zipf on the corpus.")
    parser.add_argument("--top", type=int, default=20, help="how many frequent terms to list")
    parser.add_argument("--stop", type=int, default=15, help="how many top-df terms count as stop words")
    parser.add_argument("--queries", default=None, help="queries tsv for the stop word experiment")
    parser.add_argument("--qrels", default=None, help="judgments for the stop word experiment")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)

    index = load_index()
    stats = term_stats(index)
    print(f"{index.N} articles, {len(stats)} distinct terms")

    print(f"\nMost frequent {args.top} terms")
    print(f"{'term':<14}{'df':>5}{'cf':>6}{'idf':>8}")
    for term, df, cf, idf in top_terms(stats, args.top):
        print(f"{term:<14}{df:>5}{cf:>6}{idf:>8.3f}")

    stop = stop_words(stats, args.stop)
    print(f"\nStop words from the data (top {args.stop} by df): " + " ".join(stop))

    slope, _ = zipf_fit(stats)
    sign = "-" if slope < 0 else "+"
    print(f"\nZipf fit: log10(cf) = a {sign} {abs(slope):.2f} x log10(rank), slope {slope:.2f} (natural text is close to -1)")

    from dhvani.rank.real_index import real_index_available
    if args.qrels:
        queries, qrels = read_queries(args.queries or SAMPLE_DIR / "queries.tsv"), read_qrels(args.qrels)
    elif not real_index_available("none"):
        queries, qrels = read_queries(SAMPLE_DIR / "queries.tsv"), read_qrels(SAMPLE_DIR / "qrels.txt")
    else:
        queries, qrels = [], {}
    if not qrels:
        print("\nStop word experiment skipped: no judgments for the real index yet (pass --queries and --qrels)")
    else:
        print("\nStop word experiment (mean over judged queries)")
        print(f"{'setup':<22}{'P@10':>8}{'MAP':>8}{'nDCG@10':>9}")
        for name, m in stopword_experiment(index, queries, qrels, stop).items():
            print(f"{name:<22}{m['P@10']:>8.4f}{m['MAP']:>8.4f}{m['nDCG@10']:>9.4f}")

    saved = plot(stats, args.out)
    print("\nPlots: " + ", ".join(str(p) for p in saved) if saved else "\nPlots skipped: matplotlib isn't installed")


if __name__ == "__main__":
    main()
