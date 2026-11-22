"""Learning to rank: learn the weights of the score's parts from our judgments.

The net score mixes its parts with hand-picked weights (cosine + 0.2 zone +
0.1 proximity + 0.1 g(d)). Here every judged query-article pair becomes a row
of features, and a logistic regression (Lecture 15's machine-learned
relevance, pointwise) learns how much each feature says about relevance.

Features, all for the no-stemming index:
  cosine (lnc.ltc), bm25, zone (headline match), proximity, recency,
  pagerank, original (first to publish), article (1 if not a listing page or
  horoscope), stage (the query parser stage, 1 for exact phrase down to 0 for
  any word), and dense (e5 cosine, 0 when dense isn't installed).

Evaluation is leave-one-need-out: for each information need, the model is
trained on all the other needs, then used to re-rank that need's queries.
Nothing is ever scored by a model that saw its judgments. The learned ranker
re-ranks the net score's top CANDIDATES articles, and is compared with the
net score itself on MAP and P@10 over the same queries.

    python -m dhvani.eval.ltr          # once judgments/ has judgments
"""

import argparse
import math
from collections import defaultdict

from dhvani.eval.metrics import average_precision, precision_at_k

FEATURES = ("cosine", "bm25", "zone", "proximity", "recency", "pagerank", "original", "article", "stage", "dense")
CANDIDATES = 30
STAGE_VALUE = {"phrase": 1.0, "sub-phrase": 0.75, "all words": 0.5, "all words, with variants": 0.25, "any word": 0.0}


def query_features(query, index, static_parts, k=CANDIDATES, dense=None):
    """[(doc_id, {feature: value})] for the net score's top k candidates of one query."""
    from dhvani.rank.bm25 import bm25_scores
    from dhvani.rank.parser import parse_and_rank
    from dhvani.rank.quality import page_type

    results = parse_and_rank(query, index, k=k, ranker="net")
    bm25, _ = bm25_scores(query, index)
    sims = dense.similarity(query, [d for d, _, _ in results]) if dense else {}
    rows = []
    for doc_id, _score, e in results:
        parts = static_parts.get(doc_id, {})
        rows.append((doc_id, {
            "cosine": e.get("cosine", 0.0),
            "bm25": bm25.get(doc_id, 0.0),
            "zone": e.get("zone", 0.0),
            "proximity": e.get("proximity", 0.0),
            "recency": parts.get("recency", e.get("recency", 0.0)),
            "pagerank": parts.get("pagerank", 0.0),
            "original": parts.get("original", 0.0),
            "article": 1.0 if page_type(doc_id, index) == "article" else 0.0,
            "stage": STAGE_VALUE.get(e.get("stage"), 0.0),
            "dense": sims.get(doc_id, 0.0),
        }))
    return rows


def build_table(queries, qrels, index, build_query, static_parts, k=CANDIDATES, dense=None):
    """{qid: (need, [(doc_id, features, grade or None)])} for every query of a judged need."""
    table = {}
    for qid, need, _form, text in queries:
        if not qrels.get(need):
            continue
        rows = query_features(build_query(text, "none"), index, static_parts, k, dense)
        table[qid] = (need, [(d, f, qrels[need].get(d)) for d, f in rows])
    return table


class Logistic:
    """Logistic regression with standardised features, trained by gradient descent."""

    def __init__(self, features=FEATURES, l2=0.01, steps=2000, rate=0.1):
        self.features, self.l2, self.steps, self.rate = features, l2, steps, rate

    def fit(self, rows, labels):
        n, m = len(rows), len(self.features)
        cols = [[r[f] for r in rows] for f in self.features]
        self.mean = [sum(c) / n for c in cols]
        self.std = [math.sqrt(sum((x - mu) ** 2 for x in c) / n) or 1.0 for c, mu in zip(cols, self.mean)]
        X = [[(r[f] - mu) / sd for f, mu, sd in zip(self.features, self.mean, self.std)] for r in rows]
        self.w, self.b = [0.0] * m, 0.0
        pos = sum(labels) or 1
        weight = {1: n / (2 * pos), 0: n / (2 * max(n - pos, 1))}     # balance relevant and not relevant
        for _ in range(self.steps):
            gw, gb = [0.0] * m, 0.0
            for x, y in zip(X, labels):
                p = 1 / (1 + math.exp(-max(-30, min(30, self.b + sum(wi * xi for wi, xi in zip(self.w, x))))))
                g = (p - y) * weight[y]
                gb += g
                for j in range(m):
                    gw[j] += g * x[j]
            self.b -= self.rate * gb / n
            self.w = [wj - self.rate * (gj / n + self.l2 * wj) for wj, gj in zip(self.w, gw)]
        return self

    def score(self, row):
        return self.b + sum(w * (row[f] - mu) / sd for w, f, mu, sd in zip(self.w, self.features, self.mean, self.std))

    def weights(self):
        return dict(zip(self.features, self.w))


def leave_one_need_out(table, k=10):
    """Per-query AP and P@k for the net score's order and the learned order, plus the average weights."""
    needs = sorted({need for need, _ in table.values()})
    net, learned = {}, {}
    weights = defaultdict(float)
    for held_out in needs:
        train_rows, labels = [], []
        for need, rows in table.values():
            if need == held_out:
                continue
            for _d, f, grade in rows:
                if grade is not None:
                    train_rows.append(f)
                    labels.append(1 if grade >= 1 else 0)
        if not train_rows or len(set(labels)) < 2:
            continue
        model = Logistic().fit(train_rows, labels)
        for f, w in model.weights().items():
            weights[f] += w / len(needs)
        for qid, (need, rows) in table.items():
            if need != held_out:
                continue
            judged = {d: g for d, _f, g in rows if g is not None}
            base = [d for d, _f, _g in rows]
            mine = [d for d, _f, _g in sorted(rows, key=lambda r: -model.score(r[1]))]
            net[qid] = (average_precision(base, judged), precision_at_k(base, judged, k))
            learned[qid] = (average_precision(mine, judged), precision_at_k(mine, judged, k))
    return net, learned, dict(weights)


def main(argv=None):
    from dhvani.eval.experiments import build_query, read_needs
    from dhvani.eval.judge import all_judgments
    from dhvani.eval.significance import compare
    from dhvani.rank.authority import static_scores
    from dhvani.rank.real_index import load_index

    parser = argparse.ArgumentParser(description="Learning to rank with leave-one-need-out evaluation.")
    parser.add_argument("--dense", action="store_true", help="include the e5 cosine as a feature")
    args = parser.parse_args(argv)

    qrels = all_judgments()
    if not qrels:
        print("No judgments yet in judgments/.")
        return
    index = load_index("none")
    _g, parts = static_scores(index)
    dense = None
    if args.dense:
        from dhvani.rank.dense import DenseIndex, SentenceEncoder
        dense = DenseIndex(index, SentenceEncoder())
    table = build_table(read_needs(), qrels, index, build_query, parts, dense=dense)
    net, learned, weights = leave_one_need_out(table)
    if not net:
        print("Not enough judged needs to train (need relevant and non-relevant examples in at least two needs).")
        return
    n = len(net)
    print(f"{n} queries from {len({table[q][0] for q in net})} judged needs, net score's top {CANDIDATES} re-ranked")
    for name, run in (("net score", net), ("learned", learned)):
        print(f"{name:<10} MAP {sum(v[0] for v in run.values()) / n:.4f}  P@10 {sum(v[1] for v in run.values()) / n:.4f}")
    c = compare({q: v[0] for q, v in learned.items()}, {q: v[0] for q, v in net.items()})
    print(f"learned vs net score on AP: p = {c['p_randomization']:.3f} (randomization), {c['p_t']:.3f} (t-test)")
    print("Average learned weights (standardised features):")
    for f, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:<10} {w:+.3f}")


if __name__ == "__main__":
    main()
