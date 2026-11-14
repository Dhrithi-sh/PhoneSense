"""Ranked retrieval metrics (Lecture 8) and TREC-style file helpers.

All metrics take a ranked list of doc ids (best first) and the judgments for
one query, as {doc_id: grade}. Grades are 0 (not relevant), 1 (partly) or
2 (fully). For the binary metrics anything with grade >= 1 counts as relevant.

    precision@k   relevant in the top k / k
    recall@k      relevant in the top k / all relevant
    AP            mean of precision@r over the ranks r of relevant docs,
                  divided by the total number of relevant docs (so missed
                  relevant docs count as zero)
    MAP           mean AP over queries
    DCG@k         sum over r <= k of grade_r / log2(r + 1)
    nDCG@k        DCG@k / DCG@k of the ideal ordering
    11-point      interpolated precision at recall 0.0, 0.1, ..., 1.0, where
                  interpolated precision at r is the max precision at any
                  recall >= r

File formats (see documentation/formats.md):
    run file:   qid Q0 doc_id rank score run_name
    judgments:  qid 0 doc_id grade
"""

import math
from collections import defaultdict

RECALL_LEVELS = [i / 10 for i in range(11)]


def _relevant(qrels, threshold=1):
    return {d for d, g in qrels.items() if g >= threshold}


def precision_at_k(ranking, qrels, k):
    if k <= 0:
        return 0.0
    rel = _relevant(qrels)
    return sum(1 for d in ranking[:k] if d in rel) / k


def recall_at_k(ranking, qrels, k):
    rel = _relevant(qrels)
    if not rel:
        return 0.0
    return sum(1 for d in ranking[:k] if d in rel) / len(rel)


def f1_at_k(ranking, qrels, k):
    p, r = precision_at_k(ranking, qrels, k), recall_at_k(ranking, qrels, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision(ranking, qrels):
    rel = _relevant(qrels)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def mean_average_precision(rankings, qrels_by_query):
    """rankings: {qid: [doc_ids]}. Queries with no judgments are skipped."""
    aps = [average_precision(rankings.get(q, []), qrels)
           for q, qrels in qrels_by_query.items() if _relevant(qrels)]
    return sum(aps) / len(aps) if aps else 0.0


def dcg_at_k(ranking, qrels, k):
    return sum(qrels.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1))


def ndcg_at_k(ranking, qrels, k):
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg_at_k(ranking, qrels, k) / idcg if idcg else 0.0


def precision_recall_points(ranking, qrels):
    """(recall, precision) after each rank, for drawing a PR curve."""
    rel = _relevant(qrels)
    if not rel:
        return []
    points, hits = [], 0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
        points.append((hits / len(rel), hits / i))
    return points


def interpolated_11_point(ranking, qrels):
    """Interpolated precision at recall 0.0, 0.1, ..., 1.0 for one query."""
    points = precision_recall_points(ranking, qrels)
    out = []
    for level in RECALL_LEVELS:
        candidates = [p for r, p in points if r >= level - 1e-12]
        out.append(max(candidates) if candidates else 0.0)
    return out


def average_11_point(rankings, qrels_by_query):
    """11-point interpolated precision averaged over queries."""
    curves = [interpolated_11_point(rankings.get(q, []), qrels)
              for q, qrels in qrels_by_query.items() if _relevant(qrels)]
    if not curves:
        return [0.0] * len(RECALL_LEVELS)
    return [sum(c[i] for c in curves) / len(curves) for i in range(len(RECALL_LEVELS))]


def evaluate(rankings, qrels_by_query, k=10):
    """Per-query and mean metrics. Returns (per_query, means)."""
    per_query = {}
    for q, qrels in qrels_by_query.items():
        if not _relevant(qrels):
            continue
        ranking = rankings.get(q, [])
        per_query[q] = {
            f"P@{k}": precision_at_k(ranking, qrels, k),
            f"R@{k}": recall_at_k(ranking, qrels, k),
            "AP": average_precision(ranking, qrels),
            f"nDCG@{k}": ndcg_at_k(ranking, qrels, k),
        }
    if not per_query:
        return {}, {}
    names = next(iter(per_query.values())).keys()
    means = {n: sum(m[n] for m in per_query.values()) / len(per_query) for n in names}
    means["MAP"] = means.pop("AP")
    return per_query, means


# --- TREC-style files -------------------------------------------------------

def write_run(path, results_by_query, run_name):
    """results_by_query: {qid: [(doc_id, score), ...]} best first."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, results in results_by_query.items():
            for rank_no, (doc_id, score) in enumerate(results, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank_no} {score:.6f} {run_name}\n")


def read_run(path):
    """{qid: [doc_ids]} in rank order."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                rows[parts[0]].append((int(parts[3]), parts[2]))
    return {q: [d for _, d in sorted(r)] for q, r in rows.items()}


def read_qrels(path):
    """{qid: {doc_id: grade}}."""
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)
