"""Reciprocal rank fusion (RRF) of several rankers.

Each ranker gives its own ordered list. RRF ignores the raw scores and only
uses each article's rank in each list:

    rrf(d) = sum over lists of 1 / (c + rank of d in that list)

with c = 60 (Cormack, Clarke and Buettcher, SIGIR 2009). An article that
several rankers put near the top beats one that only a single ranker likes.
Because only ranks are used, scores on different scales (cosine, BM25, e5)
never need to be made comparable.

Lists fused here: lnc.ltc, BM25 and the net score, plus dense (e5) when it's
installed and switched on. For dense, the articles found by the sparse lists
are ordered by e5 cosine, so dense never adds articles the index didn't find.
"""

from collections import defaultdict

from dhvani.rank.bm25 import search_bm25
from dhvani.rank.scoring import rank
from dhvani.rank.vsm import search

RRF_C = 60
DEFAULT_DEPTH = 50


def rrf(lists, k=10, c=RRF_C):
    """Fuse {name: [(doc_id, score, explain), ...]} into one top-k list.

    explain for each article keeps the first available term scores under
    "terms" and adds "rrf": {name: rank or None} and "rrf_score".
    """
    fused = defaultdict(float)
    ranks = defaultdict(dict)
    terms = {}
    for name, results in lists.items():
        for position, (doc_id, _score, explain) in enumerate(results, start=1):
            fused[doc_id] += 1.0 / (c + position)
            ranks[doc_id][name] = position
            if doc_id not in terms:
                terms[doc_id] = explain.get("terms", explain) if isinstance(explain, dict) else {}
    out = []
    for doc_id, score in fused.items():
        explain = {
            "terms": dict(terms[doc_id]),
            "rrf": {name: ranks[doc_id].get(name) for name in lists},
            "rrf_score": score,
        }
        out.append((doc_id, score, explain))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out[:k] if k else out


def dense_list(query, dense, candidates):
    """The candidate articles ordered by e5 cosine, as a ranked list."""
    sims = dense.similarity(query, candidates)
    ordered = sorted(sims.items(), key=lambda item: (-item[1], item[0]))
    return [(doc_id, sim, {"dense_cosine": sim}) for doc_id, sim in ordered]


def search_rrf(query, index, k=10, depth=DEFAULT_DEPTH, doc_filter=None, static=None, dense=None, c=RRF_C):
    """Fuse lnc.ltc, BM25, the net score and (optionally) dense with RRF."""
    depth = max(depth, k)
    lists = {
        "lnc.ltc": search(query, index, k=depth, doc_filter=doc_filter),
        "BM25": search_bm25(query, index, k=depth, doc_filter=doc_filter),
        "net": rank(query, index, k=depth, doc_filter=doc_filter, static=static),
    }
    if dense is not None:
        candidates = {d for results in lists.values() for d, _, _ in results}
        lists["dense"] = dense_list(query, dense, candidates)
    return rrf(lists, k=k, c=c)
