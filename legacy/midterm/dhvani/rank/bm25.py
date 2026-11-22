"""Okapi BM25, our main baseline next to lnc.ltc.

    score(q, d) = sum over query terms t of
        w_q(t) * idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

- tf saturates: the 10th occurrence of a word adds much less than the 2nd,
  unlike raw tf. k1 controls how fast it saturates.
- b controls length normalisation: b = 1 fully penalises long articles,
  b = 0 ignores length.
- The "1 +" inside the log keeps idf positive even for words in more than
  half the articles (the classic Robertson idf goes negative there).
- w_q(t) is the query weight, i.e. the summed expansion weights from the
  query object, so phonetic and translated variants count less than exact
  matches.

headline_boost lets headline occurrences count extra, a simple version of
zone weighting (BM25F style). With the default 1.0 it's plain BM25.
"""

import heapq
import math
from collections import defaultdict

from dhvani.rank.vsm import ZONES

K1 = 1.2
B = 0.75


def doc_lengths(index):
    """Article length in tokens for every document.

    Uses index.doc_len if the index provides it, otherwise adds up tf over
    the postings once and caches the result on the index.
    """
    if getattr(index, "doc_len", None):
        return index.doc_len
    cached = getattr(index, "_bm25_doc_len", None)
    if cached is not None:
        return cached
    lengths = defaultdict(int)
    for term in index.vocab:
        for zone in ZONES:
            for doc_id, tf, _ in index.postings(term, zone):
                lengths[doc_id] += tf
    lengths = dict(lengths)
    index._bm25_doc_len = lengths
    return lengths


def idf(df, n):
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def query_weights(query):
    weights = defaultdict(float)
    for token in query["tokens"]:
        for term, weight, _source in token["expansions"]:
            weights[term] += weight
    return weights


def bm25_scores(query, index, k1=K1, b=B, headline_boost=1.0):
    """BM25 score for every document sharing a term with the query.

    Returns (scores, contributions) like vsm.cosine_scores.
    """
    lengths = doc_lengths(index)
    avgdl = sum(lengths.values()) / len(lengths) if lengths else 0.0

    scores = defaultdict(float)
    contributions = defaultdict(dict)
    for term, q_weight in query_weights(query).items():
        df = index.df(term)
        if df == 0:
            continue
        term_idf = idf(df, index.N)
        tf_by_doc = defaultdict(float)
        for zone in ZONES:
            boost = headline_boost if zone == "headline" else 1.0
            for doc_id, tf, _ in index.postings(term, zone):
                tf_by_doc[doc_id] += boost * tf
        for doc_id, tf in tf_by_doc.items():
            norm = k1 * (1 - b + b * lengths[doc_id] / avgdl)
            part = q_weight * term_idf * tf * (k1 + 1) / (tf + norm)
            scores[doc_id] += part
            contributions[doc_id][term] = part
    return scores, contributions


def search_bm25(query, index, k=10, k1=K1, b=B, headline_boost=1.0, doc_filter=None):
    """Top k by BM25: [(doc_id, score, {term: contribution})], best first."""
    scores, contributions = bm25_scores(query, index, k1, b, headline_boost)
    candidates = scores.items()
    if doc_filter:
        candidates = [(d, s) for d, s in candidates if doc_filter(d)]
    top = heapq.nlargest(k, candidates, key=lambda item: (item[1], item[0]))
    return [(doc_id, score, contributions[doc_id]) for doc_id, score in top]
