"""Pseudo-relevance feedback with Rocchio (Lecture: relevance feedback and query expansion).

Search once, assume the top real articles are relevant, and move the query
towards them with Viraja's Rocchio update (dhvani/query/rocchio.py):

    q_new = alpha x q + (beta / |Dr|) x sum of the feedback article vectors

The strongest new terms are added to the query with source "prf" and the
search runs again. This helps when the query's words are rare or misspelt
but the top articles are right: their other words (names, places, the
Hindi word for an English term) pull in more articles about the same story.

Two changes from the plain version, needed on the full crawl:
- Article vectors are tf x idf (ltc weights), built from the feedback
  articles' own text. With plain tf the "best" new terms were के, में, की;
  idf keeps the words that describe those articles. Building them from the
  text also avoids scanning the whole vocabulary for every query.
- Feedback articles are taken after listing pages and horoscopes are pushed
  down, so a section page full of unrelated headlines never shapes the query.
"""

import math
from collections import Counter

from dhvani.query.rocchio import expand_query
from dhvani.rank.parser import parse_and_rank
from dhvani.rank.quality import demote, page_type

FEEDBACK_DOCS = 5
FEEDBACK_TERMS = 5
FIRST_PASS = 30


def _analyze(text, mode):
    try:
        from text.analyzer import analyze
        return [t for t, _pos in analyze(text, mode)]
    except Exception:   # sample index without Dhrithi's analyzer: plain words
        from dhvani.rank.sample_index import tokenize
        return tokenize(text)


def article_vector(index, doc_id, mode="none"):
    """{term: (1 + log10 tf) x idf} for one article, from its headline and body."""
    a = getattr(index, "articles", {}).get(doc_id, {})
    tf = Counter(_analyze(a.get("headline", "") + " " + a.get("body", ""), mode))
    vec = {}
    for term, count in tf.items():
        df = index.df(term)
        if df:
            vec[term] = (1 + math.log10(count)) * math.log10(index.N / df)
    return vec


def feedback_query(query, index, mode="none", ranker="net", docs=FEEDBACK_DOCS, terms=FEEDBACK_TERMS, static=None):
    """Return (expanded query, feedback doc ids). The input query isn't changed."""
    first = demote(parse_and_rank(query, index, k=FIRST_PASS, ranker=ranker, static=static), index)
    feedback = [d for d, _s, _e in first if page_type(d, index) == "article"][:docs]
    if not feedback:
        return query, []
    expanded = {"raw": query["raw"], "tokens": [dict(t, expansions=list(t["expansions"])) for t in query["tokens"]]}
    expand_query(expanded, [article_vector(index, d, mode) for d in feedback], k=terms)
    return expanded, feedback
