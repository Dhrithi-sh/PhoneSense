"""Query difficulty: guess when the results are probably poor.

Query performance prediction estimates how well a query will do without
any judgments. We use three signals, two from before ranking and one after:

- specificity (pre-retrieval): the highest idf among the query's words. A
  word's idf is taken from its most common strong spelling (weight 0.3 or
  more), so the Hinglish "kya" counts as common because क्या is, even though
  the Roman spelling itself is rare. A query made only of common words
  (के में, news, बड़ी खबर) can't single anything out.
- query scope (pre-retrieval): the share of the collection that contains any
  query word. Close to 1 means the query matches almost everything.
- clarity (post-retrieval, Cronen-Townsend, Zhou and Croft, SIGIR 2002): the
  KL divergence between the language model of the top results and the whole
  collection. A focused result list uses words very differently from the
  collection; a vague one looks like a random sample of it.

Plus the query parser's stage: if only the "any word" stage found anything,
no result contains all the query words.

A query is flagged "low confidence" when the stage is "any word" or the
specificity is below SPECIFICITY_LOW. Clarity is reported but not used for
the flag: on the full crawl it hardly separated good queries from vague
ones, because vague queries often hit a set of near-identical listing pages,
which look very focused. The threshold was set from our 64 needs queries on
the full crawl (lowest specificity 1.24) and a few vague queries (0.01 to
0.8); it's a starting point until there are judgments to fit it properly.
"""

import math
import re
from collections import Counter

from dhvani.rank.vsm import ZONES, query_vector

SPECIFICITY_LOW = 1.0   # the rarest query word is in more than 10% of articles
STRONG_WEIGHT = 0.3
TOP_DOCS = 10
SMOOTHING = 0.6          # weight of the document model vs the collection model
_WORD = re.compile(r"[\wऀ-ॿ]+")


def _cf(index, word):
    return sum(tf for zone in ZONES for _d, tf, _p in index.postings(word, zone))


def _total_tokens(index):
    total = getattr(index, "_total_tokens", None)
    if total is None:
        lengths = getattr(index, "doc_len", None)
        if lengths:
            total = sum(lengths.values())
        else:
            total = sum(len(_WORD.findall(a.get("headline", "") + " " + a.get("body", "")))
                        for a in getattr(index, "articles", {}).values())
        try:
            index._total_tokens = total
        except AttributeError:
            pass
    return max(total, 1)


def clarity(results, index, top=TOP_DOCS, smoothing=SMOOTHING):
    """KL(query model || collection model) in bits over the top results."""
    head = results[:top]
    if not head:
        return 0.0
    articles = getattr(index, "articles", {})
    scores = [max(s, 0.0) for _, s, _ in head]
    total_score = sum(scores) or float(len(head))
    weights = [s / total_score if sum(scores) else 1.0 / len(head) for s in scores]
    n_tokens = _total_tokens(index)
    query_model = Counter()
    doc_counts = []
    for doc_id, _s, _e in head:
        a = articles.get(doc_id, {})
        words = _WORD.findall((a.get("headline", "") + " " + a.get("body", "")).lower())
        doc_counts.append((Counter(words), max(len(words), 1)))
    vocab = set().union(*(c for c, _ in doc_counts))
    collection = {w: max(_cf(index, w), 1) / n_tokens for w in vocab}
    for (counts, length), weight in zip(doc_counts, weights):
        for w in vocab:
            p = smoothing * counts.get(w, 0) / length + (1 - smoothing) * collection[w]
            query_model[w] += weight * p
    return sum(p * math.log2(p / collection[w]) for w, p in query_model.items() if p > 0)


def word_idf(token, index):
    """idf of a query word's most common strong spelling, or None if none is in the index."""
    best = None
    for term, weight, _source in token["expansions"]:
        if weight < STRONG_WEIGHT:
            continue
        df = index.df(term)
        if df and (best is None or df > best):
            best = df
    return math.log10(index.N / best) if best else None


def predict(query, results, index):
    """{"specificity", "scope", "clarity", "stage", "low_confidence", "reasons"}."""
    qvec = query_vector(query, index)
    idfs = [i for i in (word_idf(tok, index) for tok in query["tokens"]) if i is not None]
    specificity = max(idfs) if idfs else 0.0
    matching = set()
    for term in qvec:
        for zone in ZONES:
            matching.update(d for d, _tf, _p in index.postings(term, zone))
    scope = len(matching) / max(index.N, 1)
    clar = clarity(results, index)
    stage = results[0][2].get("stage") if results and isinstance(results[0][2], dict) else None
    reasons = []
    if not results:
        reasons.append("nothing matched")
    if stage == "any word":
        reasons.append("no article has all the query words")
    if idfs and specificity < SPECIFICITY_LOW:
        reasons.append("every query word is very common")
    return {
        "specificity": specificity,
        "scope": scope,
        "clarity": clar,
        "stage": stage,
        "low_confidence": bool(reasons),
        "reasons": reasons,
    }
