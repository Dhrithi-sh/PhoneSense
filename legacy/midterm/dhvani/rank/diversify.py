"""Result diversification with maximal marginal relevance (MMR).

News search often returns the same story several times from different
papers, with different wording, so duplicate collapsing (which needs Riya's
dup_of) doesn't catch it. MMR (Carbonell and Goldstein, SIGIR 1998) re-orders
the results so each next pick is relevant but not too similar to what's
already been picked:

    pick = argmax over remaining d of  lam x rel(d) - (1 - lam) x max sim(d, already picked)

rel is the ranker's score scaled to [0, 1]; sim is the cosine between the
two articles' term vectors (log tf, length-normalized, from headline and
body). lam = 1 is the original order; lower lam spreads the top k over more
stories. Like kal and dense, it works within a query-parser stage, so a
stricter match is never pushed below a looser one.
"""

import math
import re
from collections import Counter

DEFAULT_LAMBDA = 0.7
_WORD = re.compile(r"[\wऀ-ॿ]+")


def text_vector(article):
    """{word: weight}, log tf, unit length, from headline + body."""
    words = _WORD.findall((article.get("headline", "") + " " + article.get("body", "")).lower())
    tf = Counter(words)
    vec = {w: 1 + math.log10(c) for w, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {w: v / norm for w, v in vec.items()}


def _cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _mmr_group(group, vectors, lam):
    scores = [s for _, s, _ in group]
    lo, hi = min(scores), max(scores)
    rel = {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s, _ in group}
    remaining = list(group)
    picked = []
    while remaining:
        best, best_val, best_sim = None, None, 0.0
        for item in remaining:
            d = item[0]
            sim = max((_cosine(vectors[d], vectors[p[0]]) for p in picked), default=0.0)
            val = lam * rel[d] - (1 - lam) * sim
            if best is None or val > best_val:
                best, best_val, best_sim = item, val, sim
        remaining.remove(best)
        doc_id, score, explain = best
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        explain["mmr"] = {"relevance": rel[doc_id], "max_similarity": best_sim, "lambda": lam}
        picked.append((doc_id, score, explain))
    return picked


def diversify(results, index, k=None, lam=DEFAULT_LAMBDA):
    """MMR re-ordering of a result list, stage by stage. Returns the top k."""
    if not results:
        return results
    articles = getattr(index, "articles", {})
    vectors = {d: text_vector(articles.get(d, {})) for d, _, _ in results}
    groups, order = {}, []
    for item in results:
        stage = item[2].get("stage") if isinstance(item[2], dict) else None
        if stage not in groups:
            groups[stage] = []
            order.append(stage)
        groups[stage].append(item)
    out = []
    for stage in order:
        out.extend(_mmr_group(groups[stage], vectors, lam))
    return out[:k] if k else out
