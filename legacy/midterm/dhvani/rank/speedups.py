"""Faster cosine ranking by scoring fewer articles (Lecture 7).

Plain lnc.ltc scores every article that shares any word with the query. On
a real corpus that's most of the collection for common words. These methods
pick a smaller set of contenders A, score only those, and return the top k
from A. Cosine is only a proxy for what the user wants anyway, so a top k
that's close to the exact one is usually good enough.

Every search here returns (results, stats). stats says how many articles
were actually scored, so we can measure speed (articles scored) against
quality (how much of the exact top k we kept).

Index elimination
- Only high-idf query terms: words like का, की, में appear almost everywhere,
  add little to the score and drag in huge postings lists, so they're skipped.
- Only articles containing several query terms: with 3 or more query words
  an article has to contain most of them (a "soft AND"), e.g. 3 of 4.
"""

import heapq
import math
from collections import defaultdict

from dhvani.rank.vsm import ZONES, log_tf, query_vector


def _doc_tf(index, term):
    """{doc_id: tf} for a term, adding up the headline and body zones."""
    out = defaultdict(int)
    for zone in ZONES:
        for doc_id, tf, _positions in index.postings(term, zone):
            out[doc_id] += tf
    return out


def _score(qvec, index, candidates, doc_filter=None):
    """Cosine score for the candidate articles only. Returns (scores, contributions)."""
    scores = defaultdict(float)
    contributions = defaultdict(dict)
    for term, q_weight in qvec.items():
        for doc_id, tf in _doc_tf(index, term).items():
            if doc_id not in candidates or (doc_filter and not doc_filter(doc_id)):
                continue
            part = q_weight * log_tf(tf) / index.doc_norm[doc_id]
            scores[doc_id] += part
            contributions[doc_id][term] = part
    return scores, contributions


def _top(scores, contributions, k):
    top = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], item[0]))
    return [(doc_id, score, contributions[doc_id]) for doc_id, score in top]


def overlap_at_k(fast_results, exact_results, k=10):
    """Share of the exact top k that the fast method also returned in its top k."""
    exact = {d for d, _, _ in exact_results[:k]}
    if not exact:
        return 1.0
    fast = {d for d, _, _ in fast_results[:k]}
    return len(exact & fast) / len(exact)


# --- Index elimination ------------------------------------------------------

def high_idf_terms(qvec, index, min_idf=0.3):
    """Keep query terms with idf >= min_idf. Always keeps the rarest term."""
    idf = {t: math.log10(index.N / index.df(t)) for t in qvec}
    kept = [t for t in qvec if idf[t] >= min_idf]
    if not kept and qvec:
        kept = [max(qvec, key=lambda t: idf[t])]
    return kept, idf


def soft_and_candidates(index, terms, min_match):
    """Articles containing at least min_match of the given terms."""
    counts = defaultdict(int)
    for term in terms:
        for doc_id in _doc_tf(index, term):
            counts[doc_id] += 1
    return {d for d, c in counts.items() if c >= min_match}


def default_min_match(n_terms):
    """Lecture 7's example: with 4 query terms, ask for at least 3."""
    return max(1, math.ceil(0.75 * n_terms)) if n_terms >= 3 else 1


def search_index_elimination(query, index, k=10, min_idf=0.3, min_match=None, doc_filter=None):
    """lnc.ltc over high-idf terms and articles matching enough of them.

    If the soft AND leaves fewer than k articles, it's relaxed one term at a
    time, so a strict setting can't return an empty page.
    """
    qvec = query_vector(query, index)
    all_candidates = set()
    for term in qvec:
        all_candidates.update(_doc_tf(index, term))

    kept, idf = high_idf_terms(qvec, index, min_idf)
    need = default_min_match(len(kept)) if min_match is None else min_match
    need = min(need, len(kept)) if kept else 0
    candidates = soft_and_candidates(index, kept, need) if kept else set()
    while len(candidates) < k and need > 1:
        need -= 1
        candidates = soft_and_candidates(index, kept, need)

    kept_vec = {t: qvec[t] for t in kept}
    scores, contributions = _score(kept_vec, index, candidates, doc_filter)
    stats = {
        "method": "index elimination",
        "terms_kept": kept,
        "terms_dropped": [t for t in qvec if t not in kept],
        "idf": idf,
        "min_match": need,
        "scored": len(candidates),
        "full_candidates": len(all_candidates),
    }
    return _top(scores, contributions, k), stats


# --- Champion lists ---------------------------------------------------------
#
# For each term, precompute the r articles where the term carries the most
# weight (its "champion list"). At query time only articles in the champion
# lists of the query terms get scored. r is fixed when the lists are built,
# so a query can end up with fewer than k contenders; then we fall back to the
# full postings, which is Lecture 7's high list / low list idea.
#
# With static_scores (g(d), e.g. recency or PageRank) the lists are ordered by
# term weight + g(d) instead, so authoritative articles make the list too.

class ChampionLists:
    def __init__(self, index, r=50, static_scores=None):
        self.r = r
        self.lists = {}
        g = static_scores or {}
        for term in index.vocab:
            weights = {d: log_tf(tf) / index.doc_norm[d] for d, tf in _doc_tf(index, term).items()}
            ranked = heapq.nlargest(r, weights.items(), key=lambda item: (item[1] + g.get(item[0], 0.0), item[0]))
            self.lists[term] = [d for d, _ in ranked]

    def champions(self, term):
        return self.lists.get(term, [])


def search_champions(query, index, champions, k=10, doc_filter=None):
    """lnc.ltc scoring only the champion-list articles of the query terms."""
    qvec = query_vector(query, index)
    all_candidates = set()
    for term in qvec:
        all_candidates.update(_doc_tf(index, term))

    candidates = set()
    for term in qvec:
        candidates.update(champions.champions(term))
    if doc_filter:
        candidates = {d for d in candidates if doc_filter(d)}

    fell_back = False
    if len(candidates) < k:
        fell_back = True
        candidates = {d for d in all_candidates if not doc_filter or doc_filter(d)}

    scores, contributions = _score(qvec, index, candidates, doc_filter)
    stats = {
        "method": "champion lists",
        "r": champions.r,
        "scored": len(candidates),
        "full_candidates": len(all_candidates),
        "fell_back": fell_back,
    }
    return _top(scores, contributions, k), stats


# --- Recent-news tiers ------------------------------------------------------
#
# Lecture 7's tiered index: split articles into tiers of decreasing
# importance and search the top tier first, dropping to the next tier only
# if it doesn't yield k results. For news, importance is freshness: the
# newest articles are tier 0. "Now" is the newest article in the index, the
# same as the recency score, so results don't depend on the real clock.

from datetime import datetime  # noqa: E402

DEFAULT_TIER_DAYS = (2, 7, None)   # last 2 days, last week, everything older


def _newest(index):
    dates = [m["date"] for m in index.meta.values() if m.get("date")]
    return max(datetime.fromisoformat(d) for d in dates) if dates else None


class RecencyTiers:
    def __init__(self, index, tier_days=DEFAULT_TIER_DAYS, now=None):
        self.tier_days = tier_days
        now = now or _newest(index)
        self.tier_of = {}
        for doc_id, meta in index.meta.items():
            if not meta.get("date") or now is None:
                self.tier_of[doc_id] = len(tier_days) - 1
                continue
            age = (now - datetime.fromisoformat(meta["date"])).total_seconds() / 86400
            for i, limit in enumerate(tier_days):
                if limit is None or age <= limit:
                    self.tier_of[doc_id] = i
                    break

    def size(self, tier):
        return sum(1 for t in self.tier_of.values() if t == tier)


def search_tiered(query, index, tiers, k=10, doc_filter=None):
    """Search tier 0 first; add the next tier only if there are fewer than k results."""
    qvec = query_vector(query, index)
    all_candidates = set()
    for term in qvec:
        all_candidates.update(_doc_tf(index, term))

    results, stats_tiers, candidates = [], [], set()
    for tier in range(len(tiers.tier_days)):
        candidates |= {d for d in all_candidates if tiers.tier_of.get(d) == tier}
        scores, contributions = _score(qvec, index, candidates, doc_filter)
        results = _top(scores, contributions, k)
        stats_tiers.append(tier)
        if len(results) >= k:
            break

    stats = {
        "method": "recent-news tiers",
        "tiers_used": stats_tiers,
        "scored": len(candidates),
        "full_candidates": len(all_candidates),
    }
    return results, stats


# --- Cluster pruning --------------------------------------------------------
#
# Lecture 7's leaders and followers. Pick sqrt(N) articles at random as
# leaders; every other article follows the leader it's most similar to
# (cosine of their lnc vectors). At query time the query is compared with the
# leaders only, and just the clusters of the closest b leaders get scored.
# Random leaders are fast to pick and tend to land where the articles are
# dense, which is what makes this work.
#
# If the chosen clusters give fewer than k results, the next-closest leaders
# are added one at a time, the same fallback idea as champion lists.

import random  # noqa: E402


def doc_vectors(index):
    """{doc_id: {term: lnc weight}}, built from the postings."""
    vectors = defaultdict(dict)
    for term in index.vocab:
        for doc_id, tf in _doc_tf(index, term).items():
            vectors[doc_id][term] = log_tf(tf) / index.doc_norm[doc_id]
    return vectors


def _dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


class ClusterPruning:
    def __init__(self, index, n_leaders=None, seed=0):
        docs = sorted(index.meta)
        n_leaders = n_leaders or max(1, round(math.sqrt(len(docs))))
        self.leaders = sorted(random.Random(seed).sample(docs, min(n_leaders, len(docs))))
        self.vectors = doc_vectors(index)
        self.members = {leader: [leader] for leader in self.leaders}
        self.leader_of = {leader: leader for leader in self.leaders}
        for doc_id in docs:
            if doc_id in self.leader_of:
                continue
            vec = self.vectors.get(doc_id, {})
            # Ties (including no shared words at all) go to the first leader by id.
            best = max(self.leaders, key=lambda l: (_dot(vec, self.vectors.get(l, {})), -self.leaders.index(l)))
            self.members[best].append(doc_id)
            self.leader_of[doc_id] = best

    def nearest_leaders(self, qvec):
        """Leaders ordered by similarity to the query, closest first."""
        return sorted(self.leaders, key=lambda l: (-_dot(qvec, self.vectors.get(l, {})), l))


def search_clusters(query, index, clusters, k=10, b=1, doc_filter=None):
    """lnc.ltc scoring only the clusters of the b leaders closest to the query."""
    qvec = query_vector(query, index)
    all_candidates = set()
    for term in qvec:
        all_candidates.update(_doc_tf(index, term))

    order = clusters.nearest_leaders(qvec)
    used, candidates, results = 0, set(), []
    while used < len(order):
        candidates.update(clusters.members[order[used]])
        used += 1
        if used < b:
            continue
        scores, contributions = _score(qvec, index, candidates, doc_filter)
        results = _top(scores, contributions, k)
        if len(results) >= k:
            break

    stats = {
        "method": "cluster pruning",
        "leaders": len(clusters.leaders),
        "leaders_used": used,
        "scored": len(candidates & all_candidates),
        "full_candidates": len(all_candidates),
    }
    return results, stats


# --- Impact-ordered postings ------------------------------------------------
#
# Lecture 7: sort each term's postings by how much the term weighs in the
# article (its lnc weight), highest first, instead of by doc_id. A query then
# walks each list from the top and stops early: after a fixed number of
# articles, or once the weight has dropped below a share of the list's best
# weight. Query terms are processed in decreasing idf, so the words that
# matter most are read first. Articles past the stopping point only lose that
# term's (small) contribution, so the top k stays close to exact.

class ImpactOrdered:
    def __init__(self, index):
        self.lists = {}
        for term in index.vocab:
            weights = [(d, log_tf(tf) / index.doc_norm[d]) for d, tf in _doc_tf(index, term).items()]
            weights.sort(key=lambda item: (-item[1], item[0]))
            self.lists[term] = weights

    def postings(self, term):
        return self.lists.get(term, [])


def search_impact(query, index, impact, k=10, max_docs=20, min_share=0.0, doc_filter=None):
    """lnc.ltc reading each impact-ordered list only until it's no longer worth it.

    A list stops after max_docs articles (None means no limit) or once an
    article's weight falls below min_share x the best weight in that list. On
    news the weights inside one list are close together, so max_docs is the
    setting that actually saves work; min_share is there to compare.
    """
    qvec = query_vector(query, index)
    idf = {t: math.log10(index.N / index.df(t)) for t in qvec}
    all_candidates = set()
    for term in qvec:
        all_candidates.update(_doc_tf(index, term))

    scores = defaultdict(float)
    contributions = defaultdict(dict)
    read, total = 0, 0
    for term in sorted(qvec, key=lambda t: (-idf[t], t)):
        plist = impact.postings(term)
        total += len(plist)
        if not plist:
            continue
        floor = min_share * plist[0][1]
        for i, (doc_id, weight) in enumerate(plist):
            if (max_docs is not None and i >= max_docs) or weight < floor:
                break
            read += 1
            if doc_filter and not doc_filter(doc_id):
                continue
            part = qvec[term] * weight
            scores[doc_id] += part
            contributions[doc_id][term] = part

    stats = {
        "method": "impact-ordered postings",
        "postings_read": read,
        "postings_total": total,
        "scored": len(scores),
        "full_candidates": len(all_candidates),
    }
    return _top(scores, contributions, k), stats
