"""Net score: cosine plus zone, proximity and recency signals (Lecture 7).

    net(q, d) = cosine(q, d)
              + w_zone * zone_score(q, d)
              + w_prox * proximity(q, d)
              + w_recency * g(d)

- zone_score: weighted zone scoring. Each zone has a weight (headline counts
  more than body) and scores the fraction of query terms found in it.
- proximity: users prefer documents where the query words sit close together.
  We find the smallest window containing all matched query terms and score it
  as (number of terms) / (window length), so adjacent words give 1.0.
- g(d): query-independent static quality. For news that's recency,
  g(d) = exp(-age_in_days / tau), so a story from today gets 1.0.
"""

import heapq
import math
from collections import defaultdict
from datetime import datetime

from dhvani.rank.vsm import ZONES, cosine_scores, query_vector

ZONE_WEIGHTS = {"headline": 0.7, "body": 0.3}

DEFAULT_WEIGHTS = {"zone": 0.2, "prox": 0.1, "recency": 0.1}

RECENCY_TAU_DAYS = 7.0


def smallest_window(positions_by_term):
    """Length of the smallest span of positions that covers every term.

    positions_by_term: {term: [positions]}. Returns None if fewer than two
    terms are present (a single word has no proximity to measure).
    """
    terms = [t for t, pos in positions_by_term.items() if pos]
    if len(terms) < 2:
        return None
    events = sorted((p, t) for t in terms for p in positions_by_term[t])
    need = len(terms)
    counts = defaultdict(int)
    covered = 0
    best = None
    left = 0
    for right, (pos_r, term_r) in enumerate(events):
        counts[term_r] += 1
        if counts[term_r] == 1:
            covered += 1
        while covered == need:
            pos_l, term_l = events[left]
            span = pos_r - pos_l + 1
            if best is None or span < best:
                best = span
            counts[term_l] -= 1
            if counts[term_l] == 0:
                covered -= 1
            left += 1
    return best


def recency(date_str, now):
    """g(d) = exp(-age_days / tau), clipped to [0, 1]."""
    if not date_str:
        return 0.0
    published = datetime.fromisoformat(date_str)
    age_days = max((now - published).total_seconds() / 86400, 0.0)
    return math.exp(-age_days / RECENCY_TAU_DAYS)


def _latest_date(index):
    dates = [m["date"] for m in index.meta.values() if m.get("date")]
    return max(datetime.fromisoformat(d) for d in dates)


def rank(query, index, k=10, weights=None, now=None, doc_filter=None, static=None):
    """Top k documents by net score.

    Returns [(doc_id, net_score, explain)] where explain holds every part of
    the score so --explain can show where the number came from.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Default "now" is the newest article in the index, so results don't
    # drift depending on when you run them.
    now = now or _latest_date(index)

    cos, contributions = cosine_scores(query, index)
    terms = list(query_vector(query, index))

    # Gather positions per document and zone for the matched terms.
    positions = defaultdict(lambda: {zone: defaultdict(list) for zone in ZONES})
    for term in terms:
        for zone in ZONES:
            for doc_id, _tf, pos in index.postings(term, zone):
                positions[doc_id][zone][term] = pos

    scored = []
    for doc_id, cosine in cos.items():
        if doc_filter and not doc_filter(doc_id):
            continue
        by_zone = positions[doc_id]

        zone_score = sum(
            ZONE_WEIGHTS[zone] * len(by_zone[zone]) / len(terms) for zone in ZONES
        ) if terms else 0.0

        windows = [w for w in (smallest_window(by_zone[zone]) for zone in ZONES) if w]
        matched = max(len(by_zone[zone]) for zone in ZONES)
        prox = matched / min(windows) if windows else 0.0

        # g(d): plain recency by default, or a full static score (recency +
        # PageRank + first to publish) when one is passed in.
        g = static[doc_id] if static is not None else recency(index.meta[doc_id].get("date"), now)

        net = cosine + weights["zone"] * zone_score + weights["prox"] * prox + weights["recency"] * g
        explain = {
            "cosine": cosine,
            "terms": contributions[doc_id],
            "zone": zone_score,
            "proximity": prox,
            "recency": g,
            "static": static is not None,
            "net": net,
        }
        scored.append((doc_id, net, explain))

    return heapq.nlargest(k, scored, key=lambda item: (item[1], item[0]))
