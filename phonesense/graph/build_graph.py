"""Phone graph (T16) for the PageRank prior.

Edge types (each builder returns a list of (src_doc_id, dst_doc_id, weight)):

    co_review  at least ``min_shared`` reviewers rated both phones   both ways  min(shared, cap)/cap

phones.jsonl has no ``bought_together`` field (CONTRACT.md 4.1), so co-purchase links from
the raw metadata are not used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations

from phonesense.schemas import Review

Edge = tuple[str, str, float]


def co_review_edges(reviews: Sequence[Review], doc_ids: Iterable[str], min_shared: int = 2,
                    cap: int = 5, max_user_phones: int = 30) -> list[Edge]:
    """Link two phones when at least ``min_shared`` reviewers rated both. Undirected, so each
    pair is written in both directions. Reviewers with more than ``max_user_phones`` phones
    are skipped: they add a quadratic number of pairs and little signal."""
    ids = set(doc_ids)
    by_user: dict[str, set[str]] = defaultdict(set)
    for r in reviews:
        if r.doc_id in ids:
            by_user[r.user_id].add(r.doc_id)
    shared: Counter[tuple[str, str]] = Counter()
    for docs in by_user.values():
        if 2 <= len(docs) <= max_user_phones:
            shared.update(combinations(sorted(docs), 2))
    edges: list[Edge] = []
    for (a, b), n in sorted(shared.items()):
        if n >= min_shared:
            weight = min(n, cap) / cap
            edges += [(a, b, weight), (b, a, weight)]
    return edges
