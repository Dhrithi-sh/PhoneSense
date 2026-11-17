"""Phone graph (T16) for the PageRank prior.

Edge types (each builder returns a list of (src_doc_id, dst_doc_id, weight)):

    co_review  at least ``min_shared`` reviewers rated both phones   both ways  min(shared, cap)/cap
    knn        dst is among src's k most similar phones (embeddings)  src -> dst  cosine similarity

phones.jsonl has no ``bought_together`` field (CONTRACT.md 4.1), so co-purchase links from
the raw metadata are not used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np

from phonesense.graph.pagerank import Graph
from phonesense.schemas import GraphEdge, Review

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


def knn_edges(doc_ids: Sequence[str], vectors, k: int = 8, chunk: int = 1000) -> list[Edge]:
    """Each phone points to its k most cosine-similar phones. Non-positive similarities are
    dropped. Computed in chunks of rows to bound memory."""
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or len(doc_ids) != len(x):
        raise ValueError("vectors must be a 2-D array with one row per doc_id")
    n = len(doc_ids)
    k = min(k, n - 1)
    if k <= 0:
        return []
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms == 0, 1.0, norms)
    edges: list[Edge] = []
    for start in range(0, n, chunk):
        sims = x[start:start + chunk] @ x.T
        rows = np.arange(len(sims))
        sims[rows, start + rows] = -np.inf  # never pick yourself
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for i in rows:
            for j in sorted(top[i], key=lambda j, i=i: (-sims[i, j], doc_ids[j])):
                if sims[i, j] > 0:
                    edges.append((doc_ids[start + i], doc_ids[j], float(sims[i, j])))
    return edges


def _largest_component(n: int, src: np.ndarray, dst: np.ndarray) -> int:
    """Size of the largest weakly connected component (union-find)."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(src.tolist(), dst.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return max(Counter(find(i) for i in range(n)).values()) if n else 0


def graph_stats(graph: Graph, edges_by_type: dict[str, list[Edge]]) -> dict:
    """Edge counts and coverage per type, plus dangling/isolated phones of the merged graph.
    These numbers go into the report's dataset table."""
    n = len(graph.doc_ids)
    stats: dict = {"phones": n, "types": {}}
    for name, edges in edges_by_type.items():
        touched = {e[0] for e in edges} | {e[1] for e in edges}
        stats["types"][name] = {"edges": len(edges), "phones_with_edge": len(touched),
                                "coverage": round(len(touched) / n, 4) if n else 0.0}
    has_out = np.zeros(n, dtype=bool)
    has_out[graph.src] = True
    touched = has_out.copy()
    touched[graph.dst] = True
    stats["merged"] = {
        "edges": int(len(graph.src)),
        "dangling_phones": int((~has_out).sum()),  # no out-edges
        "isolated_phones": int((~touched).sum()),  # no edges at all
        "largest_component": _largest_component(n, graph.src, graph.dst),
    }
    return stats


def build_graph(doc_ids: Iterable[str], reviews: Sequence[Review] | None = None,
                embeddings: tuple[Sequence[str], np.ndarray] | None = None, knn_k: int = 8,
                co_review_min_shared: int = 2) -> tuple[list[GraphEdge], Graph, dict]:
    """Merge every available edge type (weights of repeated pairs are summed).

    ``embeddings`` is (doc_ids, matrix) or None. A missing input just leaves its edge type
    out, so the graph can be built before dense embeddings exist. Returns the edges in
    contract form (for graph_edges.tsv), the Graph, and its stats.
    """
    doc_ids = sorted(set(doc_ids))
    id_set = set(doc_ids)
    edges_by_type: dict[str, list[Edge]] = {}
    if reviews is not None:
        edges_by_type["co_review"] = co_review_edges(reviews, doc_ids,
                                                     min_shared=co_review_min_shared)
    if embeddings is not None:
        emb_ids, matrix = embeddings
        keep = [i for i, d in enumerate(emb_ids) if d in id_set]
        edges_by_type["knn"] = knn_edges([emb_ids[i] for i in keep],
                                         np.asarray(matrix)[keep], k=knn_k)
    graph = Graph.from_edges(doc_ids, [e for edges in edges_by_type.values() for e in edges])
    edges = [GraphEdge(graph.doc_ids[s], graph.doc_ids[d], float(w))
             for s, d, w in zip(graph.src, graph.dst, graph.w, strict=True)]
    return edges, graph, graph_stats(graph, edges_by_type)
