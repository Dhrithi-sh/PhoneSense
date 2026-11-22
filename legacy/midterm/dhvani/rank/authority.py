"""Authority: PageRank and "first to publish" in the static score g(d).

Lecture 7's net score adds a query-independent quality score g(d) to cosine.
For news we build g(d) from three signals:

- recency: newer stories matter more (same formula as the net score)
- PageRank over the links between crawled articles: an article that other
  articles link to is treated as more authoritative. Computed by power
  iteration with damping 0.85. Articles with no outgoing links spread their
  vote evenly over every article, so the scores always add up to 1.
- first to publish: when several papers carry the same wire story, Riya's
  dedup marks later copies with dup_of pointing at the earliest one. The
  original gets credit; the copies don't.

    g(d) = a * recency + b * PageRank (scaled to [0, 1]) + c * original flag

with a + b + c = 1, so g(d) stays in [0, 1].
"""

from datetime import datetime

from dhvani.rank.scoring import recency

DAMPING = 0.85
DEFAULT_MIX = {"recency": 0.5, "pagerank": 0.3, "original": 0.2}


def link_graph(index):
    """{doc_id: [linked doc_ids]}, keeping only links to articles in the index."""
    docs = set(index.meta)
    graph = {}
    for doc_id, meta in index.meta.items():
        targets = [t for t in (meta.get("links") or []) if t in docs and t != doc_id]
        graph[doc_id] = list(dict.fromkeys(targets))
    return graph


def pagerank(graph, damping=DAMPING, tol=1e-10, max_iter=200):
    """PageRank by power iteration. Returns {doc_id: score}, summing to 1."""
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}
    pr = {d: 1.0 / n for d in nodes}
    for _ in range(max_iter):
        dangling = sum(pr[d] for d in nodes if not graph[d])
        new = {d: (1 - damping) / n + damping * dangling / n for d in nodes}
        for d in nodes:
            out = graph[d]
            if out:
                share = damping * pr[d] / len(out)
                for t in out:
                    new[t] += share
        if sum(abs(new[d] - pr[d]) for d in nodes) < tol:
            pr = new
            break
        pr = new
    return pr


def originals(index):
    """Articles that later copies point to with dup_of: the first to publish."""
    heads = {m.get("dup_of") for m in index.meta.values() if m.get("dup_of")}
    return {d for d in heads if d in index.meta}


def static_scores(index, mix=None, now=None):
    """g(d) in [0, 1] for every article, plus the parts it was built from.

    Returns (g, parts) where parts[doc_id] = {"recency", "pagerank", "original"}.
    """
    mix = {**DEFAULT_MIX, **(mix or {})}
    if now is None:
        dates = [m["date"] for m in index.meta.values() if m.get("date")]
        now = max(datetime.fromisoformat(d) for d in dates) if dates else None
    pr = pagerank(link_graph(index))
    top = max(pr.values()) if pr else 0.0
    firsts = originals(index)
    g, parts = {}, {}
    for doc_id, meta in index.meta.items():
        rec = recency(meta.get("date"), now) if now else 0.0
        pr_scaled = pr.get(doc_id, 0.0) / top if top else 0.0
        orig = 1.0 if doc_id in firsts else 0.0
        parts[doc_id] = {"recency": rec, "pagerank": pr_scaled, "original": orig}
        g[doc_id] = mix["recency"] * rec + mix["pagerank"] * pr_scaled + mix["original"] * orig
    return g, parts
