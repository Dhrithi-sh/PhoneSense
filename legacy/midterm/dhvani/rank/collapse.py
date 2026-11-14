"""Collapse duplicate wire stories into one result with "also in: ...".

When PTI or ANI copy runs in several papers, a search can return the same
story four times. Riya's dedup gives every later copy a dup_of pointing at
the earliest one, so every copy and its original share a cluster. Here we
keep only the best-ranked article of each cluster and list the other papers
that carried it, so the top k shows k different stories.

Ask the ranker for more than k results first (collapse_pool gives a sensible
number), then collapse, then cut to k.
"""

from collections import defaultdict


def cluster_id(doc_id, index):
    """The original's doc_id for a copy, the article's own id otherwise."""
    return index.meta.get(doc_id, {}).get("dup_of") or doc_id


def clusters(index):
    """{cluster_id: [doc_ids]} for every article in the index."""
    out = defaultdict(list)
    for doc_id in index.meta:
        out[cluster_id(doc_id, index)].append(doc_id)
    return out


def collapse_pool(k):
    """How many results to ask the ranker for before collapsing to k."""
    return 3 * k


def collapse_duplicates(results, index, k=None):
    """Keep the best-ranked article per cluster and note where else it ran.

    Each kept result's explain gets "duplicates" (the other doc_ids in its
    cluster) and "also_in" (their newspapers).
    """
    members = clusters(index)
    seen, out = set(), []
    for doc_id, score, explain in results:
        cid = cluster_id(doc_id, index)
        if cid in seen:
            continue
        seen.add(cid)
        others = sorted(d for d in members[cid] if d != doc_id)
        # lnc.ltc and BM25 give a plain {term: contribution} dict; keep the term
        # scores under "terms" so the extra info doesn't look like a term.
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        explain["duplicates"] = others
        explain["also_in"] = sorted({index.meta[d].get("source") for d in others if index.meta[d].get("source")})
        out.append((doc_id, score, explain))
    return out[:k] if k else out
