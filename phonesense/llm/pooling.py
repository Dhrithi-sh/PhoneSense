"""Pooling: the set of documents that get judged (owner: B, T09).

Judging every phone for every query is impossible, so we judge the union of the top ``depth``
documents of each system (Voorhees, TREC pooling). A pool is ``{qid: {doc_id: [systems]}}``
where ``systems`` lists the runs that put the document in their top ``depth``.

Pooling bias, to state honestly in the report: a document that no pooled system retrieves is
never judged and counts as not relevant. A system added after pooling is therefore scored
unfairly low, so pool *every* system we evaluate (including the baselines) before judging.

The pool is stored as ``runs/pool_<split>.tsv`` with the header ``qid<TAB>doc_id<TAB>systems``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from phonesense.io import Ranking

Pool = dict[str, dict[str, list[str]]]
HEADER = "qid\tdoc_id\tsystems"


def pool_runs(
    runs: Mapping[str, Mapping[str, Ranking]],
    depth: int,
    qids: Iterable[str] | None = None,
) -> Pool:
    """Union of the top ``depth`` docs per query over ``runs`` ({run_name: {qid: ranking}}).

    Only queries in ``qids`` (default: every query that appears in any run) are pooled. A run
    that has no ranking for a query simply contributes nothing to it.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    wanted = set(qids) if qids is not None else {q for r in runs.values() for q in r}
    pool: Pool = {}
    for name in sorted(runs):
        for qid, ranking in runs[name].items():
            if qid not in wanted:
                continue
            for doc_id, _ in ranking[:depth]:
                pool.setdefault(qid, {}).setdefault(doc_id, []).append(name)
    return {qid: dict(sorted(docs.items())) for qid, docs in sorted(pool.items())}


def queries_without_relevant(
    qrels: Mapping[str, Mapping[str, int]], threshold: int = 2
) -> list[str]:
    """Judged queries in which no phone reached ``threshold`` (default: grade 2 = relevant).

    Such a query is either unanswerable in the collection or its relevant phones sit below the
    pooling depth. Both look the same in the metrics (the query is dropped), so we pool these
    queries deeper (:func:`deepen_pool`) to tell them apart. Queries missing from ``qrels`` are
    not judged yet and are not listed.
    """
    return sorted(q for q, docs in qrels.items() if docs and max(docs.values()) < threshold)


def deepen_pool(
    pool: Pool, runs: Mapping[str, Mapping[str, Ranking]], depth: int, qids: Iterable[str]
) -> Pool:
    """Return ``pool`` with the queries in ``qids`` re-pooled to ``depth`` (a superset of their
    old entries, since a deeper top-k contains the shallower one). Other queries are unchanged."""
    deeper = pool_runs(runs, depth, qids)
    merged = {qid: dict(docs) for qid, docs in pool.items()}
    for qid, docs in deeper.items():
        merged[qid] = docs
    return dict(sorted(merged.items()))


def write_pool(path: str | Path, pool: Pool) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for qid, docs in sorted(pool.items()):
        for doc_id, systems in sorted(docs.items()):
            lines.append(f"{qid}\t{doc_id}\t{','.join(systems)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_pool(path: str | Path) -> Pool:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError(f"{path}: expected header {HEADER!r}")
    pool: Pool = {}
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{i}: expected 3 tab-separated fields, got {len(parts)}")
        qid, doc_id, systems = parts
        pool.setdefault(qid, {})[doc_id] = [s for s in systems.split(",") if s]
    return pool
