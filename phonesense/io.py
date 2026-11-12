"""Read/write helpers for every shared file format (CONTRACT.md section 4).

Rules for everyone:
  * Always read and write shared files through these helpers: they validate.
  * Writers validate *before* touching the file, so a bad record never produces a
    half-written file.
  * Files are UTF-8, "\\n" line endings, one record per line.
  * Errors name the file and the 1-based line number.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from phonesense.schemas import (
    GRAPH_EDGES_HEADER,
    MAX_RUN_DEPTH,
    REVIEWER_TRUST_HEADER,
    GraphEdge,
    Phone,
    Qrel,
    Query,
    Results,
    Review,
    ReviewerTrust,
    RunEntry,
    SchemaError,
    validate_run,
    validate_run_name,
)

PathLike = str | Path
Ranking = list[tuple[str, float]]  # [(doc_id, score), ...] score descending


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_lines(path: PathLike, lines: Iterable[str]) -> Path:
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line + "\n")
    return path


def _iter_lines(path: PathLike) -> Iterator[tuple[int, str]]:
    """Yield (line_no, stripped_line) for non-blank lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for i, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line:
                yield i, line


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    out = []
    for i, line in _iter_lines(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SchemaError(f"{path}:{i}: invalid JSON: {e}") from None
    return out


def write_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> Path:
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records]
    return _write_lines(path, lines)


def _read_records(path: PathLike, cls, id_attr: str) -> list:
    out, seen = [], set()
    for i, line in _iter_lines(path):
        where = f"{path}:{i}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SchemaError(f"{where}: invalid JSON: {e}") from None
        rec = cls.from_dict(obj, where)
        key = getattr(rec, id_attr)
        if key in seen:
            raise SchemaError(f"{where}: duplicate {id_attr} {key!r}")
        seen.add(key)
        out.append(rec)
    return out


def _write_records(path: PathLike, records: Sequence, cls, id_attr: str) -> Path:
    seen = set()
    dicts = []
    for n, rec in enumerate(records):
        d = rec.to_dict()
        cls.from_dict(d, f"record[{n}]")  # re-validate
        key = d[id_attr]
        if key in seen:
            raise SchemaError(f"record[{n}]: duplicate {id_attr} {key!r}")
        seen.add(key)
        dicts.append(d)
    return write_jsonl(path, dicts)


# --------------------------------------------------------------------------------------
# 4.1 Phones / 4.2 Reviews / 4.3 Queries  (JSONL)
# --------------------------------------------------------------------------------------


def read_phones(path: PathLike) -> list[Phone]:
    return _read_records(path, Phone, "doc_id")


def write_phones(path: PathLike, phones: Sequence[Phone]) -> Path:
    return _write_records(path, phones, Phone, "doc_id")


def read_reviews(path: PathLike, *, allow_synthetic: bool = False) -> list[Review]:
    """Read reviews. Real data must never contain is_synthetic=true; pass
    ``allow_synthetic=True`` only for files written by the attack simulator."""
    reviews = _read_records(path, Review, "review_id")
    if not allow_synthetic:
        syn = [r.review_id for r in reviews if r.is_synthetic]
        if syn:
            raise SchemaError(
                f"{path}: {len(syn)} synthetic reviews (e.g. {syn[0]}) in a file read with "
                "allow_synthetic=False"
            )
    return reviews


def write_reviews(path: PathLike, reviews: Sequence[Review]) -> Path:
    return _write_records(path, reviews, Review, "review_id")


def read_queries(path: PathLike, split: str | None = None) -> list[Query]:
    """Read queries, optionally keeping only one split ('dev' or 'test')."""
    queries = _read_records(path, Query, "qid")
    if split is not None:
        if split not in ("dev", "test"):
            raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
        queries = [q for q in queries if q.split == split]
    return queries


def write_queries(path: PathLike, queries: Sequence[Query]) -> Path:
    return _write_records(path, queries, Query, "qid")


# --------------------------------------------------------------------------------------
# 4.5 Qrels (TREC: "qid 0 doc_id grade")
# --------------------------------------------------------------------------------------


def read_qrels_entries(path: PathLike) -> list[Qrel]:
    out, seen = [], set()
    for i, line in _iter_lines(path):
        q = Qrel.from_fields(line.split(), f"{path}:{i}")
        if (q.qid, q.doc_id) in seen:
            raise SchemaError(f"{path}:{i}: duplicate judgment for ({q.qid}, {q.doc_id})")
        seen.add((q.qid, q.doc_id))
        out.append(q)
    return out


def read_qrels(path: PathLike) -> dict[str, dict[str, int]]:
    """Return {qid: {doc_id: grade}}."""
    out: dict[str, dict[str, int]] = {}
    for q in read_qrels_entries(path):
        out.setdefault(q.qid, {})[q.doc_id] = q.grade
    return out


def write_qrels(path: PathLike, qrels: Mapping[str, Mapping[str, int]] | Sequence[Qrel]) -> Path:
    """Write qrels from {qid: {doc_id: grade}} or a list of Qrel. Sorted by qid, doc_id."""
    if isinstance(qrels, Mapping):
        entries = [Qrel(qid, d, g) for qid, docs in qrels.items() for d, g in docs.items()]
    else:
        entries = list(qrels)
    seen = set()
    for q in entries:
        q.validate()
        if (q.qid, q.doc_id) in seen:
            raise SchemaError(f"duplicate judgment for ({q.qid}, {q.doc_id})")
        seen.add((q.qid, q.doc_id))
    entries.sort(key=lambda q: (q.qid, q.doc_id))
    return _write_lines(path, (q.to_line() for q in entries))


# --------------------------------------------------------------------------------------
# 4.6 Runs (TREC: "qid Q0 doc_id rank score run_name")
# --------------------------------------------------------------------------------------


def rankings_to_entries(
    rankings: Mapping[str, Ranking], run_name: str, k: int = MAX_RUN_DEPTH
) -> list[RunEntry]:
    """Turn {qid: [(doc_id, score), ...]} into ranked RunEntry objects.

    Sorts by score descending, ties broken by doc_id ascending (deterministic), keeps the
    top ``k`` (<= 100). Duplicate doc_ids within a query raise SchemaError (a retriever
    bug, not something to hide). Queries are written in qid order.
    """
    validate_run_name(run_name)
    if not 1 <= k <= MAX_RUN_DEPTH:
        raise ValueError(f"k must be in [1, {MAX_RUN_DEPTH}], got {k}")
    entries = []
    for qid in sorted(rankings):
        ranking = list(rankings[qid])
        doc_ids = [d for d, _ in ranking]
        if len(set(doc_ids)) != len(doc_ids):
            raise SchemaError(f"{run_name}: duplicate doc_ids in ranking for {qid}")
        ranking.sort(key=lambda ds: (-float(ds[1]), ds[0]))
        for rank, (doc_id, score) in enumerate(ranking[:k], start=1):
            entries.append(RunEntry(qid, doc_id, rank, float(score), run_name))
    return entries


def write_run(path: PathLike, entries: Sequence[RunEntry]) -> Path:
    """Write a run file. Entries must already be valid (grouped by qid, ranks 1..n,
    scores non-increasing, no duplicate docs, <= 100 per query); nothing is fixed up."""
    entries = list(entries)
    validate_run(entries)
    return _write_lines(path, (e.to_line() for e in entries))


def write_rankings(
    path: PathLike, rankings: Mapping[str, Ranking], run_name: str, k: int = MAX_RUN_DEPTH
) -> Path:
    """Convenience for rerankers/fusion: rankings dict -> valid run file."""
    return write_run(path, rankings_to_entries(rankings, run_name, k))


def read_run_entries(path: PathLike) -> list[RunEntry]:
    entries = [RunEntry.from_fields(line.split(), f"{path}:{i}") for i, line in _iter_lines(path)]
    try:
        validate_run(entries)
    except SchemaError as e:
        raise SchemaError(f"{path}: {e}") from None
    return entries


def read_run(path: PathLike) -> dict[str, Ranking]:
    """Return {qid: [(doc_id, score), ...]} in rank order."""
    out: dict[str, Ranking] = {}
    for e in read_run_entries(path):
        out.setdefault(e.qid, []).append((e.doc_id, e.score))
    return out


# --------------------------------------------------------------------------------------
# 4.8 Graph files (TSV with header)
# --------------------------------------------------------------------------------------


def _read_tsv(path: PathLike, header: tuple[str, ...]) -> Iterator[tuple[str, list[str]]]:
    first = True
    for i, line in _iter_lines(path):
        parts = line.split("\t")
        if first:
            if tuple(parts) != header:
                expected = "<TAB>".join(header)
                raise SchemaError(f"{path}:{i}: header must be {expected}")
            first = False
            continue
        if len(parts) != len(header):
            raise SchemaError(f"{path}:{i}: expected {len(header)} tab-separated fields")
        yield f"{path}:{i}", parts
    if first:
        raise SchemaError(f"{path}: empty file (header required)")


def _parse_float(s: str, where: str) -> float:
    try:
        return float(s)
    except ValueError:
        raise SchemaError(f"{where}: not a number: {s!r}") from None


def read_graph_edges(path: PathLike) -> list[GraphEdge]:
    out, seen = [], set()
    for where, (src, dst, w) in _read_tsv(path, GRAPH_EDGES_HEADER):
        e = GraphEdge(src, dst, _parse_float(w, where)).validate(where)
        if (src, dst) in seen:
            raise SchemaError(f"{where}: duplicate edge {src} -> {dst}")
        seen.add((src, dst))
        out.append(e)
    return out


def write_graph_edges(path: PathLike, edges: Sequence[GraphEdge]) -> Path:
    seen = set()
    for n, e in enumerate(edges):
        e.validate(f"edge[{n}]")
        if (e.src_doc_id, e.dst_doc_id) in seen:
            raise SchemaError(f"edge[{n}]: duplicate edge {e.src_doc_id} -> {e.dst_doc_id}")
        seen.add((e.src_doc_id, e.dst_doc_id))
    lines = ["\t".join(GRAPH_EDGES_HEADER)]
    lines += [f"{e.src_doc_id}\t{e.dst_doc_id}\t{float(e.weight)!r}" for e in edges]
    return _write_lines(path, lines)


def read_reviewer_trust(path: PathLike) -> dict[str, float]:
    out: dict[str, float] = {}
    for where, (uid, t) in _read_tsv(path, REVIEWER_TRUST_HEADER):
        rt = ReviewerTrust(uid, _parse_float(t, where)).validate(where)
        if uid in out:
            raise SchemaError(f"{where}: duplicate user_id {uid}")
        out[uid] = rt.trust
    return out


def write_reviewer_trust(path: PathLike, trust: Mapping[str, float]) -> Path:
    for uid, t in trust.items():
        ReviewerTrust(uid, t).validate(f"trust[{uid}]")
    lines = ["\t".join(REVIEWER_TRUST_HEADER)]
    lines += [f"{uid}\t{float(t)!r}" for uid, t in sorted(trust.items())]
    return _write_lines(path, lines)


# --------------------------------------------------------------------------------------
# 4.9 Results (JSON)
# --------------------------------------------------------------------------------------


def read_results(path: PathLike) -> Results:
    with Path(path).open(encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
        except json.JSONDecodeError as e:
            raise SchemaError(f"{path}: invalid JSON: {e}") from None
    return Results.from_dict(obj, str(path))


def write_results(path: PathLike, results: Results) -> Path:
    d = results.to_dict()
    Results.from_dict(d)
    path = Path(path)
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path
