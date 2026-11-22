"""Evaluate a run file against qrels and write the contract's results JSON (T11).

    from phonesense.eval.evaluate import evaluate_run
    res = evaluate_run("runs/bm25_tuned_dev.run", "qrels/qrels.txt")
    # -> writes results/bm25_tuned_dev.json

Which queries are scored:

* The split comes from the run name (``..._dev`` -> dev). ``split=`` may be passed but must
  agree with the run name (the results JSON requires it). For ``dev`` / ``test`` the qids are
  taken from ``queries/queries.jsonl``; for ``all``, every query (or every judged query if
  there is no queries file).
* Queries of that split with **no relevant document** in the qrels (grade >= threshold) are
  excluded from the averages and listed in ``meta.excluded_qids``. Queries with no judgments
  at all are listed in ``meta.unjudged_qids``.
* A scored query the run has no lines for gets all-zero metrics (it still counts).
* Aggregates are the mean over scored queries; ``per_query`` holds each query's values
  (under ``MAP``, that query's AP), as CONTRACT.md 4.9 requires.

Adapted from our midterm project Dhvani-lingual (``dhvani/eval/metrics.py``: ``evaluate``,
``dhvani/eval/experiments.py``: ``score_run``), rewritten around ``phonesense.io`` and the
``Results`` schema.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phonesense.config import get_path, repo_root
from phonesense.eval.metrics import DEFAULT_KS, default_threshold, metric_names, score_query
from phonesense.io import read_qrels, read_queries, read_run, read_run_entries, write_results
from phonesense.schemas import Results, run_name_split

QrelsDict = dict[str, dict[str, int]]


def resolve_qrels_path(path: str | Path | None = None, *, quiet: bool = False) -> Path:
    """``path`` if given, else ``qrels/qrels.txt``, else ``qrels/llm.txt`` (with a warning).

    Raises ``FileNotFoundError`` with a clear message if none exists.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"qrels file not found: {p}")
        return p
    final, llm = get_path("qrels"), get_path("qrels_llm")
    if final.exists():
        return final
    if llm.exists():
        if not quiet:
            print(f"warning: {final} not found; falling back to LLM qrels {llm}", file=sys.stderr)
        return llm
    raise FileNotFoundError(
        f"no qrels yet: neither {final} nor {llm} exists (qrels are produced in T09)."
    )


def split_qids(split: str, queries_path: str | Path | None = None) -> set[str] | None:
    """qids of ``split`` from the queries file; ``None`` for ``all`` without a queries file."""
    path = Path(queries_path) if queries_path is not None else get_path("queries")
    if not path.exists():
        if split == "all":
            return None
        raise FileNotFoundError(
            f"{path} not found: it is needed to select the {split} queries (queries are T08)."
        )
    return {q.qid for q in read_queries(path, split=None if split == "all" else split)}


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    qids: set[str] | None = None,
    ks: Sequence[int] = DEFAULT_KS,
    judged_ks: Sequence[int] = (10,),
    threshold: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float], list[str], list[str]]:
    """Score rankings ``{qid: [doc_id, ...]}``.

    Returns ``(per_query, means, excluded_qids, unjudged_qids)``; see the module docstring.
    """
    t = default_threshold() if threshold is None else threshold
    candidates = set(qrels) if qids is None else set(qids)
    unjudged = sorted(q for q in candidates if not qrels.get(q))
    excluded = sorted(
        q for q in candidates if qrels.get(q) and not any(g >= t for g in qrels[q].values())
    )
    scored = sorted(candidates - set(unjudged) - set(excluded))
    per_query = {
        q: score_query(list(rankings.get(q, [])), qrels[q], ks, judged_ks, t) for q in scored
    }
    names = metric_names(ks, judged_ks)
    means = (
        {n: sum(m[n] for m in per_query.values()) / len(per_query) for n in names}
        if per_query
        else {}
    )
    return per_query, means, excluded, unjudged


def _display_path(p: Path) -> str:
    try:
        return p.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(p)


def run_name_of(run_path: str | Path) -> str:
    """The run name from the file's lines (or its file name if the run is empty)."""
    entries = read_run_entries(run_path)
    return entries[0].run_name if entries else Path(run_path).stem


def evaluate_run(
    run_path: str | Path,
    qrels_path: str | Path | None = None,
    split: str | None = None,
    *,
    queries_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    write: bool = True,
    ks: Sequence[int] = DEFAULT_KS,
    judged_ks: Sequence[int] = (10,),
    threshold: int | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one run file; return (and by default write) the results JSON as a dict.

    ``qrels_path`` defaults to :func:`resolve_qrels_path`. Output goes to
    ``<out_dir>/<run_name>.json`` (``out_dir`` defaults to ``paths.results_dir``).
    Raises ``ValueError`` if no query can be scored.
    """
    run_path = Path(run_path)
    qrels_file = resolve_qrels_path(qrels_path)
    run_name = run_name_of(run_path)
    run_split = run_name_split(run_name)
    if split is not None and split != run_split:
        raise ValueError(
            f"--split {split} does not match run {run_name!r} (split {run_split}); the results "
            "JSON split must equal the run-name suffix"
        )
    split = run_split
    t = default_threshold() if threshold is None else threshold
    qrels: QrelsDict = read_qrels(qrels_file)
    qids = split_qids(split, queries_path)
    rankings = {q: [d for d, _ in docs] for q, docs in read_run(run_path).items()}
    per_query, means, excluded, unjudged = evaluate_rankings(
        rankings, qrels, qids, ks, judged_ks, t
    )
    if not per_query:
        raise ValueError(
            f"no query of split {split} has a relevant judgment in {qrels_file}; nothing to score"
        )
    full_meta: dict[str, Any] = {
        "relevance_threshold": t,
        "ndcg_gain": "2^grade - 1",
        "unjudged_docs": "non-relevant",
        "n_queries": len(per_query),
        "excluded_qids": excluded,
        "unjudged_qids": unjudged,
        "run_qids_not_evaluated": sorted(set(rankings) - set(per_query)),
    }
    full_meta.update(meta or {})
    res = Results(
        run=run_name,
        qrels=_display_path(qrels_file),
        split=split,
        metrics=means,
        per_query=per_query,
        meta=full_meta,
    )
    out = res.to_dict()
    Results.from_dict(out)  # validate against the contract
    if write:
        out_path = Path(out_dir) if out_dir is not None else get_path("results_dir")
        write_results(out_path / f"{run_name}.json", res)
    return out
