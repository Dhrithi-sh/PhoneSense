"""Retriever interface shared by every teammate (CONTRACT.md 4.7)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import Ranking, write_rankings
from phonesense.schemas import MAX_RUN_DEPTH, Phone, Query, run_name_split, validate_run_name


class Retriever(ABC):
    """Base class for first-stage retrievers.

    Subclasses set ``name`` (used as the default run-name prefix, e.g. "bm25_k1.2_b0.75")
    and implement ``fit`` and ``search``. ``search`` returns at most ``k`` (doc_id, score)
    pairs, highest score first, no duplicate doc_ids. Higher score = more relevant.
    """

    name: str = ""

    @abstractmethod
    def fit(self, phones: list[Phone]) -> None:
        """Build any index/model state from the phone collection."""

    @abstractmethod
    def search(self, query: Query, k: int = 100) -> Ranking:
        """Return up to k (doc_id, score) pairs sorted by score descending."""


def _default_run_name(retriever: Retriever, queries: Sequence[Query]) -> str:
    splits = {q.split for q in queries}
    split = splits.pop() if len(splits) == 1 else "all"
    return f"{retriever.name}_{split}"


def run_retriever(
    retriever: Retriever,
    queries: Sequence[Query],
    k: int = MAX_RUN_DEPTH,
    run_name: str | None = None,
    runs_dir: str | Path | None = None,
) -> Path:
    """Run ``retriever`` on every query and write ``<runs_dir>/<run_name>.run``.

    * ``run_name`` defaults to ``<retriever.name>_<split>`` (split = dev/test, or "all" if
      the queries mix splits). Its split suffix must match the queries' split(s).
    * ``runs_dir`` defaults to ``paths.runs_dir`` in config.yaml.
    * Results are sorted (score desc, doc_id asc for ties) and cut to ``k``.

    Returns the path of the written run file.
    """
    if not isinstance(retriever.name, str) or not retriever.name:
        raise ValueError(f"{type(retriever).__name__}.name must be a non-empty string")
    if not queries:
        raise ValueError("no queries given")
    if not 1 <= k <= MAX_RUN_DEPTH:
        raise ValueError(f"k must be in [1, {MAX_RUN_DEPTH}], got {k}")
    run_name = validate_run_name(run_name or _default_run_name(retriever, queries))
    split = run_name_split(run_name)
    if split != "all":
        wrong = [q.qid for q in queries if q.split != split]
        if wrong:
            raise ValueError(f"run name {run_name!r} says {split} but got queries {wrong}")
    rankings = {q.qid: list(retriever.search(q, k)) for q in queries}
    out_dir = Path(runs_dir) if runs_dir is not None else get_path("runs_dir")
    return write_rankings(out_dir / f"{run_name}.run", rankings, run_name, k)
