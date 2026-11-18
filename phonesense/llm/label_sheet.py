"""Hand-labelling sheets: export pairs for people to grade, import their grades (owner: B, L-B).

Everyone in the team labels a share of the same test sample so that we can measure how well the
LLM judge agrees with people (kappa, ``agreement.py``). The sheet is a CSV that opens in any
spreadsheet. Two rules keep the comparison honest:

* the sheet **never shows the LLM grade** (it would anchor the human), nor which system found
  the phone;
* the sample is drawn with a fixed seed, so every person gets the same pairs next time and the
  shares are disjoint: pair ``i`` of the sample goes to share ``i % n_shares``.

Grades are the CONTRACT ones: 0 not relevant, 1 marginal, 2 relevant, 3 perfect.
"""

from __future__ import annotations

import csv
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from phonesense.llm.judge import phone_summary
from phonesense.llm.pooling import Pool
from phonesense.schemas import VALID_GRADES, Phone, Query

COLUMNS = ("qid", "query", "doc_id", "phone", "grade")
RUBRIC = (
    "0 = not relevant (wrong product / breaks a requirement), 1 = marginal, "
    "2 = relevant, 3 = perfect"
)


def sample_pairs(pool: Pool, n: int, seed: int) -> list[tuple[str, str]]:
    """Draw ``n`` (qid, doc_id) pairs, spread as evenly as possible over the pooled queries.

    Deterministic for a given ``pool`` and ``seed``; the result is sorted. If the pool has fewer
    than ``n`` pairs, all of them are returned.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = random.Random(seed)
    per_query = {qid: rng.sample(sorted(docs), len(docs)) for qid, docs in sorted(pool.items())}
    chosen: list[tuple[str, str]] = []
    depth = 0
    while len(chosen) < n and any(depth < len(d) for d in per_query.values()):
        for qid, docs in per_query.items():  # round-robin: one more pair per query each pass
            if depth < len(docs) and len(chosen) < n:
                chosen.append((qid, docs[depth]))
        depth += 1
    return sorted(chosen)


def share_of(pairs: Sequence[tuple[str, str]], index: int, n_shares: int) -> list[tuple[str, str]]:
    """Pairs ``index, index + n_shares, ...`` (``index`` is 1-based, ``1 <= index <= n_shares``)."""
    if n_shares < 1 or not 1 <= index <= n_shares:
        raise ValueError(f"share must satisfy 1 <= index <= n_shares, got {index}/{n_shares}")
    return list(pairs)[index - 1 :: n_shares]


def write_sheet(
    path: str | Path,
    pairs: Sequence[tuple[str, str]],
    queries: Mapping[str, Query],
    phones: Mapping[str, Phone],
) -> Path:
    """Write the sheet; the ``grade`` column is left empty for the labeller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for qid, doc_id in pairs:
            q = queries[qid]
            hints = f"{q.text}\n[{RUBRIC}]"
            w.writerow([qid, hints, doc_id, phone_summary(phones[doc_id]), ""])
    return path


def read_sheet(path: str | Path) -> dict[str, dict[str, int]]:
    """Read a filled sheet into ``{qid: {doc_id: grade}}``; blank grades are skipped.

    Raises ``ValueError`` (with the row number) for a grade outside 0-3 or a missing column.
    """
    out: dict[str, dict[str, int]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("qid", "doc_id", "grade") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
        for row_no, row in enumerate(reader, start=2):
            raw = (row["grade"] or "").strip()
            if not raw:
                continue
            if not re.fullmatch(r"[0-9]", raw) or int(raw) not in VALID_GRADES:
                raise ValueError(f"{path}: row {row_no}: grade must be 0-3, got {raw!r}")
            out.setdefault(row["qid"].strip(), {})[row["doc_id"].strip()] = int(raw)
    return out
