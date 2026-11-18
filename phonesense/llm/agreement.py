"""Agreement between the LLM judge and human labels, and the final merged qrels (owner: B, T09b).

The LLM judge (``judge.py``) produced our relevance labels; people labelled a shared sample.
Cohen's kappa measures how often they agree beyond what chance alone would give::

    kappa = (p_o - p_e) / (1 - p_e)

``p_o`` is the observed share of identical grades, ``p_e`` the share expected if both labelled
independently with their own grade frequencies. We report three numbers because the grades are
ordered (0-3) and a 2-vs-3 disagreement is milder than 0-vs-3:

* ``kappa``           plain Cohen's kappa on the four grades,
* ``weighted_kappa``  kappa with quadratic weights ``1 - ((i - j) / 3)^2`` (near misses cost less),
* ``kappa_binary``    kappa after cutting at the relevance threshold (grade >= 2 is relevant),
                      the view that matters for P, R, F1 and MAP.

Rules of thumb (Landis & Koch, 1977, verify before citing): below 0.4 is weak, 0.4-0.6
moderate, 0.6-0.8 substantial, above 0.8 near-perfect. With few pairs the estimate is noisy;
:func:`agreement_report` returns the pair count so the report can say so.

Final qrels (``merge_qrels``): where a human label exists it replaces the LLM label; everything
else keeps the LLM grade. Pairs are never invented.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from phonesense.schemas import RELEVANCE_THRESHOLD, VALID_GRADES

Qrels = Mapping[str, Mapping[str, int]]


def paired_grades(
    llm: Qrels, human: Qrels, qids: Iterable[str] | None = None
) -> list[tuple[str, str, int, int]]:
    """Pairs judged by both: ``[(qid, doc_id, llm_grade, human_grade), ...]``, sorted."""
    wanted = set(qids) if qids is not None else None
    out = []
    for qid, docs in human.items():
        if wanted is not None and qid not in wanted:
            continue
        for doc_id, h in docs.items():
            if doc_id in llm.get(qid, {}):
                out.append((qid, doc_id, llm[qid][doc_id], h))
    return sorted(out)


def _kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int], weight) -> float:
    """Generic (weighted) kappa; ``weight(i, j)`` is the agreement credit in [0, 1]."""
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    n = len(a)
    if n == 0:
        raise ValueError("no pairs to compare")
    ca, cb = Counter(a), Counter(b)
    po = sum(weight(x, y) for x, y in zip(a, b, strict=True)) / n
    pe = sum(weight(i, j) * ca[i] * cb[j] for i in labels for j in labels) / (n * n)
    if pe >= 1.0:  # both always gave the same single grade: perfect, but kappa is 0/0
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cohen_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = VALID_GRADES) -> float:
    """Plain Cohen's kappa."""
    return _kappa(a, b, labels, lambda i, j: 1.0 if i == j else 0.0)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int] = VALID_GRADES
) -> float:
    """Quadratic-weighted kappa for ordered grades."""
    span = max(labels) - min(labels)
    return _kappa(a, b, labels, lambda i, j: 1.0 - ((i - j) / span) ** 2)


def binarise(grades: Sequence[int], threshold: int = RELEVANCE_THRESHOLD) -> list[int]:
    """1 if ``grade >= threshold`` else 0."""
    return [1 if g >= threshold else 0 for g in grades]


def agreement_report(llm: Qrels, human: Qrels, qids: Iterable[str] | None = None) -> dict:
    """Agreement numbers over the pairs both judged (see the module docstring)."""
    pairs = paired_grades(llm, human, qids)
    if not pairs:
        raise ValueError("no pair has both an LLM and a human label")
    a = [p[2] for p in pairs]
    b = [p[3] for p in pairs]
    confusion = [[0] * len(VALID_GRADES) for _ in VALID_GRADES]
    for x, y in zip(a, b, strict=True):
        confusion[x][y] += 1  # rows: LLM grade, columns: human grade
    return {
        "n_pairs": len(pairs),
        "n_queries": len({p[0] for p in pairs}),
        "exact_agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(pairs),
        "kappa": cohen_kappa(a, b),
        "weighted_kappa": weighted_kappa(a, b),
        "kappa_binary": cohen_kappa(binarise(a), binarise(b), labels=(0, 1)),
        "confusion_llm_rows_human_cols": confusion,
    }


def merge_qrels(llm: Qrels, human: Qrels) -> dict[str, dict[str, int]]:
    """Final qrels: every LLM judgment, with a human label replacing it where one exists."""
    merged = {q: dict(d) for q, d in llm.items()}
    for qid, docs in human.items():
        merged.setdefault(qid, {}).update(docs)
    return {q: dict(sorted(d.items())) for q, d in sorted(merged.items())}
