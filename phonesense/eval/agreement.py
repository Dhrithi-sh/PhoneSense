"""Inter-annotator agreement (T11, used by T09b: LLM judge vs. human labels).

    from phonesense.eval.agreement import agreement_report, cohen_kappa, weighted_kappa
    cohen_kappa([1, 0, 1], [1, 1, 1])                  # binary (or any nominal) labels
    weighted_kappa([3, 2, 0], [2, 2, 1])               # graded 0-3 labels, linear weights
    agreement_report("qrels/human.txt", "qrels/llm.txt")   # paths or {qid: {doc: grade}}

* Cohen's kappa: ``kappa = (p_o - p_e) / (1 - p_e)``, with ``p_o`` the observed agreement and
  ``p_e = sum_c p_a(c) * p_b(c)`` the agreement expected from each annotator's own label
  distribution. 1 = perfect, 0 = chance level, < 0 = worse than chance.
* Weighted kappa (linear): ``kappa_w = 1 - sum w_ij O_ij / sum w_ij E_ij`` with
  ``w_ij = |i - j| / (K - 1)`` over the K grades, ``O`` the observed and ``E`` the expected
  (outer product of marginals) proportions. Disagreeing by one grade costs less than by three.
* If both annotators use a single identical label everywhere, ``p_e = 1`` and kappa is
  defined here as 1.0 (perfect agreement), or 0.0 if they disagree.

Written new for PhoneSense (the midterm judged with a single annotator per query).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from phonesense.schemas import RELEVANCE_THRESHOLD, VALID_GRADES

QrelsLike = Mapping[str, Mapping[str, int]] | str | Path


def _check(a: Sequence[Any], b: Sequence[Any]) -> None:
    if len(a) != len(b):
        raise ValueError(f"label lists differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("need at least one pair of labels")


def cohen_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    """Cohen's kappa for two annotators' nominal labels (e.g. 0/1 relevance)."""
    _check(labels_a, labels_b)
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b), key=str)
    p_o = sum(x == y for x, y in zip(labels_a, labels_b, strict=True)) / n
    p_e = sum((list(labels_a).count(c) / n) * (list(labels_b).count(c) / n) for c in cats)
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def weighted_kappa(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
    categories: Sequence[int] = VALID_GRADES,
) -> float:
    """Linearly weighted kappa for ordinal labels (default grades 0-3)."""
    _check(labels_a, labels_b)
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    if k < 2:
        raise ValueError("need at least two categories")
    obs = np.zeros((k, k))
    for x, y in zip(labels_a, labels_b, strict=True):
        if x not in idx or y not in idx:
            raise ValueError(f"label {x if x not in idx else y!r} not in categories {categories}")
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denom = float((w * exp).sum())
    if denom == 0.0:
        return 1.0 if float((w * obs).sum()) == 0.0 else 0.0
    return 1.0 - float((w * obs).sum()) / denom


def _as_qrels(q: QrelsLike) -> Mapping[str, Mapping[str, int]]:
    if isinstance(q, str | Path):
        from phonesense.io import read_qrels

        return read_qrels(q)
    return q


def agreement_report(
    qrels_a: QrelsLike, qrels_b: QrelsLike, threshold: int = RELEVANCE_THRESHOLD
) -> dict[str, Any]:
    """Agreement between two qrels over the (qid, doc_id) pairs both judged.

    Returns ``n_shared``, ``n_only_a``, ``n_only_b``, ``exact_agreement`` (graded),
    ``binary_agreement`` and ``kappa_binary`` (grade >= threshold), ``kappa_weighted``
    (linear, grades 0-3) and the 4x4 ``confusion`` matrix (rows = a's grade, cols = b's).
    """
    a, b = _as_qrels(qrels_a), _as_qrels(qrels_b)
    pairs_a = {(q, d): g for q, docs in a.items() for d, g in docs.items()}
    pairs_b = {(q, d): g for q, docs in b.items() for d, g in docs.items()}
    shared = sorted(pairs_a.keys() & pairs_b.keys())
    if not shared:
        raise ValueError("the two qrels share no (qid, doc_id) pair")
    ga = [pairs_a[p] for p in shared]
    gb = [pairs_b[p] for p in shared]
    ba = [int(g >= threshold) for g in ga]
    bb = [int(g >= threshold) for g in gb]
    confusion = [[0] * len(VALID_GRADES) for _ in VALID_GRADES]
    for x, y in zip(ga, gb, strict=True):
        confusion[x][y] += 1
    return {
        "n_shared": len(shared),
        "n_only_a": len(pairs_a.keys() - pairs_b.keys()),
        "n_only_b": len(pairs_b.keys() - pairs_a.keys()),
        "exact_agreement": sum(x == y for x, y in zip(ga, gb, strict=True)) / len(shared),
        "binary_agreement": sum(x == y for x, y in zip(ba, bb, strict=True)) / len(shared),
        "kappa_binary": cohen_kappa(ba, bb),
        "kappa_weighted": weighted_kappa(ga, gb),
        "threshold": threshold,
        "confusion": confusion,
    }
