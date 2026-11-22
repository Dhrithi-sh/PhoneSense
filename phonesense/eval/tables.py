"""Results tables in Markdown and LaTeX with significance markers (T11).

    from phonesense.eval.tables import results_table
    t = results_table(["results/tfidf_lnc.ltc_dev.json", "results/bm25_tuned_dev.json"],
                      ["P@10", "MAP", "nDCG@10", "judged@10"], baseline="tfidf_lnc.ltc")
    print(t.markdown); open("results/table.tex", "w").write(t.latex)

* One row per run, one column per metric (mean over queries), 3 decimals; the best value of
  each column is bold.
* With a ``baseline``, every other run is compared to it per metric with the paired
  randomization test (:mod:`.significance`), Holm-corrected over the compared runs. A
  dagger (†) marks ``p_holm < alpha`` (default 0.05). ``judged@k`` is coverage, not quality,
  so it gets no marker.
* ``baseline`` may be a full run name (``bm25_default_dev``) or the name without its split
  suffix (``bm25_default``).

Adapted from our midterm project Dhvani-lingual (``dhvani/eval/experiments.py``:
``results_rows``, ``significance_rows``, ``_print_table``), extended with LaTeX output, Holm
correction and markers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phonesense.eval.metrics import canonical_metric
from phonesense.eval.significance import PERMUTATIONS, compare_many
from phonesense.io import read_results
from phonesense.schemas import Results

DAGGER = "†"


@dataclass
class Table:
    """A rendered results table and the significance tests behind its markers."""

    markdown: str
    latex: str
    significance: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def _load(r: Results | Mapping[str, Any] | str | Path) -> Results:
    if isinstance(r, Results):
        return r
    if isinstance(r, str | Path):
        return read_results(r)
    return Results.from_dict(dict(r))


def _strip_split(name: str) -> str:
    return re.sub(r"_(dev|test|all)$", "", name)


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def results_table(
    results: Sequence[Results | Mapping[str, Any] | str | Path],
    metrics: Sequence[str],
    baseline: str | None = None,
    alpha: float = 0.05,
    permutations: int = PERMUTATIONS,
    caption: str | None = None,
) -> Table:
    """Build the table; see the module docstring."""
    rs = [_load(r) for r in results]
    if not rs:
        raise ValueError("no results given")
    known = list(rs[0].metrics)
    cols = [canonical_metric(m, known) for m in metrics]
    for r in rs:
        missing = [c for c in cols if c not in r.metrics]
        if missing:
            raise KeyError(f"{r.run} has no metric(s) {missing}")

    base: Results | None = None
    if baseline is not None:
        matches = [r for r in rs if baseline in (r.run, _strip_split(r.run))]
        if not matches:
            raise ValueError(f"baseline {baseline!r} not among runs {[r.run for r in rs]}")
        base = matches[0]

    sig: dict[str, dict[str, dict[str, Any]]] = {}
    marked: set[tuple[str, str]] = set()
    if base is not None and len(rs) > 1:
        others = {r.run: r.per_query for r in rs if r is not base}
        for c in cols:
            if c.startswith("judged@"):
                continue
            sig[c] = compare_many(base.per_query, others, c, permutations)
            for run, row in sig[c].items():
                if row["p_randomization_holm"] < alpha:
                    marked.add((run, c))

    best = {c: max(r.metrics[c] for r in rs) for c in cols}

    def cell(r: Results, c: str, latex: bool) -> str:
        v = f"{r.metrics[c]:.3f}"
        if abs(r.metrics[c] - best[c]) < 1e-12 and len(rs) > 1:
            v = rf"\textbf{{{v}}}" if latex else f"**{v}**"
        if (r.run, c) in marked:
            v += r"$^\dagger$" if latex else DAGGER
        return v

    md = ["| Run | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r in rs:
        name = f"{r.run} (baseline)" if r is base else r.run
        md.append(f"| {name} | " + " | ".join(cell(r, c, False) for c in cols) + " |")
    if base is not None:
        md.append("")
        md.append(
            f"{DAGGER} p < {alpha} vs. {base.run} (paired randomization test, "
            f"{permutations} permutations, Holm-corrected). n = {len(base.per_query)} queries."
        )

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        "Run & " + " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for r in rs:
        tex.append(
            _latex_escape(r.run) + " & " + " & ".join(cell(r, c, True) for c in cols) + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    cap = caption or "Retrieval effectiveness."
    if base is not None:
        cap += (
            rf" $^\dagger$: $p<{alpha}$ vs.\ {_latex_escape(base.run)} (paired randomization "
            "test, Holm-corrected)."
        )
    tex += [rf"\caption{{{cap}}}", r"\end{table}"]
    return Table("\n".join(md) + "\n", "\n".join(tex) + "\n", sig)
