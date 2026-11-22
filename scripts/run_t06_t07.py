"""Filtered and phrase-boosted BM25 runs, their evaluation and the constraint subset (T06/T07).

    python scripts/run_t06_t07.py --split dev      # after tune_bm25.py and tune_phrase.py
    python scripts/run_t06_t07.py --split test     # ONCE, with the params chosen on dev

Runs (``runs/<name>_<split>.run``), all on the tuned BM25 (``bm25.tuned``):

* ``bm25_tuned``                 baseline (rewritten here so all four share one setup)
* ``bm25_tuned+filters``         pre-filter with the query's filter (queries.jsonl, else parser)
* ``bm25_tuned+phrase``          + phrase boost (``phrase.tuned.weight``, else ``phrase.weight``)
* ``bm25_tuned+filters+phrase``  both

The filtered runs also write ``runs/<run_name>.filters.json``: per query the filter, its
source (``queries.jsonl`` / ``parser`` / ``none``), the text given to the ranker and the
number of phones that pass.

With qrels available, each run is evaluated (results/<run>.json) and compared with
``bm25_tuned`` (paired randomization test + t-test, Holm over the three systems) on nDCG@10
and MAP: ``results/compare_t06_t07_<split>.json`` and ``results/table_t06_t07_<split>.md``
(+ .tex). The constraint-subset analysis (the same metrics over queries *with* a filter vs.
*without*) goes to ``results/filters_subset_<split>.json`` and ``.md``.

``--split test`` refuses to overwrite existing test runs unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from phonesense.config import get_config, get_path
from phonesense.eval.cli import write_table
from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path
from phonesense.eval.significance import compare, compare_many
from phonesense.index import StaleIndexError, open_index
from phonesense.io import read_phones, read_queries
from phonesense.retrieval.base import Retriever, run_retriever
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.filters import FilteredRetriever
from phonesense.retrieval.positional import PhraseBoostRetriever

BASE = "bm25_tuned"
METRICS = ("P@10", "MAP", "nDCG@10", "judged@10")
SUBSET_METRICS = ("P@10", "MAP", "nDCG@10", "judged@10")


def build(index, phones, tuned: dict, phrase_weight: float) -> list[Retriever]:
    base = BM25Retriever(tuned["k1"], tuned["b"], index=index, name=BASE)
    phrase = PhraseBoostRetriever(base, phrase_weight, name=f"{BASE}+phrase")
    return [
        base,
        FilteredRetriever(base, phones, mode="pre", name=f"{BASE}+filters"),
        phrase,
        FilteredRetriever(phrase, phones, mode="pre", name=f"{BASE}+filters+phrase"),
    ]


def subset_analysis(results: dict[str, dict], constrained: set[str]) -> dict[str, Any]:
    """Mean metrics over constrained vs. unconstrained queries, per run (+ p vs. baseline)."""
    out: dict[str, Any] = {"constrained_qids": sorted(constrained), "runs": {}}
    base_pq = results[next(n for n in results if n.startswith(f"{BASE}_"))]["per_query"]
    for run, res in results.items():
        pq = res["per_query"]
        row: dict[str, Any] = {}
        for label, keep in (("with_constraints", True), ("without_constraints", False)):
            qids = [q for q in pq if (q in constrained) == keep]
            sub = {"n": len(qids)}
            for m in SUBSET_METRICS:
                sub[m] = sum(pq[q][m] for q in qids) / len(qids) if qids else None
            if qids and not run.startswith(f"{BASE}_"):
                c = compare({q: base_pq[q] for q in qids}, {q: pq[q] for q in qids}, "nDCG@10")
                sub["p_nDCG@10_vs_base"] = c["p_randomization"]
            row[label] = sub
        out["runs"][run] = row
    return out


def subset_markdown(sub: dict[str, Any]) -> str:
    lines = [
        "| Run | Subset | n | " + " | ".join(SUBSET_METRICS) + " | p (nDCG@10 vs base) |",
        "|---|---|---:|" + "---:|" * len(SUBSET_METRICS) + "---:|",
    ]
    for run, row in sub["runs"].items():
        for label, s in row.items():
            vals = " | ".join("-" if s[m] is None else f"{s[m]:.3f}" for m in SUBSET_METRICS)
            p = s.get("p_nDCG@10_vs_base")
            lines.append(
                f"| {run} | {label.replace('_', ' ')} | {s['n']} | {vals} | "
                f"{'-' if p is None else f'{p:.4f}'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--qrels")
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--runs-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--permutations", type=int, default=10_000)
    ap.add_argument("--force", action="store_true", help="allow overwriting existing test runs")
    args = ap.parse_args(argv)

    cfg = get_config()
    tuned = (cfg.get("bm25") or {}).get("tuned")
    if not tuned:
        print(
            "error: bm25.tuned is not set; run python scripts/tune_bm25.py first", file=sys.stderr
        )
        return 1
    pcfg = cfg.get("phrase") or {}
    if pcfg.get("tuned"):
        phrase_weight = float(pcfg["tuned"]["weight"])
    else:
        phrase_weight = float(pcfg.get("weight", 1.0))
        print(
            f"note: phrase.tuned not set; using phrase.weight = {phrase_weight:g} "
            "(run scripts/tune_phrase.py on dev)"
        )
    queries_path = args.queries or get_path("queries")
    if not queries_path.exists():
        print(f"error: {queries_path} does not exist yet (queries are T08).", file=sys.stderr)
        return 1
    queries = read_queries(queries_path, split=args.split)
    if not queries:
        print(f"error: no {args.split} queries in {queries_path}", file=sys.stderr)
        return 1
    try:
        index, is_sample = open_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    src = (index.source or {}).get("path")
    phones_path = (
        Path(src)
        if src and Path(src).exists()
        else (get_path("sample_dir") / "phones.jsonl" if is_sample else get_path("phones"))
    )
    phones = read_phones(phones_path)

    retrievers = build(index, phones, tuned, phrase_weight)
    runs_dir = args.runs_dir or get_path("runs_dir")
    if args.split == "test" and not args.force:
        existing = [r.name for r in retrievers[1:] if (runs_dir / f"{r.name}_test.run").exists()]
        if existing:
            print(
                f"error: test runs already exist for {existing}; the test split is run once "
                "with dev-chosen params (use --force only if you mean it).",
                file=sys.stderr,
            )
            return 1

    paths, constrained = [], set()
    for r in retrievers:
        name = f"{r.name}_{args.split}"
        paths.append(run_retriever(r, queries, run_name=name, runs_dir=runs_dir))
        print(f"wrote {paths[-1]}")
        if isinstance(r, FilteredRetriever):
            r.write_log(runs_dir / f"{name}.filters.json")
            constrained |= {q for q, e in r.log.items() if e["source"] != "none"}
    sources = retrievers[1].log  # type: ignore[attr-defined]
    n_src = {
        s: sum(1 for e in sources.values() if e["source"] == s)
        for s in ("queries.jsonl", "parser", "none")
    }
    print(f"filter sources: {n_src}")

    try:
        qrels = resolve_qrels_path(args.qrels)
    except FileNotFoundError as e:
        print(f"note: {e} Runs written; evaluation skipped.")
        return 0
    results_dir = args.results_dir or get_path("results_dir")
    results = {}
    meta = {"phrase_weight": phrase_weight, "k1": tuned["k1"], "b": tuned["b"]}
    for p in paths:
        results[p.stem] = evaluate_run(
            p, qrels, queries_path=queries_path, out_dir=results_dir, meta=meta
        )
    base_name = f"{BASE}_{args.split}"
    systems = {n: r["per_query"] for n, r in results.items() if n != base_name}
    comparison = {
        m: compare_many(results[base_name]["per_query"], systems, m, args.permutations)
        for m in ("nDCG@10", "MAP")
    }
    cmp_path = results_dir / f"compare_t06_t07_{args.split}.json"
    cmp_path.write_text(
        json.dumps(
            {"baseline": base_name, "filter_sources": n_src, "comparison": comparison}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    table = results_dir / f"table_t06_t07_{args.split}.md"
    files = [results_dir / f"{n}.json" for n in results]
    print()
    print(write_table(files, list(METRICS), base_name, table, args.permutations))
    for n, row in comparison["nDCG@10"].items():
        print(
            f"{n}: nDCG@10 diff {row['mean_diff']:+.4f}, p={row['p_randomization']:.4f}, "
            f"p_holm={row['p_randomization_holm']:.4f}, +/-/= {row['better']}/{row['worse']}/"
            f"{row['tied']}"
        )

    sub = subset_analysis(results, constrained)
    sub.update({"split": args.split, "filter_sources": n_src})
    sub_json = results_dir / f"filters_subset_{args.split}.json"
    sub_json.write_text(json.dumps(sub, indent=2) + "\n", encoding="utf-8")
    md = subset_markdown(sub)
    sub_json.with_suffix(".md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"wrote {table} (+ .tex), {cmp_path}, {sub_json} (+ .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
