"""Write and evaluate the classic baselines (T04/T05).

    python scripts/run_baselines.py --split dev      # after scripts/tune_bm25.py
    python scripts/run_baselines.py --split test     # ONCE, with the params chosen on dev
    python scripts/run_baselines.py --split all      # e.g. runs for the judging pool (T09)

Runs (``runs/<name>_<split>.run`` via ``run_retriever``):

* ``tfidf_lnc.ltc``  TF-IDF lnc.ltc on ``text``
* ``bm25_default``   BM25, k1 = 1.2, b = 0.75 (``bm25.default``)
* ``bm25_tuned``     BM25 with ``bm25.tuned`` k1/b from config.yaml
* ``bm25f_tuned``    BM25F-lite with the tuned k1/b and field weights

If ``bm25.tuned`` is not set yet, the tuned runs are skipped with a notice. Each run is then
evaluated (results/<run>.json) and a table is printed and written to
results/table_baselines_<split>.md (+ .tex), with TF-IDF as the significance baseline. If no
qrels exist yet, the runs are still written and evaluation is skipped.

For ``--split test`` the script refuses to overwrite existing test runs unless ``--force``:
test is evaluated once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phonesense.config import get_config, get_path
from phonesense.eval.cli import TABLE_METRICS, write_table
from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path
from phonesense.index import StaleIndexError, open_index
from phonesense.io import read_queries
from phonesense.retrieval.base import Retriever, run_retriever
from phonesense.retrieval.bm25 import BM25FRetriever, BM25Retriever
from phonesense.retrieval.tfidf import TfidfRetriever


def build_retrievers(index, cfg: dict) -> tuple[list[Retriever], list[str]]:
    """The baseline retrievers on ``index`` and notes about skipped ones."""
    d = cfg.get("default") or {"k1": 1.2, "b": 0.75}
    out: list[Retriever] = [
        TfidfRetriever(index=index),
        BM25Retriever(d["k1"], d["b"], index=index, name="bm25_default"),
    ]
    tuned = cfg.get("tuned")
    if not tuned:
        return out, [
            "bm25.tuned is not set in config.yaml: skipping bm25_tuned and bm25f_tuned "
            "(run python scripts/tune_bm25.py on dev first)"
        ]
    out.append(BM25Retriever(tuned["k1"], tuned["b"], index=index, name="bm25_tuned"))
    out.append(
        BM25FRetriever(
            tuned["field_weights"], tuned["k1"], tuned["b"], index=index, name="bm25f_tuned"
        )
    )
    return out, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, choices=("dev", "test", "all"))
    ap.add_argument("--sample", action="store_true", help="use the sample index")
    ap.add_argument("--qrels", help="qrels file (default qrels/qrels.txt, else qrels/llm.txt)")
    ap.add_argument("--queries", type=Path, help="queries file (default queries/queries.jsonl)")
    ap.add_argument("--runs-dir", type=Path, help="default runs/")
    ap.add_argument("--results-dir", type=Path, help="default results/")
    ap.add_argument("--force", action="store_true", help="allow overwriting existing test runs")
    args = ap.parse_args(argv)

    queries_path = args.queries or get_path("queries")
    if not queries_path.exists():
        print(f"error: {queries_path} does not exist yet (queries are T08).", file=sys.stderr)
        return 1
    queries = read_queries(queries_path, split=None if args.split == "all" else args.split)
    if not queries:
        print(f"error: no {args.split} queries in {queries_path}", file=sys.stderr)
        return 1
    try:
        index, is_sample = open_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    retrievers, notes = build_retrievers(index, get_config().get("bm25") or {})
    for n in notes:
        print(f"note: {n}")
    runs_dir = args.runs_dir or get_path("runs_dir")
    if args.split == "test" and not args.force:
        existing = [r.name for r in retrievers if (runs_dir / f"{r.name}_test.run").exists()]
        if existing:
            print(
                f"error: test runs already exist for {existing}. The test split is evaluated "
                "once with the dev-chosen params; pass --force only if you really mean to "
                "re-run it.",
                file=sys.stderr,
            )
            return 1

    paths = []
    for r in retrievers:
        p = run_retriever(r, queries, run_name=f"{r.name}_{args.split}", runs_dir=runs_dir)
        print(f"wrote {p}")
        paths.append(p)

    try:
        qrels = resolve_qrels_path(args.qrels)
    except FileNotFoundError as e:
        print(f"note: {e} Runs written; evaluation skipped.")
        return 0
    results_dir = args.results_dir or get_path("results_dir")
    files = []
    for p in paths:
        try:
            evaluate_run(
                p,
                qrels,
                queries_path=queries_path,
                out_dir=results_dir,
                meta={"index": "sample" if is_sample else "full"},
            )
        except ValueError as e:
            print(f"note: not evaluating {p.name}: {e}")
            continue
        files.append(results_dir / f"{p.stem}.json")
    if not files:
        return 0
    out = results_dir / f"table_baselines_{args.split}.md"
    print()
    print(write_table(files, list(TABLE_METRICS), f"tfidf_lnc.ltc_{args.split}", out, 10_000))
    print(f"wrote {out} (+ .tex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
