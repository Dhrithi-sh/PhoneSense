"""Tune the phrase-boost weight on the DEV queries only (T07).

    python scripts/tune_phrase.py                  # grid phrase.weight_grid, objective from config
    python scripts/tune_phrase.py --no-write-config

Scores ``bm25_tuned + phrase boost`` for every weight in ``phrase.weight_grid``
({0.5, 1, 2, 4}) plus 0 (= plain tuned BM25, for reference), picks the best mean objective
(``bm25.tuning.objective``, default nDCG@10; ties -> the weight closest to 1), and writes
``results/phrase_tuning.json`` and ``phrase.tuned`` in config.yaml (between its marker
comments). Needs ``bm25.tuned`` (run scripts/tune_bm25.py first).

``--split test`` is refused: tuning on test would invalidate the final test results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phonesense.config import get_config, get_path, write_tuned_block
from phonesense.eval.evaluate import evaluate_rankings, resolve_qrels_path
from phonesense.eval.metrics import canonical_metric, metric_names
from phonesense.index import StaleIndexError, open_index
from phonesense.io import read_qrels, read_queries
from phonesense.retrieval.bm25 import BM25Retriever
from phonesense.retrieval.positional import PhraseBoostRetriever, query_phrases

REFUSE_TEST = (
    "error: tune_phrase.py only tunes on the dev queries. Choosing phrase_weight on the {split} "
    "queries would leak test information into the parameter and invalidate our final test "
    "results. Tune on dev (the default), then run `python scripts/run_t06_t07.py --split test` "
    "once."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--objective")
    ap.add_argument("--qrels")
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--no-write-config", action="store_true")
    ap.add_argument("--out", type=Path, help="default results/phrase_tuning.json")
    args = ap.parse_args(argv)
    if args.split != "dev":
        print(REFUSE_TEST.format(split=args.split), file=sys.stderr)
        return 2

    cfg = get_config()
    tuned = (cfg.get("bm25") or {}).get("tuned")
    if not tuned:
        print(
            "error: bm25.tuned is not set; run python scripts/tune_bm25.py first", file=sys.stderr
        )
        return 1
    queries_path = args.queries or get_path("queries")
    if not queries_path.exists():
        print(f"error: {queries_path} does not exist yet (queries are T08).", file=sys.stderr)
        return 1
    queries = read_queries(queries_path, split="dev")
    try:
        qrels = read_qrels(resolve_qrels_path(args.qrels))
        index, is_sample = open_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    pcfg = cfg.get("phrase") or {}
    objective = canonical_metric(
        args.objective or (cfg.get("bm25") or {}).get("tuning", {}).get("objective", "nDCG@10"),
        metric_names(),
    )
    base = BM25Retriever(tuned["k1"], tuned["b"], index=index, name="bm25_tuned")
    with_phrases = [q.qid for q in queries if query_phrases(q.text, pcfg.get("auto_phrases", []))]
    rows = []
    for w in [0.0, *[float(x) for x in pcfg.get("weight_grid", [0.5, 1, 2, 4])]]:
        r = PhraseBoostRetriever(base, w)
        rankings = {q.qid: [d for d, _ in r.search(q, args.k)] for q in queries}
        _, means, _, _ = evaluate_rankings(rankings, qrels, {q.qid for q in queries})
        rows.append({"weight": w, objective: means[objective], "metrics": means})
        print(f"  phrase_weight={w:g}: {objective}={means[objective]:.4f}")
    grid = [r for r in rows if r["weight"] > 0]
    best = min(grid, key=lambda r: (-round(r[objective], 12), abs(r["weight"] - 1.0)))
    print(
        f"best phrase_weight={best['weight']:g} ({objective}={best[objective]:.4f}; "
        f"w=0: {rows[0][objective]:.4f}); dev queries with phrases: {len(with_phrases)}"
    )
    out = args.out or get_path("results_dir") / "phrase_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": "dev",
                "objective": objective,
                "n_queries": len(queries),
                "queries_with_phrases": with_phrases,
                "bm25": {"k1": tuned["k1"], "b": tuned["b"]},
                "best": {"weight": best["weight"], objective: best[objective]},
                "baseline_w0": rows[0][objective],
                "grid": rows,
                "index": "sample" if is_sample else "full",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    if is_sample:
        print("note: tuned on the sample index; not writing config.yaml")
    elif not args.no_write_config:
        path = write_tuned_block(
            "phrase",
            {
                "weight": best["weight"],
                "objective": objective,
                "dev_score": round(best[objective], 4),
                "dev_score_w0": round(rows[0][objective], 4),
                "n_dev_queries_with_phrases": len(with_phrases),
            },
        )
        print(f"wrote phrase.tuned to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
