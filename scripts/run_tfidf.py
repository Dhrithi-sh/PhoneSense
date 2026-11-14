"""TF-IDF lnc.ltc runs and quick checks (T04).

    python scripts/run_tfidf.py --split dev              # -> runs/tfidf_lnc.ltc_dev.run
    python scripts/run_tfidf.py --split all --sample     # use the sample index
    python scripts/run_tfidf.py --query "samsung 5g 128 gb"   # print the top 10

Uses the saved index from scripts/build_index.py (data/index/, or data/index/sample/ with
--sample). Without --sample, falls back to the sample index (with a notice) if the full index
has not been built. Prints the mean and p95 query latency of a run and writes it to
results/latency_<run_name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from phonesense.config import get_config, get_path
from phonesense.index import InvertedIndex, StaleIndexError, index_path
from phonesense.io import Ranking, read_phones, read_queries
from phonesense.retrieval.base import Retriever, run_retriever
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Phone, Query


class _Timed(Retriever):
    """Wraps a retriever and records the wall-clock time of every search call."""

    def __init__(self, inner: Retriever) -> None:
        self.inner, self.name, self.latencies = inner, inner.name, []

    def fit(self, phones: list[Phone]) -> None:
        self.inner.fit(phones)

    def search(self, query: Query, k: int = 100) -> Ranking:
        t0 = time.perf_counter()
        out = self.inner.search(query, k)
        self.latencies.append(time.perf_counter() - t0)
        return out


def _load_index(sample: bool) -> tuple[InvertedIndex, bool]:
    path = index_path(sample)
    if (
        not sample
        and not (path / "meta.json").exists()
        and (index_path(True) / "meta.json").exists()
    ):
        print(
            "note: full index not built; using the sample index (run `make index` for the full one)"
        )
        path, sample = index_path(True), True
    if not (path / "meta.json").exists():
        flag = " --sample" if sample else ""
        raise FileNotFoundError(
            f"no index at {path}. Build it first: python scripts/build_index.py{flag}"
        )
    return InvertedIndex.load(path), sample


def _titles(index: InvertedIndex, sample: bool) -> dict[str, str]:
    src = (index.source or {}).get("path")
    path = (
        Path(src)
        if src and Path(src).exists()
        else (get_path("sample_dir") / "phones.jsonl" if sample else get_path("phones"))
    )
    return {p.doc_id: p.title for p in read_phones(path)} if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: don't crash on odd title chars
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--split", choices=("dev", "test", "all"), help="write a run for these queries"
    )
    mode.add_argument("--query", help="free-text query: print the top results")
    ap.add_argument("--sample", action="store_true", help="use the sample index")
    ap.add_argument("--queries", type=Path, help="override queries file (default: paths.queries)")
    ap.add_argument("--runs-dir", type=Path, help="override output dir (default: paths.runs_dir)")
    ap.add_argument("--field", help="indexed field to score (default: retrieval.tfidf_field)")
    ap.add_argument(
        "--k",
        type=int,
        help="results per query (default: 10 for --query, retrieval.default_k for runs)",
    )
    args = ap.parse_args(argv)

    if args.split:
        queries_path = args.queries or get_path("queries")
        if not queries_path.exists():
            print(
                f"error: {queries_path} does not exist yet (queries are task T08). "
                'Use --query "..." for a quick check.',
                file=sys.stderr,
            )
            return 1
    try:
        index, sample = _load_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    retriever = TfidfRetriever(field=args.field, index=index)

    if args.query:
        k = args.k or 10
        titles = _titles(index, sample)
        which = "sample" if sample else "full"
        print(
            f"top {k} for {args.query!r} ({retriever.name}, {which} index, "
            f"terms {list(retriever.query_weights(args.query))}):"
        )
        for rank, (doc_id, score) in enumerate(retriever.score_text(args.query, k), 1):
            print(f"{rank:3d}  {score:.4f}  {doc_id}  {titles.get(doc_id, '')[:90]}")
        return 0

    queries = read_queries(queries_path, split=None if args.split == "all" else args.split)
    k = args.k or int(get_config().get("retrieval", {}).get("default_k", 100))
    timed = _Timed(retriever)
    run_name = f"{retriever.name}_{args.split}"
    path = run_retriever(timed, queries, k=k, run_name=run_name, runs_dir=args.runs_dir)
    ms = np.array(timed.latencies) * 1000
    lat = {
        "run": run_name,
        "index": "sample" if sample else "full",
        "N": index.N,
        "queries": len(ms),
        "k": k,
        "mean_ms": round(float(ms.mean()), 4),
        "p95_ms": round(float(np.percentile(ms, 95)), 4),
        "max_ms": round(float(ms.max()), 4),
    }
    out_dir = args.runs_dir if args.runs_dir else get_path("results_dir")
    lat_path = out_dir / f"latency_{run_name}.json"
    lat_path.parent.mkdir(parents=True, exist_ok=True)
    lat_path.write_text(json.dumps(lat, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(queries)} queries, k={k})")
    print(f"latency: mean {lat['mean_ms']:.3f} ms, p95 {lat['p95_ms']:.3f} ms -> {lat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
