"""TREC Run Pooling Tool (Format 5 in formats.md).

Pools the top-k document candidates across multiple TREC run files to create
unbiased judgment pools for manual relevance annotation.

By default, aggregates multiple surface forms of an information need (e.g.
R01_hi, R01_hinglish, R01_messy, R01_en) under their canonical need_id (R01),
because downstream qrels and evaluation functions operate on need_id.

Preserves the --raw-qid option for cases where per-surface-form pooling
is explicitly requested.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set


def load_queries_mapping(queries_path: str) -> Dict[str, str]:
    """Load qid -> need_id mapping from a queries.tsv file.

    Expected TSV format with header (e.g. qid, need_id, form, text).
    """
    mapping: Dict[str, str] = {}
    p = Path(queries_path)
    if not p.exists():
        return mapping

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] != "qid":
                mapping[parts[0]] = parts[1]
    return mapping


def extract_need_id(qid: str, qid_to_need: Optional[Dict[str, str]] = None) -> str:
    """Map query ID to its information need ID.

    1. If a queries mapping is provided and contains qid, that takes precedence.
    2. Otherwise, if qid contains an underscore (e.g. <need_id>_<form>), splits on the
       first underscore to extract <need_id> (e.g. 'R01_hi' -> 'R01').
    3. If no underscore exists (e.g. 'N01'), preserves qid as-is.
    """
    if qid_to_need and qid in qid_to_need:
        return qid_to_need[qid]
    if "_" in qid:
        return qid.split("_")[0]
    return qid


def pool_runs(
    run_files: List[str],
    top_k: int = 10,
    per_need: bool = True,
    queries_file: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """Pool top-k document IDs per need (or query ID) across multiple TREC run files.

    Args:
        run_files: List of paths to TREC run files.
        top_k: Top-k rank cutoff for candidate pooling (default: 10).
        per_need: If True (default), aggregate all surface forms under need_id.
                  If False, pool by literal qid.
        queries_file: Optional path to queries.tsv for explicit qid -> need_id mapping.

    Returns:
        Dictionary mapping pool key (need_id or qid) to set of unique doc_ids.
    """
    qid_to_need = load_queries_mapping(queries_file) if queries_file else None

    pools: Dict[str, Set[str]] = {}
    for run_path in run_files:
        with open(run_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, doc_id, rank, score, _ = parts[:6]
                    if int(rank) <= top_k:
                        pool_key = extract_need_id(qid, qid_to_need) if per_need else qid
                        pools.setdefault(pool_key, set()).add(doc_id)
    return pools


def main():
    parser = argparse.ArgumentParser(
        description="Pool TREC run files for relevance judgment (Format 5 in formats.md)"
    )
    parser.add_argument("--runs", nargs="+", required=True, help="List of TREC run files")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for pooling (default: 10)")
    parser.add_argument("--out", type=str, default="data/judgments_pool.txt", help="Output judgment file")
    parser.add_argument(
        "--queries",
        type=str,
        default="",
        help="Optional queries.tsv file for explicit qid -> need_id mapping",
    )
    parser.add_argument(
        "--raw-qid",
        action="store_true",
        help="Pool by raw qid surface form instead of aggregating per need",
    )
    args = parser.parse_args()

    pools = pool_runs(
        args.runs,
        top_k=args.top_k,
        per_need=not args.raw_qid,
        queries_file=args.queries if args.queries else None,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pool_key in sorted(pools.keys()):
            for doc_id in sorted(pools[pool_key]):
                # Standard judgment format: need_id 0 doc_id 0 (formats.md Format 5)
                f.write(f"{pool_key} 0 {doc_id} 0\n")
                total_pairs += 1

    unit_name = "raw query forms" if args.raw_qid else "information needs"
    print(f"Pooled {total_pairs} candidate pairs across {len(pools)} {unit_name} into {args.out}")


if __name__ == "__main__":
    main()


