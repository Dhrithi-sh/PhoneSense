"""Compare LLM and human labels (kappa) and build the final qrels (T09b).

    python scripts/kappa.py                       # agreement over all doubly-judged pairs
    python scripts/kappa.py --split test          # only test queries
    python scripts/kappa.py --merge               # also write qrels/qrels.txt

Reads qrels/llm.txt and qrels/human.txt (CONTRACT 4.5), writes results/kappa.json. ``--merge``
writes qrels/qrels.txt in which a human grade replaces the LLM grade wherever one exists.
Read the pair count before trusting a kappa: with fewer than 30 pairs the estimate is noisy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_qrels, read_queries, write_qrels
from phonesense.llm.agreement import agreement_report, merge_qrels

MIN_PAIRS = 30


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all")
    ap.add_argument("--llm", type=Path, help="default: paths.qrels_llm")
    ap.add_argument("--human", type=Path, help="default: paths.qrels_human")
    ap.add_argument("--out", type=Path, help="default: results/kappa.json")
    ap.add_argument("--merge", action="store_true", help="also write the merged qrels/qrels.txt")
    ap.add_argument("--merged-out", type=Path, help="default: paths.qrels")
    args = ap.parse_args(argv)

    llm_path = args.llm or get_path("qrels_llm")
    human_path = args.human or get_path("qrels_human")
    needed = ((llm_path, "scripts/judge_pool.py"), (human_path, "scripts/label_sheet.py import"))
    for path, hint in needed:
        if not path.exists():
            print(f"missing {path}: produce it first ({hint})", file=sys.stderr)
            return 1
    llm, human = read_qrels(llm_path), read_qrels(human_path)

    qids = None
    if args.split != "all":
        qids = [q.qid for q in read_queries(get_path("queries"), args.split)]
    try:
        report = agreement_report(llm, human, qids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report["split"] = args.split

    print(f"{report['n_pairs']} pairs over {report['n_queries']} queries ({args.split})")
    print(f"  exact agreement  {report['exact_agreement']:.3f}")
    print(f"  kappa            {report['kappa']:.3f}")
    print(f"  weighted kappa   {report['weighted_kappa']:.3f}")
    print(f"  kappa (binary)   {report['kappa_binary']:.3f}   (relevant = grade >= 2)")
    print("  confusion (rows LLM 0-3, columns human 0-3):")
    for g, row in enumerate(report["confusion_llm_rows_human_cols"]):
        print(f"    {g}: " + " ".join(f"{c:4d}" for c in row))
    if report["n_pairs"] < MIN_PAIRS:
        print(f"warning: only {report['n_pairs']} pairs (< {MIN_PAIRS}); kappa is noisy",
              file=sys.stderr)

    out = args.out or get_path("results_dir") / "kappa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    if args.merge:
        merged = merge_qrels(llm, human)
        target = args.merged_out or get_path("qrels")
        write_qrels(target, merged)
        replaced = sum(
            1
            for qid, docs in human.items()
            for doc, grade in docs.items()
            if doc in llm.get(qid, {}) and llm[qid][doc] != grade
        )
        print(f"wrote {target}: {sum(len(d) for d in merged.values())} judgments, "
              f"{replaced} LLM grades replaced by human ones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
