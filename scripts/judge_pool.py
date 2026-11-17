"""Judge a pool with the LLM and write qrels/llm.txt (T09a round 1 on dev, T09b round 2 on test).

    python scripts/judge_pool.py --split dev                  # runs/pool_dev.tsv -> qrels/llm.txt
    python scripts/judge_pool.py --split test --offline       # replay logged judgments only
    python scripts/judge_pool.py --split dev --limit 20       # try the first 20 pairs

Reads the pool from scripts/make_pool.py, grades every (query, phone) pair 0-3 with
phonesense.llm.judge, and merges the grades into qrels/llm.txt (TREC format, CONTRACT 4.5),
keeping judgments of the other split. Every call is logged to runs/llm_logs/judge.jsonl
(committed) and repeated calls are replayed, so an interrupted run resumes for free.
The judge never sees which system retrieved a phone.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_phones, read_qrels, read_queries, write_qrels
from phonesense.llm.client import LLMClient, LLMError
from phonesense.llm.judge import JudgeError, LLMJudge
from phonesense.llm.pooling import read_pool


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--pool", type=Path, help="default: runs/pool_<split>.tsv")
    ap.add_argument("--phones", type=Path, help="default: paths.phones")
    ap.add_argument("--out", type=Path, help="default: paths.qrels_llm")
    ap.add_argument("--provider", help="gemini | openai (default: llm.provider in config.yaml)")
    ap.add_argument("--model", help="default: llm.models.<provider> in config.yaml")
    ap.add_argument("--offline", action="store_true", help="replay logged answers, no new calls")
    ap.add_argument("--limit", type=int, help="judge only the first N pairs (for a trial)")
    args = ap.parse_args(argv)

    pool_path = args.pool or get_path("runs_dir") / f"pool_{args.split}.tsv"
    phones_path = args.phones or get_path("phones")
    for path, hint in ((pool_path, "scripts/make_pool.py"), (phones_path, "scripts/build_data.py")):
        if not path.exists():
            print(f"missing {path}: run {hint} first", file=sys.stderr)
            return 1

    queries = {q.qid: q for q in read_queries(get_path("queries"), args.split)}
    phones = {p.doc_id: p for p in read_phones(phones_path)}
    pairs = [(qid, doc) for qid, docs in read_pool(pool_path).items() for doc in docs]
    unknown = [(q, d) for q, d in pairs if q not in queries or d not in phones]
    if unknown:
        print(f"pool has {len(unknown)} pairs not in the {args.split} queries/phones, "
              f"e.g. {unknown[0]}", file=sys.stderr)
        return 1
    if args.limit:
        pairs = pairs[: args.limit]

    out = args.out or get_path("qrels_llm")
    qrels = read_qrels(out) if out.exists() else {}
    for qid in [q for q in qrels if q in queries]:  # re-judging this split replaces its grades
        del qrels[qid]

    judge = LLMJudge(client or LLMClient(args.provider, args.model, offline=args.offline))
    done = 0
    try:
        for qid, doc in pairs:
            j = judge.judge(queries[qid], phones[doc])
            qrels.setdefault(qid, {})[doc] = j.grade
            done += 1
            if done % 25 == 0:
                print(f"  judged {done}/{len(pairs)}")
    except (LLMError, JudgeError) as exc:
        print(f"stopped after {done}/{len(pairs)} pairs: {exc}", file=sys.stderr)
        print("grades so far are NOT written; re-run to resume (answers already logged "
              "are replayed).", file=sys.stderr)
        return 2

    write_qrels(out, qrels)
    counts = Counter(g for qid in queries if qid in qrels for g in qrels[qid].values())
    print(f"judged {done} pairs for {len(queries)} {args.split} queries -> {out}")
    print("grade counts: " + ", ".join(f"{g}: {counts.get(g, 0)}" for g in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
