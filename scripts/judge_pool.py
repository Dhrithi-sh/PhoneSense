"""Judge a pool with the LLM and write qrels/llm.txt (T09a round 1 on dev, T09b round 2 on test).

    python scripts/judge_pool.py --split dev                  # runs/pool_dev.tsv -> qrels/llm.txt
    python scripts/judge_pool.py --split test --offline       # replay logged judgments only
    python scripts/judge_pool.py --split dev --limit 20       # try the first 20 pairs
    python scripts/judge_pool.py --split dev --provider groq \\
        --models openai/gpt-oss-120b,qwen/qwen3.8-27b          # split the queries over 2 models

Reads the pool from scripts/make_pool.py, grades every (query, phone) pair 0-3 with
phonesense.llm.judge, and merges the grades into qrels/llm.txt (TREC format, CONTRACT 4.5),
keeping judgments of the other split. Every call is logged to runs/llm_logs/judge.jsonl
(committed) and repeated calls are replayed, so an interrupted run resumes for free.
The judge never sees which system retrieved a phone.

``--models a,b`` gives every query to exactly one of the models (round robin over the sorted
query ids) and alternates calls between them, which suits free plans that cap tokens per model
per day. Progress is saved every ``--save-every`` pairs and when a call fails, so a daily quota
that runs out costs only the unjudged remainder; re-run later and the logged calls are replayed.
Which model judged which query is written to results/judge_models_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_phones, read_qrels, read_queries, write_qrels
from phonesense.llm.client import LLMClient, LLMError
from phonesense.llm.judge import JudgeError, LLMJudge, assign_models, interleave
from phonesense.llm.pooling import read_pool


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--pool", type=Path, help="default: runs/pool_<split>.tsv")
    ap.add_argument("--phones", type=Path, help="default: paths.phones")
    ap.add_argument("--out", type=Path, help="default: paths.qrels_llm")
    ap.add_argument("--provider", help="gemini | openai (default: llm.provider in config.yaml)")
    ap.add_argument("--model", help="default: llm.models.<provider> in config.yaml")
    ap.add_argument("--models", help="comma-separated models; each query goes to one of them")
    ap.add_argument("--save-every", type=int, default=50, help="save qrels every N judged pairs")
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

    model_names = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else []
    if client is not None:
        judges = {"": LLMJudge(client)}
        model_of = {qid: "" for qid in queries}
    elif model_names:
        judges = {m: LLMJudge(LLMClient(args.provider, m, offline=args.offline))
                  for m in model_names}
        model_of = assign_models(list(queries), model_names)
    else:
        only = LLMClient(args.provider, args.model, offline=args.offline)
        judges = {only.model: LLMJudge(only)}
        model_of = {qid: only.model for qid in queries}
    by_model = {m: [pr for pr in pairs if model_of[pr[0]] == m] for m in judges}
    ordered = interleave(list(by_model.values()))

    def save() -> None:
        write_qrels(out, qrels)
        mapping = get_path("results_dir") / f"judge_models_{args.split}.json"
        mapping.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_text(json.dumps(model_of, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    done = 0
    try:
        for qid, doc in ordered:
            j = judges[model_of[qid]].judge(queries[qid], phones[doc])
            qrels.setdefault(qid, {})[doc] = j.grade
            done += 1
            if done % 25 == 0:
                print(f"  judged {done}/{len(ordered)}")
            if args.save_every and done % args.save_every == 0:
                save()
    except (LLMError, JudgeError) as exc:
        save()
        print(f"stopped after {done}/{len(ordered)} pairs: {exc}", file=sys.stderr)
        print(f"saved the {done} grades so far to {out}; the rest are unjudged. Re-run later: "
              "logged calls are replayed, only the missing ones are sent.", file=sys.stderr)
        return 2

    save()
    counts = Counter(g for qid in queries if qid in qrels for g in qrels[qid].values())
    print(f"judged {done} pairs for {len(queries)} {args.split} queries -> {out}")
    print("grade counts: " + ", ".join(f"{g}: {counts.get(g, 0)}" for g in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
