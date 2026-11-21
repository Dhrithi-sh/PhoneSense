"""Pooling and judgment files for relevance judging.

Pooling (TREC style): for every information need, take the top k articles of
every run for every form of that need (Hindi, Hinglish, messy, English) and
merge them. One judgment per need and article covers all four forms, as
`documentation/formats.md` says, so each article is judged once.

Files, all in `judgments/` at the repo root and committed to git (they hold
only ids and grades, no article text):

- `pool.tsv`: need_id <TAB> doc_id, the articles to judge
- `qrels_<person>.txt`: one per person, in the formats.md judgment format
  `need_id 0 doc_id rel` with rel 0 (not relevant), 1 (partly), 2 (fully).
  One file each means four people judging at once never edit the same file.

Build the pool after the experiment runner has written its run files. By
default new articles are added to the existing pool, so nothing already
judged is dropped; --fresh starts over.

    python -m dhvani.eval.experiments --out data/eval/full
    python -m dhvani.eval.judge --runs data/eval/full/runs

Then judge in the app (the "judge" page in the sidebar).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from dhvani.eval.metrics import read_qrels, read_run

ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS_DIR = ROOT / "judgments"
NEEDS_DIR = ROOT / "documentation" / "needs"
PEOPLE = ("rishit", "riya", "dhrithi", "viraja")
GRADES = {0: "Not relevant", 1: "Partly", 2: "Fully"}


def read_need_descriptions(needs_dir=NEEDS_DIR):
    """{need_id: "what the user wants"} from the tables in the needs files."""
    out = {}
    for path in sorted(Path(needs_dir).glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*([A-Z]\d{2})\s*\|\s*([^|]+?)\s*\|", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def build_pool(runs, queries, k=10):
    """{need_id: [doc_ids]} from {run_name: {qid: [doc_ids]}}, keeping first-seen order."""
    need_of = {qid: need for qid, need, _form, _text in queries}
    pool = defaultdict(list)
    seen = defaultdict(set)
    for name in sorted(runs):
        for qid, docs in runs[name].items():
            need = need_of.get(qid)
            if not need:
                continue
            for doc_id in docs[:k]:
                if doc_id not in seen[need]:
                    seen[need].add(doc_id)
                    pool[need].append(doc_id)
    return dict(pool)


def merge_pools(old, new):
    """Everything in the old pool, then anything new, per need. Nothing judged is ever dropped."""
    merged = {need: list(docs) for need, docs in old.items()}
    for need, docs in new.items():
        have = set(merged.get(need, []))
        merged.setdefault(need, []).extend(d for d in docs if d not in have)
    return merged


def write_pool(pool, path=JUDGMENTS_DIR / "pool.tsv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("need_id\tdoc_id\n")
        for need in sorted(pool):
            for doc_id in pool[need]:
                f.write(f"{need}\t{doc_id}\n")
    return path


def read_pool(path=JUDGMENTS_DIR / "pool.tsv"):
    pool = defaultdict(list)
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                pool[parts[0]].append(parts[1])
    return dict(pool)


def qrels_path(person, judgments_dir=JUDGMENTS_DIR):
    return Path(judgments_dir) / f"qrels_{person}.txt"


def read_person(person, judgments_dir=JUDGMENTS_DIR):
    path = qrels_path(person, judgments_dir)
    return read_qrels(path) if path.exists() else {}


def save_judgment(person, need, doc_id, grade, judgments_dir=JUDGMENTS_DIR):
    """Set one judgment in the person's file (rewriting it keeps one line per pair)."""
    if grade not in GRADES:
        raise ValueError(f"grade must be 0, 1 or 2, not {grade}")
    qrels = read_person(person, judgments_dir)
    qrels.setdefault(need, {})[doc_id] = grade
    path = qrels_path(person, judgments_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n in sorted(qrels):
            for d in sorted(qrels[n]):
                f.write(f"{n} 0 {d} {qrels[n][d]}\n")


def all_judgments(judgments_dir=JUDGMENTS_DIR):
    """Everyone's judgments merged: {need_id: {doc_id: grade}}.

    If two people judged the same pair, the higher grade is kept.
    """
    merged = defaultdict(dict)
    for path in sorted(Path(judgments_dir).glob("qrels_*.txt")):
        for need, docs in read_qrels(path).items():
            for doc_id, grade in docs.items():
                merged[need][doc_id] = max(grade, merged[need].get(doc_id, 0))
    return dict(merged)


def progress(pool, qrels):
    """{need_id: (judged, total)}."""
    return {need: (sum(1 for d in docs if d in qrels.get(need, {})), len(docs)) for need, docs in pool.items()}


def main(argv=None):
    from dhvani.eval.experiments import read_needs

    parser = argparse.ArgumentParser(description="Build the judging pool from run files.")
    parser.add_argument("--runs", default="data/eval/full/runs")
    parser.add_argument("--k", type=int, default=10, help="top k of each run that goes into the pool")
    parser.add_argument("--out", default=JUDGMENTS_DIR / "pool.tsv")
    parser.add_argument("--fresh", action="store_true", help="replace the pool instead of adding to it")
    args = parser.parse_args(argv)

    runs = {p.stem: read_run(p) for p in sorted(Path(args.runs).glob("*.txt"))}
    if not runs:
        print(f"No run files in {args.runs}")
        return
    pool = build_pool(runs, read_needs(), args.k)
    old = {} if args.fresh else read_pool(args.out)
    added = sum(len(set(v) - set(old.get(n, []))) for n, v in pool.items())
    pool = merge_pools(old, pool)
    path = write_pool(pool, args.out)
    total = sum(len(v) for v in pool.values())
    if old:
        print(f"{added} new articles added to the existing pool")
    print(f"{len(runs)} runs, {len(pool)} needs, {total} articles to judge ({total / max(len(pool), 1):.0f} per need) -> {path}")


if __name__ == "__main__":
    main()
