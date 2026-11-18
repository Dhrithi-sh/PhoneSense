"""Hand-labelling sheets for the LLM-vs-human agreement check (L-B).

    # 1. everyone exports THEIR share of the same sample (fixed seed, disjoint shares)
    python scripts/label_sheet.py export --split test --n 90 --share 2/3 --out sheets/viraja.csv

    # 2. fill the empty `grade` column (0-3) in a spreadsheet; do not look at LLM grades
    # 3. import the filled sheet into qrels/human.txt (existing human grades are kept)
    python scripts/label_sheet.py import --sheet sheets/viraja.csv

The sample is drawn from runs/pool_<split>.tsv (scripts/make_pool.py), spread evenly over the
queries. ``--share i/n`` selects the i-th of n disjoint parts, so three people use 1/3, 2/3, 3/3.
The sheet shows the query and the phone only: no LLM grade and no system names.
Afterwards run scripts/kappa.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phonesense.config import get_path, get_seed
from phonesense.io import read_phones, read_qrels, read_queries, write_qrels
from phonesense.llm.label_sheet import read_sheet, sample_pairs, share_of, write_sheet
from phonesense.llm.pooling import read_pool


def _parse_share(text: str) -> tuple[int, int]:
    try:
        i, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--share must look like 2/3, got {text!r}") from None
    if not 1 <= i <= n:
        raise argparse.ArgumentTypeError(f"--share needs 1 <= i <= n, got {text!r}")
    return i, n


def export(args: argparse.Namespace) -> int:
    pool_path = args.pool or get_path("runs_dir") / f"pool_{args.split}.tsv"
    phones_path = args.phones or get_path("phones")
    for path, hint in ((pool_path, "scripts/make_pool.py"), (phones_path, "scripts/build_data.py")):
        if not path.exists():
            print(f"missing {path}: run {hint} first", file=sys.stderr)
            return 1
    queries = {q.qid: q for q in read_queries(get_path("queries"), args.split)}
    phones = {p.doc_id: p for p in read_phones(phones_path)}
    pool = {q: d for q, d in read_pool(pool_path).items() if q in queries}
    sample = sample_pairs(pool, args.n, args.seed)
    i, n = args.share
    mine = share_of(sample, i, n)
    out = write_sheet(args.out, mine, queries, phones)
    print(f"sample of {len(sample)} pairs (seed {args.seed}); your share {i}/{n}: "
          f"{len(mine)} pairs -> {out}")
    return 0


def import_(args: argparse.Namespace) -> int:
    try:
        new = read_sheet(args.sheet)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out = args.out or get_path("qrels_human")
    current = read_qrels(out) if out.exists() else {}
    added = changed = 0
    for qid, docs in new.items():
        for doc, grade in docs.items():
            old = current.setdefault(qid, {}).get(doc)
            if old is None:
                added += 1
            elif old != grade:
                changed += 1
            current[qid][doc] = grade
    write_qrels(out, current)
    print(f"{out}: {added} new, {changed} changed, {sum(len(d) for d in current.values())} total")
    if changed:
        print("warning: some grades changed; check nobody labelled the same pair twice",
              file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a sheet with your share of the sample")
    ex.add_argument("--split", choices=("dev", "test"), default="test")
    ex.add_argument("--n", type=int, default=90, help="size of the whole sample (all shares)")
    ex.add_argument("--share", type=_parse_share, default=(1, 1), help="i/n, e.g. 2/3")
    ex.add_argument("--seed", type=int, default=get_seed())
    ex.add_argument("--pool", type=Path)
    ex.add_argument("--phones", type=Path)
    ex.add_argument("--out", type=Path, required=True)
    ex.set_defaults(func=export)
    im = sub.add_parser("import", help="merge a filled sheet into qrels/human.txt")
    im.add_argument("--sheet", type=Path, required=True)
    im.add_argument("--out", type=Path, help="default: paths.qrels_human")
    im.set_defaults(func=import_)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
