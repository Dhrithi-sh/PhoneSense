"""Evaluation CLI (T11).

    python -m phonesense.eval.cli eval --run runs/X_dev.run [runs/Y_dev.run ...]
        [--qrels Q] [--split dev]
    python -m phonesense.eval.cli compare --baseline runs/A_test.run
        --runs runs/B_test.run runs/C_test.run --metric ndcg@10 [--qrels Q] [--split test]
    python -m phonesense.eval.cli table --results "results/*_dev.json"
        --metrics P@10 MAP nDCG@10 judged@10 [--baseline tfidf_lnc.ltc] [--out results/t.md]
    python -m phonesense.eval.cli all [--qrels Q]       # every run in runs/ (= make eval)
    python -m phonesense.eval.cli agree --a qrels/human.txt --b qrels/llm.txt
    python -m phonesense.eval.cli validate --run runs/X.run --qrels Q
        (the pre-T11 form without "validate" still works)

``--qrels`` defaults to qrels/qrels.txt, falling back to qrels/llm.txt with a warning. ``eval``
writes results/<run_name>.json (CONTRACT.md 4.9). ``table --out x.md`` also writes x.tex.
Globs in ``--results`` are expanded here, so they work in PowerShell too.
Exit codes: 0 = ok, 1 = invalid input or missing files.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from phonesense.eval.metrics import canonical_metric
from phonesense.io import read_qrels, read_run_entries
from phonesense.schemas import SchemaError, run_name_split

TABLE_METRICS = ("P@10", "R@10", "MAP", "nDCG@10", "RR", "judged@10")


# ------------------------------------------------------------------ validate (pre-T11 behaviour)


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        entries = read_run_entries(args.run)
        qrels = read_qrels(args.qrels)
    except (SchemaError, OSError) as e:
        print(f"INVALID INPUT: {e}", file=sys.stderr)
        return 1
    run_qids = {e.qid for e in entries}
    run_name = entries[0].run_name if entries else "(empty)"
    n_judg = sum(len(d) for d in qrels.values())
    print(
        f"run   OK: {args.run} (run_name={run_name}, {len(run_qids)} queries, {len(entries)} lines)"
    )
    print(f"qrels OK: {args.qrels} ({len(qrels)} queries, {n_judg} judgments)")
    missing = sorted(run_qids - qrels.keys())
    if missing:
        print(f"warning: {len(missing)} run queries have no judgments: {', '.join(missing)}")
    unretrieved = sorted(qrels.keys() - run_qids)
    if unretrieved:
        print(
            f"warning: {len(unretrieved)} judged queries absent from run: {', '.join(unretrieved)}"
        )
    print("Inputs are valid.")
    return 0


# ------------------------------------------------------------------ eval / compare / table


def _print_summary(res: dict) -> None:
    m = res["metrics"]
    keys = [k for k in TABLE_METRICS if k in m]
    print(
        f"{res['run']}: "
        + "  ".join(f"{k}={m[k]:.4f}" for k in keys)
        + f"  (n={res['meta']['n_queries']}, excluded={len(res['meta']['excluded_qids'])})"
    )


def cmd_eval(args: argparse.Namespace) -> int:
    from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path

    qrels = resolve_qrels_path(args.qrels)
    for run in args.run:
        res = evaluate_run(run, qrels, args.split, queries_path=args.queries, out_dir=args.out_dir)
        _print_summary(res)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path
    from phonesense.eval.significance import compare_many

    qrels = resolve_qrels_path(args.qrels)

    def per_query(path: str) -> tuple[str, dict]:
        r = evaluate_run(path, qrels, args.split, queries_path=args.queries, write=False)
        return r["run"], r["per_query"]

    base_name, base = per_query(args.baseline)
    metric = canonical_metric(args.metric, list(next(iter(base.values()))))
    systems = dict(per_query(p) for p in args.runs)
    rows = compare_many(base, systems, metric, args.permutations)
    print(f"baseline {base_name}, metric {metric}, Holm-corrected over {len(rows)} system(s)")
    print(
        f"{'system':40s} {'mean':>7s} {'diff':>8s} {'p_rand':>8s} {'p_holm':>8s} "
        f"{'p_t':>8s} {'+/-/=':>9s}"
    )
    for name, r in rows.items():
        print(
            f"{name:40s} {r['mean_system']:7.4f} {r['mean_diff']:+8.4f} "
            f"{r['p_randomization']:8.4f} {r['p_randomization_holm']:8.4f} {r['p_ttest']:8.4f} "
            f"{r['better']:>3d}/{r['worse']}/{r['tied']}"
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"baseline": base_name, "metric": metric, "systems": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(Path(h) for h in (hits or [p]))
    return out


def _results_files(paths: list[Path]) -> list[Path]:
    """Keep only files that are valid results JSON (skips e.g. index_stats.json)."""
    from phonesense.io import read_results

    keep = []
    for p in paths:
        try:
            read_results(p)
            keep.append(p)
        except (SchemaError, OSError, KeyError, TypeError):
            print(f"skipping {p}: not a results file", file=sys.stderr)
    return keep


def write_table(
    files: list[Path], metrics: list[str], baseline: str | None, out: Path | None, permutations: int
) -> str:
    from phonesense.eval.tables import results_table

    t = results_table(files, metrics, baseline, permutations=permutations)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(t.markdown, encoding="utf-8")
        out.with_suffix(".tex").write_text(t.latex, encoding="utf-8")
    return t.markdown


def cmd_table(args: argparse.Namespace) -> int:
    files = _results_files(_expand(args.results))
    if not files:
        print("error: no results files matched", file=sys.stderr)
        return 1
    print(write_table(files, args.metrics, args.baseline, args.out, args.permutations))
    if args.out:
        print(f"wrote {args.out} and {args.out.with_suffix('.tex')}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Evaluate every run in runs/ and write one table per split."""
    from phonesense.config import get_path
    from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path

    qrels = resolve_qrels_path(args.qrels)
    runs = sorted(get_path("runs_dir").glob("*.run"))
    if not runs:
        print(
            f"no runs in {get_path('runs_dir')}; produce some first (e.g. scripts/run_baselines.py)"
        )
        return 0
    by_split: dict[str, list[Path]] = {}
    for run in runs:
        try:
            res = evaluate_run(run, qrels)
        except (ValueError, FileNotFoundError, SchemaError) as e:
            print(f"skipping {run.name}: {e}", file=sys.stderr)
            continue
        _print_summary(res)
        out = get_path("results_dir") / f"{res['run']}.json"
        by_split.setdefault(run_name_split(res["run"]), []).append(out)
    for split, files in by_split.items():
        names = [f.stem for f in files]
        base = next((n for n in names if n.startswith("tfidf_lnc.ltc_")), None)
        out = get_path("results_dir") / f"table_{split}.md"
        print(f"\n## {split}\n")
        print(write_table(files, list(TABLE_METRICS), base, out, args.permutations))
        print(f"wrote {out}")
    return 0


def cmd_agree(args: argparse.Namespace) -> int:
    from phonesense.eval.agreement import agreement_report

    rep = agreement_report(args.a, args.b, args.threshold)
    print(json.dumps(rep, indent=2))
    return 0


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m phonesense.eval.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a run and qrels file")
    v.add_argument("--run", required=True)
    v.add_argument("--qrels", required=True)
    v.set_defaults(func=cmd_validate)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--qrels", help="qrels file (default qrels/qrels.txt, else qrels/llm.txt)")
        p.add_argument(
            "--split",
            choices=("dev", "test", "all"),
            help="must match the run names; default: from the run name",
        )
        p.add_argument("--queries", help="queries file (default queries/queries.jsonl)")

    e = sub.add_parser("eval", help="evaluate runs -> results/<run>.json")
    e.add_argument("--run", nargs="+", required=True)
    e.add_argument("--out-dir", help="results directory (default results/)")
    common(e)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("compare", help="significance tests vs. a baseline")
    c.add_argument("--baseline", required=True)
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--metric", default="nDCG@10")
    c.add_argument("--permutations", type=int, default=10_000)
    c.add_argument("--out", help="also write the comparison as JSON")
    common(c)
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("table", help="Markdown + LaTeX table from results JSON files")
    t.add_argument("--results", nargs="+", required=True)
    t.add_argument("--metrics", nargs="+", default=list(TABLE_METRICS))
    t.add_argument("--baseline")
    t.add_argument("--out", type=Path)
    t.add_argument("--permutations", type=int, default=10_000)
    t.set_defaults(func=cmd_table)

    a = sub.add_parser("all", help="evaluate every run in runs/ (= make eval)")
    a.add_argument("--qrels")
    a.add_argument("--permutations", type=int, default=10_000)
    a.set_defaults(func=cmd_all)

    g = sub.add_parser("agree", help="agreement (kappa) between two qrels files")
    g.add_argument("--a", required=True)
    g.add_argument("--b", required=True)
    g.add_argument("--threshold", type=int, default=2)
    g.set_defaults(func=cmd_agree)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] not in ("-h", "--help"):
        argv = ["validate", *argv]  # pre-T11 form: --run X --qrels Y
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, SchemaError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
