"""T22 ablation: one table over every system of the project, plus figures and failure cases.

    python scripts/ablation.py          # = make ablation

Reads the row definitions from ``ablation.yaml`` and the run files in ``runs/`` (ours and the
teammates'), evaluates every run with the T11 harness (``evaluate_run``) and writes:

* ``results/ablation_test.{md,tex}``  main table (test split): P@10, R@10, MAP, nDCG@10, MRR,
  judged@10. † = significantly different from ``bm25_tuned`` (paired randomization test via
  ``significance.compare_many``, Holm-corrected across all rows, p < 0.05). ‡ = judged@10 <
  0.7: the run was poorly covered by the judging pool, so its scores are lower bounds. Best
  value per column in bold. Rows without a run (or without qrels) are kept as "pending".
* ``results/ablation_dev.{md,tex}``   same for dev (appendix).
* ``results/ablation_incremental.{md,tex}``  each row vs. its parent (ablation.yaml
  ``parent``): delta nDCG@10, p-value, wins/losses/ties per query.
* ``results/ablation.json``  every number above (used by scripts/report_numbers.py).
* ``report/figures/ablation_bars``, ``pr_key_systems``, ``delta_best_vs_bm25`` (PNG + PDF).
* ``results/failure_cases.md``  test queries where the best system does worst, and where it
  loses most against bm25_tuned (with a ``Why:`` placeholder per query).

"Best overall" is chosen on **dev only** (best dev nDCG@10) and then reported on test; the
test split is never used to choose anything (asserted in :func:`choose_best`).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phonesense.config import SEED, get_path, repo_root
from phonesense.eval.evaluate import evaluate_run, resolve_qrels_path
from phonesense.eval.significance import compare, compare_many
from phonesense.io import read_phones, read_qrels, read_queries, read_run

SPLITS = ("test", "dev")
LABELS = {"RR": "MRR"}
DAGGER, DDAGGER = "†", "‡"


# ------------------------------------------------------------------ run resolution


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    for row in spec["rows"]:
        for key in ("label", "run", "owner", "group"):
            if key not in row:
                raise ValueError(f"{path}: row {row} lacks {key!r}")
    return spec


def available_runs(runs_dir: Path) -> dict[str, set[str]]:
    """``{split: {base name}}`` for every ``runs/<base>_<split>.run``."""
    out: dict[str, set[str]] = {s: set() for s in SPLITS}
    for p in runs_dir.glob("*.run"):
        m = re.match(r"^(.*)_(dev|test)$", p.stem)
        if m:
            out[m.group(2)].add(m.group(1))
    return out


def _patterns(x: Any) -> list[str]:
    return [x] if isinstance(x, str) else list(x or [])


def match_bases(patterns: Any, avail: dict[str, set[str]], exclude: Any = ()) -> list[str]:
    """Bases matching the first pattern that matches anything (both splits preferred)."""
    every = avail["dev"] | avail["test"]
    for pat in _patterns(patterns):
        hits = sorted(
            b
            for b in every
            if fnmatch.fnmatchcase(b, pat)
            and not any(fnmatch.fnmatchcase(b, e) for e in _patterns(exclude))
        )
        if hits:
            both = [b for b in hits if b in avail["dev"] and b in avail["test"]]
            return both or hits
    return []


def match_bases_all(patterns: Any, avail: dict[str, set[str]], exclude: Any = ()) -> list[str]:
    """Every base matching the first pattern that matches anything, in either split."""
    every = avail["dev"] | avail["test"]
    for pat in _patterns(patterns):
        hits = sorted(
            b
            for b in every
            if fnmatch.fnmatchcase(b, pat)
            and not any(fnmatch.fnmatchcase(b, e) for e in _patterns(exclude))
        )
        if hits:
            return hits
    return []


class AmbiguousRunError(ValueError):
    """A row's ``run`` matches more than one run file in a split."""


def resolve_rows(spec: dict, avail: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Attach the run base name to each row.

    No match -> ``base = None`` (the row is pending). More than one matching file in the same
    split -> :class:`AmbiguousRunError` listing the candidates (pin the exact name in
    ablation.yaml); the table must never depend on sort order.
    """
    rows = []
    for row in spec["rows"]:
        hits = match_bases(row["run"], avail, row.get("exclude"))
        every = match_bases_all(row["run"], avail, row.get("exclude"))
        if len(every) > 1:
            files = ", ".join(f"{h}_{s}.run" for s in SPLITS for h in every if h in avail[s])
            raise AmbiguousRunError(
                f"row {row['label']!r} (run: {row['run']!r}) matches {len(every)} runs: "
                f"{files}. Pin the exact run name in ablation.yaml."
            )
        rows.append({**row, "base": hits[0] if hits else None})
    return rows


# ------------------------------------------------------------------ evaluation & selection


def evaluate_all(
    bases: list[str],
    split: str,
    qrels: Path,
    queries: Path | None,
    runs_dir: Path,
    results_dir: Path,
) -> dict[str, dict]:
    out = {}
    for base in bases:
        path = runs_dir / f"{base}_{split}.run"
        if not path.exists():
            continue
        try:
            out[base] = evaluate_run(path, qrels, queries_path=queries, out_dir=results_dir)
        except ValueError as e:
            print(f"note: {path.name} not evaluated: {e}", file=sys.stderr)
    return out


def choose_best(dev_results: dict[str, dict], candidates: list[str], metric: str) -> str | None:
    """Best candidate by DEV ``metric``. Refuses anything that isn't a dev result."""
    for res in dev_results.values():
        assert res["split"] == "dev", f"best-system selection must use dev only, got {res['run']}"
    scored = [(dev_results[b]["metrics"][metric], b) for b in candidates if b in dev_results]
    if not scored:
        return None
    return min(scored, key=lambda x: (-x[0], x[1]))[1]  # ties: alphabetical


# ------------------------------------------------------------------ tables


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _tex(s: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        s = s.replace(a, b)
    return s


def build_table(
    rows: list[dict],
    results: dict[str, dict],
    spec: dict,
    split: str,
    best: str | None,
    qrels_ok: bool,
    permutations: int,
) -> dict[str, Any]:
    """Numbers + markers for one split; also renders Markdown and LaTeX."""
    metrics, ref = spec["metrics"], spec["reference"]
    thr = float(spec.get("judged_threshold", 0.7))
    present = [r["base"] for r in rows if r["base"] in results]
    sig: dict[str, dict] = {}
    marked: set[tuple[str, str]] = set()
    if ref in results:
        others = {b: results[b]["per_query"] for b in dict.fromkeys(present) if b != ref}
        for m in metrics:
            if m.startswith("judged@") or not others:
                continue
            sig[m] = compare_many(results[ref]["per_query"], others, m, permutations)
            marked |= {(b, m) for b, r in sig[m].items() if r["p_randomization_holm"] < 0.05}
    best_val = {m: max((results[b]["metrics"][m] for b in present), default=None) for m in metrics}

    table_rows = list(rows)
    best_row = next((r for r in rows if r["base"] == best), None) if best else None
    table_rows.append(
        {
            "label": f"Best overall: {best_row['label']}" if best_row else "Best overall",
            "owner": best_row["owner"] if best_row else "-",
            "group": "Best (chosen on dev)",
            "base": best,
        }
    )

    out_rows = []
    for r in table_rows:
        base = r["base"]
        res = results.get(base) if base else None
        entry = {"label": r["label"], "owner": r["owner"], "group": r["group"], "run": base}
        if res is None:
            entry["status"] = (
                "pending: no qrels"
                if not qrels_ok and base
                else f"pending: no {split} run"
                if base
                else "pending: no run"
            )
        else:
            entry["status"] = "ok"
            entry["metrics"] = {m: res["metrics"][m] for m in metrics}
            entry["n_queries"] = res["meta"]["n_queries"]
            entry["low_coverage"] = res["metrics"].get("judged@10", 1.0) < thr
            entry["significant"] = sorted(m for m in metrics if (base, m) in marked)
        out_rows.append(entry)

    def cells(e: dict, latex: bool) -> list[str]:
        if e["status"] != "ok":
            txt = f"{e['status']} ({e['owner']})"
            return (
                [rf"\multicolumn{{{len(metrics)}}}{{c}}{{\emph{{{_tex(txt)}}}}}"]
                if latex
                else [txt] + ["-"] * (len(metrics) - 1)
            )
        out = []
        for m in metrics:
            v = _fmt(e["metrics"][m])
            if best_val[m] is not None and abs(e["metrics"][m] - best_val[m]) < 1e-12:
                v = rf"\textbf{{{v}}}" if latex else f"**{v}**"
            if m in e["significant"]:
                v += r"$^\dagger$" if latex else DAGGER
            out.append(v)
        return out

    def label(e: dict, latex: bool) -> str:
        lab = _tex(e["label"]) if latex else e["label"]
        if e.get("low_coverage"):
            lab += r"$^\ddagger$" if latex else DDAGGER
        return lab

    heads = [LABELS.get(m, m) for m in metrics]
    n = next((e["n_queries"] for e in out_rows if e["status"] == "ok"), None)
    nq = f"{n} queries" if n else "pending"
    caption = (
        f"{split.capitalize()} split ({nq}). {DAGGER}: p < 0.05 vs. BM25 tuned "
        "(paired randomization test, 10,000 permutations, Holm-corrected across all "
        f"rows). {DDAGGER}: judged@10 < {thr:g}, i.e. the run was poorly covered by the "
        "judging pool and its scores are lower bounds (unjudged documents count as "
        "non-relevant). Best value per column in bold. Best overall is chosen by dev "
        "nDCG@10."
    )
    md = [
        "| System | Owner | Group | " + " | ".join(heads) + " |",
        "|---|---|---|" + "---:|" * len(metrics),
    ]
    for e in out_rows:
        md.append(
            f"| {label(e, False)} | {e['owner']} | {e['group']} | "
            + " | ".join(cells(e, False))
            + " |"
        )
    md += ["", caption]
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll" + "r" * len(metrics) + "}",
        r"\toprule",
        "System & Owner & Group & " + " & ".join(_tex(h) for h in heads) + r" \\",
        r"\midrule",
    ]
    group = None
    for e in out_rows:
        if group is not None and e["group"] != group:
            tex.append(r"\midrule")
        group = e["group"]
        tex.append(
            f"{label(e, True)} & {e['owner']} & {_tex(e['group'])} & "
            + " & ".join(cells(e, True))
            + r" \\"
        )
    cap_tex = _tex(caption).replace(DAGGER, r"$^\dagger$").replace(DDAGGER, r"$^\ddagger$")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{cap_tex}}}",
        rf"\label{{tab:ablation-{split}}}",
        r"\end{table*}",
    ]
    return {
        "rows": out_rows,
        "significance": sig,
        "markdown": "\n".join(md) + "\n",
        "latex": "\n".join(tex) + "\n",
    }


def incremental(
    rows: list[dict],
    results: dict[str, dict],
    dev: dict[str, dict],
    avail: dict[str, set[str]],
    metric: str,
) -> list[dict[str, Any]]:
    """Each row with a parent vs. that parent (multiple parents: best on dev)."""
    out = []
    for r in rows:
        if "parent" not in r:
            continue
        cands = []
        for pat in _patterns(r["parent"]):
            cands += match_bases(pat, avail)
        parent = choose_best(dev, cands, metric) if len(cands) > 1 else (cands or [None])[0]
        e = {"label": r["label"], "owner": r["owner"], "run": r["base"], "parent": parent}
        if r["base"] in results and parent in results:
            c = compare(results[parent]["per_query"], results[r["base"]]["per_query"], metric)
            e.update(
                status="ok",
                delta=c["mean_diff"],
                p=c["p_randomization"],
                wins=c["better"],
                losses=c["worse"],
                ties=c["tied"],
                n=c["n"],
            )
        else:
            e["status"] = "pending"
        out.append(e)
    return out


def incremental_tables(inc: dict[str, list[dict]]) -> tuple[str, str]:
    md, tex = [], []
    for split, rows in inc.items():
        md += [
            f"### {split.capitalize()} split",
            "",
            "| System | Parent | Δ nDCG@10 | p | W / L / T |",
            "|---|---|---:|---:|---:|",
        ]
        tex += [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"System & Parent & $\Delta$nDCG@10 & $p$ & W/L/T \\",
            r"\midrule",
        ]
        for e in rows:
            par = e["parent"] or "-"
            if e["status"] == "ok":
                md.append(
                    f"| {e['label']} | {par} | {e['delta']:+.3f} | {e['p']:.4f} | "
                    f"{e['wins']} / {e['losses']} / {e['ties']} |"
                )
                tex.append(
                    f"{_tex(e['label'])} & {_tex(par)} & {e['delta']:+.3f} & "
                    f"{e['p']:.4f} & {e['wins']}/{e['losses']}/{e['ties']} \\\\"
                )
            else:
                md.append(f"| {e['label']} | {par} | pending ({e['owner']}) | - | - |")
                tex.append(
                    f"{_tex(e['label'])} & {_tex(par)} & "
                    rf"\multicolumn{{3}}{{c}}{{\emph{{pending ({e['owner']})}}}} \\"
                )
        md += [
            "",
            "p: paired randomization test (two-sided, 10,000 permutations, "
            "uncorrected). W/L/T: queries where the system is better / worse / tied.",
            "",
        ]
        tex += [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Incremental comparisons, {split} split: each system vs.\ the "
            r"system it builds on (paired randomization test, uncorrected).}",
            rf"\label{{tab:ablation-incremental-{split}}}",
            r"\end{table}",
            "",
        ]
    return "\n".join(md), "\n".join(tex)


# ------------------------------------------------------------------ figures


def bootstrap_ci(values: list[float], n: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean over queries."""
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def make_figures(
    table: dict,
    results: dict[str, dict],
    rows: list[dict],
    spec: dict,
    key: list[tuple[str, str]],
    best: str | None,
    qrels: dict,
    runs_dir: Path,
    fig_dir: Path | None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    from phonesense.eval.plots import STYLE, pr_curve, save

    ok = [e for e in table["rows"][:-1] if e["status"] == "ok"]
    if not ok:
        return []
    rng = np.random.default_rng(SEED)
    paths: list[Path] = []
    with plt.rc_context({**STYLE, "figure.figsize": (max(6.0, 0.55 * len(ok) + 2), 4.0)}):
        fig, ax = plt.subplots()
        x = np.arange(len(ok))
        for i, (m, shade, hatch) in enumerate((("nDCG@10", "0.25", ""), ("MAP", "0.65", "//"))):
            vals = [e["metrics"][m] for e in ok]
            cis = [
                bootstrap_ci(
                    [q[m] for q in results[e["run"]]["per_query"].values()],
                    int(spec.get("bootstrap_samples", 1000)),
                    rng,
                )
                for e in ok
            ]
            err = np.array(
                [
                    [v - lo for v, (lo, _) in zip(vals, cis, strict=True)],
                    [hi - v for v, (_, hi) in zip(vals, cis, strict=True)],
                ]
            )
            ax.bar(
                x + (i - 0.5) * 0.38,
                vals,
                0.38,
                yerr=err,
                capsize=2,
                color=shade,
                edgecolor="black",
                hatch=hatch,
                linewidth=0.6,
                label=m,
            )
        ax.set_xticks(x, [e["label"] for e in ok], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("score (test)")
        ax.set_ylim(0, 1)
        ax.set_title("Ablation on test, with 95% bootstrap CIs over queries")
        ax.legend()
        paths += save(fig, "ablation_bars", fig_dir)

    runs = {}
    for lab, base in key:
        if base in results:
            runs[lab] = {
                q: [d for d, _ in v] for q, v in read_run(runs_dir / f"{base}_test.run").items()
            }
    if runs:
        qids = list(next(iter(results.values()))["per_query"])
        paths += pr_curve(
            runs,
            qrels,
            "pr_key_systems",
            qids=qids,
            title="11-point interpolated P-R, key systems (test)",
            out_dir=fig_dir,
        )

    ref = spec["reference"]
    if best in results and ref in results and best != ref:
        a, b = results[ref]["per_query"], results[best]["per_query"]
        qs = sorted(set(a) & set(b), key=lambda q: b[q]["nDCG@10"] - a[q]["nDCG@10"])
        d = [b[q]["nDCG@10"] - a[q]["nDCG@10"] for q in qs]
        with plt.rc_context({**STYLE, "figure.figsize": (max(5.5, 0.3 * len(qs) + 1.5), 3.6)}):
            fig, ax = plt.subplots()
            ax.bar(
                range(len(qs)),
                d,
                color=["0.2" if v >= 0 else "0.7" for v in d],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(qs)), qs, rotation=90, fontsize=7)
            ax.set_ylabel("Δ nDCG@10")
            ax.set_title(f"Per-query Δ nDCG@10: {best} vs. {ref} (test)")
            paths += save(fig, "delta_best_vs_bm25", fig_dir)
    return paths


# ------------------------------------------------------------------ failure cases


def failure_cases(
    best: str | None,
    results: dict[str, dict],
    ref: str,
    queries_path: Path,
    qrels: dict,
    runs_dir: Path,
    phones_path: Path,
) -> str:
    head = ["# Failure cases (test split)", ""]
    if best is None or best not in results:
        return "\n".join(head + ["Pending: needs qrels and a test run of the best dev system.", ""])
    pq = results[best]["per_query"]
    worst = sorted(pq, key=lambda q: (pq[q]["nDCG@10"], q))[:3]
    losses = []
    if ref in results and ref != best:
        rp = results[ref]["per_query"]
        losses = [
            q
            for q in sorted(pq, key=lambda q: (pq[q]["nDCG@10"] - rp[q]["nDCG@10"], q))
            if q not in worst and q in rp and pq[q]["nDCG@10"] < rp[q]["nDCG@10"]
        ][:2]
    queries = {q.qid: q for q in read_queries(queries_path)}
    phones = {p.doc_id: p for p in read_phones(phones_path)} if phones_path.exists() else {}
    run = read_run(runs_dir / f"{best}_test.run")
    log_path = runs_dir / f"{best}_test.filters.json"
    flog = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}

    def specs(d: str) -> str:
        p = phones.get(d)
        if p is None:
            return ""
        s = p.specs
        parts = [
            f"${p.price:g}" if p.price else None,
            f"{s.ram_gb:g}GB RAM" if s.ram_gb else None,
            f"{s.storage_gb:g}GB" if s.storage_gb else None,
            f"{s.battery_mah}mAh" if s.battery_mah else None,
            f'{s.screen_in:g}"' if s.screen_in else None,
            "5G" if s.has_5g else None,
            s.os,
        ]
        return ", ".join(x for x in parts if x)

    out = head + [f"System: `{best}` (best on dev nDCG@10). Reference: `{ref}`.", ""]
    for kind, qids in (("Lowest nDCG@10", worst), ("Largest loss vs. bm25_tuned", losses)):
        for qid in qids:
            q = queries.get(qid)
            if flog:
                filt = flog.get(qid, {}).get("filter")
                src = flog.get(qid, {}).get("source", "none")
            else:
                filt, src = None, "this run applies no filter"
            ranked = [d for d, _ in run.get(qid, [])]
            judg = qrels.get(qid, {})
            delta = ""
            if ref in results and qid in results[ref]["per_query"]:
                delta = f", {ref} nDCG@10 = {results[ref]['per_query'][qid]['nDCG@10']:.3f}"
            out += [
                f"## {qid} ({kind}): {q.text if q else '?'}",
                "",
                f"nDCG@10 = {pq[qid]['nDCG@10']:.3f}{delta}. Filters ({src}): "
                f"`{json.dumps(filt) if filt else 'none'}`",
                "",
                "| Rank | doc_id | Title | Key specs | Grade |",
                "|---:|---|---|---|---|",
            ]
            for i, d in enumerate(ranked[:5], 1):
                title = phones[d].title[:70].replace("|", "/") if d in phones else ""
                out.append(f"| {i} | {d} | {title} | {specs(d)} | {judg.get(d, 'unjudged')} |")
            missed = [
                (d, g)
                for d, g in sorted(judg.items(), key=lambda x: (-x[1], x[0]))
                if g >= 2 and d not in ranked[:10]
            ]
            out += ["", "Missed relevant docs (grade >= 2, not in the top 10):", ""]
            for d, g in missed[:10]:
                where = f"rank {ranked.index(d) + 1}" if d in ranked else "not retrieved"
                title = phones[d].title[:70] if d in phones else ""
                out.append(f"- {d} (grade {g}, {where}): {title}")
            if not missed:
                out.append("- none")
            out += ["", "Why: ", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", type=Path, default=repo_root() / "ablation.yaml")
    ap.add_argument("--runs-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--figures-dir", type=Path)
    ap.add_argument("--qrels")
    ap.add_argument("--queries", type=Path)
    ap.add_argument("--phones", type=Path)
    ap.add_argument("--permutations", type=int, default=10_000)
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    spec = load_spec(args.spec)
    runs_dir = args.runs_dir or get_path("runs_dir")
    results_dir = args.results_dir or get_path("results_dir")
    queries_path = args.queries or get_path("queries")
    avail = available_runs(runs_dir)
    try:
        rows = resolve_rows(spec, avail)
    except AmbiguousRunError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for r in rows:
        print(f"  {r['label']:24s} -> {r['base'] or 'PENDING (no matching run)'}")
    bases = list(dict.fromkeys(r["base"] for r in rows if r["base"]))
    try:
        qrels_path = resolve_qrels_path(args.qrels)
        qrels_ok = queries_path.exists()
        if not qrels_ok:
            print(f"note: {queries_path} missing; every row stays pending", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"note: {e} Every row stays pending.", file=sys.stderr)
        qrels_path, qrels_ok = None, False

    results = {s: {} for s in SPLITS}
    if qrels_ok:
        for s in SPLITS:
            results[s] = evaluate_all(bases, s, qrels_path, queries_path, runs_dir, results_dir)
    metric = spec.get("selection_metric", "nDCG@10")
    best = choose_best(results["dev"], bases, metric)

    tables = {
        s: build_table(rows, results[s], spec, s, best, qrels_ok, args.permutations) for s in SPLITS
    }
    inc = {s: incremental(rows, results[s], results["dev"], avail, metric) for s in SPLITS}
    results_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (results_dir / f"ablation_{s}.md").write_text(tables[s]["markdown"], encoding="utf-8")
        (results_dir / f"ablation_{s}.tex").write_text(tables[s]["latex"], encoding="utf-8")
    inc_md, inc_tex = incremental_tables(inc)
    (results_dir / "ablation_incremental.md").write_text(inc_md, encoding="utf-8")
    (results_dir / "ablation_incremental.tex").write_text(inc_tex, encoding="utf-8")

    def best_in(groups: tuple[str, ...]) -> str | None:
        return choose_best(
            results["dev"], [r["base"] for r in rows if r["group"] in groups and r["base"]], metric
        )

    key = [
        ("TF-IDF", "tfidf_lnc.ltc"),
        ("BM25 tuned", spec["reference"]),
        ("best structured", best_in(("+Structure",))),
        ("best neural/fusion", best_in(("Neural", "Fusion"))),
        ("best overall", best),
    ]
    seen, key_u = set(), []
    for lab, b in key:
        if b and b not in seen:
            seen.add(b)
            key_u.append((f"{lab} ({b})", b))
    qrels = read_qrels(qrels_path) if qrels_ok else {}
    figs = (
        make_figures(
            tables["test"],
            results["test"],
            rows,
            spec,
            key_u,
            best,
            qrels,
            runs_dir,
            args.figures_dir,
        )
        if qrels_ok
        else []
    )
    phones_path = args.phones or get_path("phones")
    fc = (
        failure_cases(
            best, results["test"], spec["reference"], queries_path, qrels, runs_dir, phones_path
        )
        if qrels_ok
        else failure_cases(None, {}, "", queries_path, {}, runs_dir, phones_path)
    )
    (results_dir / "failure_cases.md").write_text(fc, encoding="utf-8")

    pending = [
        {"label": e["label"], "owner": e["owner"], "status": e["status"]}
        for e in tables["test"]["rows"]
        if e["status"] != "ok"
    ]
    summary = {
        "reference": spec["reference"],
        "selection_metric": metric,
        "qrels": str(qrels_path) if qrels_path else None,
        "best_overall": {
            "run": best,
            "dev": results["dev"].get(best, {}).get("metrics", {}).get(metric),
            "test": results["test"].get(best, {}).get("metrics", {}).get(metric),
        },
        "rows": {s: tables[s]["rows"] for s in SPLITS},
        "significance": {s: tables[s]["significance"] for s in SPLITS},
        "incremental": inc,
        "pending": pending,
        "figures": [str(p) for p in figs],
        "key_systems": key_u,
    }
    (results_dir / "ablation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(tables["test"]["markdown"])
    print(
        f"best overall (dev {metric}): {best} dev={summary['best_overall']['dev']} "
        f"test={summary['best_overall']['test']}"
    )
    print(
        f"pending rows: {len(pending)}; figures: {len(figs)}; wrote results/ablation_*.md|tex, "
        "ablation_incremental.*, ablation.json, failure_cases.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
