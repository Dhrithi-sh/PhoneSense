"""Build the submission ZIP, dry-run it, and write SUBMISSION_CHECKLIST.md.

    python scripts/make_submission.py               # = make submission
    python scripts/make_submission.py --no-dry-run

1. ``dist/PhoneSense_submission.zip`` (top folder ``PhoneSense/``) with the code, tests,
   configs, ``queries/``, ``qrels/``, ``runs/``, ``results/``, ``report/`` (any PDF too),
   README, CONTRACT.md, ``legacy/PORTING.md`` and the 500-phone sample in
   ``data/processed/sample/``. Excluded: ``data/raw``, the full processed data, ``data/index``,
   ``legacy/midterm/``, ``.git``, virtualenvs, caches, ``.env``, ``dist/``.
   Prints the size and the 10 largest files; warns above 50 MB.
2. Dry run: unzips into a temp dir and runs ``pytest`` there with the current interpreter,
   plus ``python -m phonesense.eval.cli table`` on the included results (skipped if there are
   no results JSON files yet). Result saved to ``results/submission_dryrun.json``.
3. ``SUBMISSION_CHECKLIST.md``: what is present, what is missing and who owns it; also added
   to the ZIP.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from phonesense.config import get_config, repo_root

TOP = "PhoneSense"
INCLUDE_FILES = [
    "README.md",
    "CONTRACT.md",
    "config.yaml",
    "ablation.yaml",
    "pyproject.toml",
    "Makefile",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-dense.txt",
    "requirements-ui.txt",
    ".env.example",
    ".gitignore",
    "legacy/PORTING.md",
    "CONTRIBUTIONS.md",
]
INCLUDE_DIRS = [
    "phonesense",
    "scripts",
    "tests",
    "queries",
    "qrels",
    "runs",
    "results",
    "report",
    "data/processed/sample",
]
EXCLUDE = [
    "*/__pycache__/*",
    "*.pyc",
    "*/.pytest_cache/*",
    "*/.ruff_cache/*",
    "*.egg-info/*",
    ".env",
    "*/.env",
    "dist/*",
    "data/raw/*",
    "data/index/*",
    "legacy/midterm/*",
    ".git/*",
    ".venv/*",
    "venv/*",
    "*/.ipynb_checkpoints/*",
    "*.faiss",
    "*.npy",
]
WARN_MB = 50


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_FILES:
        if (root / rel).is_file():
            files.append(root / rel)
    for rel in INCLUDE_DIRS:
        d = root / rel
        if d.is_dir():
            files += [p for p in d.rglob("*") if p.is_file()]
    out = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        if not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat) for pat in EXCLUDE):
            out.append(p)
    return sorted(set(out))


def build_zip(root: Path, out: Path) -> list[tuple[str, int]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect(root)
    sizes = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            rel = p.relative_to(root).as_posix()
            zf.write(p, f"{TOP}/{rel}")
            sizes.append((rel, p.stat().st_size))
    return sizes


def dry_run(zip_path: Path) -> dict:
    """Unzip into a temp dir; run pytest and the eval table CLI there."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        root = Path(tmp) / TOP
        env = {**os.environ, "PHONESENSE_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PHONESENSE_CONFIG", None)
        which = subprocess.run(
            [sys.executable, "-c", "import phonesense; print(phonesense.__file__)"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        t = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        tail = (t.stdout.strip().splitlines() or [""])[-1]
        results = (
            [p for p in (root / "results").glob("*.json")] if (root / "results").exists() else []
        )
        table = {"status": "skipped (no results JSON yet)"}
        if results:
            c = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "phonesense.eval.cli",
                    "table",
                    "--results",
                    "results/*.json",
                    "--permutations",
                    "200",
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            ok_tables = "no results files matched" not in c.stderr
            table = {
                "status": ("passed" if c.returncode == 0 else "failed")
                if ok_tables
                else "skipped (no results JSON yet)",
                "stderr_tail": c.stderr[-500:],
            }
        imported_from = which.stdout.strip()
        self_contained = imported_from.startswith(str(root))
        return {
            "pytest": "passed" if t.returncode == 0 else "failed",
            "pytest_summary": tail,
            "pytest_output_tail": t.stdout[-2000:] if t.returncode else "",
            "table_cli": table,
            "phonesense_imported_from_zip": self_contained,
            "passed": t.returncode == 0 and table["status"] != "failed" and self_contained,
        }


def tracked(root: Path, rel: str) -> bool:
    """True if ``rel`` is committed in git (False outside a git checkout)."""
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True, text=True
    )
    return r.returncode == 0


def checklist(root: Path, dry: dict | None, zip_mb: float | None) -> str:
    res = root / "results"
    abl = (
        json.loads((res / "ablation.json").read_text(encoding="utf-8"))
        if (res / "ablation.json").exists()
        else None
    )
    cfg = get_config()
    pdfs = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "report").rglob("*.pdf")
        if "figures" not in p.parts
    )
    figs = [
        "ablation_bars.png",
        "pr_key_systems.png",
        "delta_best_vs_bm25.png",
        "bm25_tuning_heatmap.png",
    ]
    missing_figs = [f for f in figs if not (root / "report" / "figures" / f).exists()]
    pending = abl["pending"] if abl else [{"label": "ablation not run", "owner": "A"}]
    items = [
        (bool(pdfs), "Report PDF in report/", "A/B/C", ", ".join(pdfs) or "no PDF yet"),
        (
            tracked(root, "queries/queries.jsonl"),
            "queries/queries.jsonl committed",
            "B",
            "merge b/T08-queries"
            + (
                " (only an untracked local copy exists)"
                if (root / "queries" / "queries.jsonl").exists()
                else ""
            ),
        ),
        (
            tracked(root, "qrels/qrels.txt"),
            "qrels/qrels.txt (final labels) committed",
            "B",
            "T09: LLM judge + human merge, then commit",
        ),
        (
            (res / "kappa.json").exists(),
            "Kappa LLM vs. human reported (results/kappa.json)",
            "B",
            "scripts/kappa.py on b/T09b-kappa",
        ),
        (
            (res / "handcheck_summary.json").exists(),
            "Hand check scored",
            "A",
            "fill the CSVs, then make handcheck-score",
        ),
        (
            bool((cfg.get("bm25") or {}).get("tuned")),
            "BM25 tuned on dev (config bm25.tuned)",
            "A",
            "make tune (needs qrels)",
        ),
        (
            bool((cfg.get("phrase") or {}).get("tuned")),
            "Phrase weight tuned on dev",
            "A",
            "python scripts/tune_phrase.py",
        ),
        (
            not pending,
            "All ablation rows non-pending",
            "A/B/C",
            "; ".join(f"{p['label']} ({p['owner']})" for p in pending),
        ),
        (not missing_figs, "Figures present", "A", ", ".join(missing_figs)),
        (
            (root / "report" / "generated" / "numbers.tex").exists(),
            "report/generated/ snippets",
            "A",
            "make report-numbers",
        ),
        (
            dry is not None and dry.get("pytest") == "passed",
            "Tests pass (in the unzipped copy)",
            "A",
            (dry or {}).get("pytest_summary", "not run"),
        ),
        (
            dry is not None and dry.get("passed"),
            "ZIP dry run passed",
            "A",
            "" if dry is None else f"table CLI: {dry['table_cli']['status']}",
        ),
        (
            zip_mb is not None and zip_mb <= WARN_MB,
            f"ZIP <= {WARN_MB} MB",
            "A",
            "not built" if zip_mb is None else f"{zip_mb:.1f} MB",
        ),
    ]
    lines = [
        "# Submission checklist",
        "",
        "Generated by `scripts/make_submission.py`; re-run it after changes.",
        "",
    ]
    for ok, what, owner, note in items:
        lines.append(
            f"- [{'x' if ok else ' '}] {what}" + ("" if ok else f" (owner {owner}: {note})")
        )
    todo = [(what, owner, note) for ok, what, owner, note in items if not ok]
    lines += ["", "## Still missing", ""]
    lines += [f"- **{owner}**: {what}: {note}" for what, owner, note in todo] or ["- nothing"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=repo_root() / "dist" / "PhoneSense_submission.zip")
    ap.add_argument("--no-dry-run", action="store_true")
    args = ap.parse_args(argv)
    root = repo_root()
    sizes = build_zip(root, args.out)
    mb = args.out.stat().st_size / 2**20
    print(
        f"wrote {args.out}: {len(sizes)} files, {mb:.2f} MB compressed, "
        f"{sum(s for _, s in sizes) / 2**20:.2f} MB uncompressed"
    )
    print("10 largest files:")
    for rel, s in sorted(sizes, key=lambda x: -x[1])[:10]:
        print(f"  {s / 2**20:8.2f} MB  {rel}")
    if mb > WARN_MB:
        print(f"WARNING: the ZIP is over {WARN_MB} MB", file=sys.stderr)
    dry = None
    if not args.no_dry_run:
        dry = dry_run(args.out)
        (root / "results").mkdir(exist_ok=True)
        (root / "results" / "submission_dryrun.json").write_text(
            json.dumps(dry, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"dry run: {'PASSED' if dry['passed'] else 'FAILED'} (pytest: "
            f"{dry['pytest_summary']}; table CLI: {dry['table_cli']['status']}; imported from "
            f"zip: {dry['phonesense_imported_from_zip']})"
        )
        if dry["pytest"] == "failed":
            print(dry["pytest_output_tail"], file=sys.stderr)
    text = checklist(root, dry, mb)
    (root / "SUBMISSION_CHECKLIST.md").write_text(text, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(args.out, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{TOP}/SUBMISSION_CHECKLIST.md", text)  # checklist is never collected above
    print("wrote SUBMISSION_CHECKLIST.md")
    print(text)
    return 0 if dry is None or dry["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
