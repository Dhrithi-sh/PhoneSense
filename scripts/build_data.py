"""Build data/processed/{phones,reviews}.jsonl from the raw Amazon Reviews 2023 dataset (T02).

Stages (each in phonesense/data/, each cached; --force rebuilds):
  download -> filter_phones -> parse_specs -> dedupe -> build_reviews -> build_phones -> stats
then writes the teammates' sample (data/processed/sample/).

    python scripts/build_data.py                  # full run (= make data)
    python scripts/build_data.py --limit 200000   # first N raw lines of each file, quick test
    python scripts/build_data.py --force          # ignore cached stage outputs

A --limit run writes to data/processed/partial_<N>/ (and its own stats file there), so it
can never be mistaken for, or overwrite, the full build.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from phonesense.config import get_config, get_path
from phonesense.data import (
    build_phones,
    build_reviews,
    dedupe,
    download,
    filter_phones,
    parse_specs,
    stats,
)
from phonesense.data.common import Settings
from phonesense.data.sample import build_sample

STAGES = [
    ("filter_phones", filter_phones.run),
    ("parse_specs", parse_specs.run),
    ("dedupe", dedupe.run),
    ("build_reviews", build_reviews.run),
    ("build_phones", build_phones.run),
]


def partial_dir(limit: int) -> Path:
    return get_path("processed_dir") / f"partial_{limit}"


def run_pipeline(s: Settings, *, force: bool = False, skip_download: bool = False) -> dict:
    t0 = time.time()
    if not skip_download and not download.run(s):
        sys.exit("raw files missing: see the message above")
    for name, fn in STAGES:
        t = time.time()
        fn(s, force=force)
        print(f"  ({name}: {time.time() - t:.1f}s)")
    st = stats.run(s)
    print(f"[build_data] done in {time.time() - t0:.1f}s")
    return st


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N raw lines of each file (quick test)")
    ap.add_argument("--force", action="store_true", help="rebuild every stage")
    ap.add_argument("--skip-download", action="store_true",
                    help="never try to download (raw files must already be in data/raw/)")
    ap.add_argument("--no-sample", action="store_true", help="don't write the sample")
    ap.add_argument("--quiet", action="store_true", help="no progress bars")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    out_dir = partial_dir(a.limit) if a.limit else None
    s = Settings.from_config(limit=a.limit, out_dir=out_dir, progress=not a.quiet)
    run_pipeline(s, force=a.force, skip_download=a.skip_download)
    if not a.no_sample:
        cfg = get_config().get("pipeline", {})
        sample_dir = get_path("sample_dir") if a.limit is None else s.out_dir / "sample"
        build_sample(s.phones, s.reviews, sample_dir, int(cfg.get("sample_size", 500)), s.seed)


if __name__ == "__main__":
    main()
