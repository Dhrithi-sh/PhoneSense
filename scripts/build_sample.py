"""Write the teammates' sample: data/processed/sample/{phones,reviews}.jsonl.

    python scripts/build_sample.py                 # sample from the full build
    python scripts/build_sample.py --limit 2000000 # if the full run is slow: build a partial
                                                   # run on the first N raw lines, sample it

The sample has ``pipeline.sample_size`` phones stratified by brand (seed from config.yaml)
and all of their reviews.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_data import partial_dir, run_pipeline  # noqa: E402

from phonesense.config import get_config, get_path  # noqa: E402
from phonesense.data.common import Settings  # noqa: E402
from phonesense.data.sample import build_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="first run the pipeline on the first N raw lines (partial build)")
    ap.add_argument("--n", type=int, default=None, help="phones in the sample")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: sample_dir)")
    a = ap.parse_args()
    n = a.n or int(get_config().get("pipeline", {}).get("sample_size", 500))
    if a.limit:
        s = Settings.from_config(limit=a.limit, out_dir=partial_dir(a.limit))
        run_pipeline(s)
    else:
        s = Settings.from_config()
        if not s.phones.exists():
            sys.exit(f"{s.phones} not found: run `python scripts/build_data.py` first, "
                     "or use --limit N for a partial build")
    build_sample(s.phones, s.reviews, a.out or get_path("sample_dir"), n, s.seed)


if __name__ == "__main__":
    main()
