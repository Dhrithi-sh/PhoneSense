"""Create the hand-check CSVs (fill them in by hand, then run score_handcheck.py).

    python scripts/make_handcheck.py            # from the full build
    python scripts/make_handcheck.py --force    # overwrite existing (possibly filled!) CSVs

Writes results/handcheck_specs.csv (50 phones x 8 spec fields) and
results/handcheck_filter.csv (20 rejected items just below the filter threshold).
See phonesense/data/handcheck.py for how to fill the blank columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phonesense.config import get_config, get_path
from phonesense.data.common import Settings
from phonesense.data.handcheck import make


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--processed-dir", type=Path, default=None,
                    help="read a partial build (e.g. data/processed/partial_200000)")
    a = ap.parse_args()
    pl = get_config().get("pipeline", {})
    s = Settings.from_config(out_dir=a.processed_dir)
    make(phones_path=s.phones, phone_meta_path=s.phone_meta, filter_log_path=s.filter_log,
         out_specs=get_path("handcheck_specs"), out_filter=get_path("handcheck_filter"),
         n_phones=int(pl.get("handcheck_phones", 50)),
         n_rejects=int(pl.get("handcheck_rejects", 20)),
         threshold=s.threshold, margin=s.handcheck_margin, seed=s.seed, force=a.force)


if __name__ == "__main__":
    main()
