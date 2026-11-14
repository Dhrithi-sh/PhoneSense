"""Score the filled hand-check CSVs -> results/handcheck_summary.json (numbers for the report).

    python scripts/score_handcheck.py
"""

from __future__ import annotations

import json

from phonesense.config import get_path
from phonesense.data.handcheck import score


def main() -> None:
    out = score(specs_csv=get_path("handcheck_specs"), filter_csv=get_path("handcheck_filter"),
                out_path=get_path("handcheck_summary"))
    print(json.dumps({k: v for k, v in out.items() if k != "spec_accuracy_per_field"},
                     indent=2))


if __name__ == "__main__":
    main()
