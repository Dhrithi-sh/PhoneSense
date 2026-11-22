"""Evaluate every run in runs/ against qrels and write results/*.json + results/table_<split>.md.

    python scripts/run_eval.py              # = make eval
    python scripts/run_eval.py --qrels qrels/llm.txt

Same as ``python -m phonesense.eval.cli all``. Qrels default to qrels/qrels.txt, falling back
to qrels/llm.txt with a warning.
"""

from __future__ import annotations

import sys

from phonesense.eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["all", *sys.argv[1:]]))
