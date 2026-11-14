#!/usr/bin/env python3
"""Regenerate all of Viraja's results for the report, in one run.

Runs, in order:
  1. 50-name phonetic eval (accuracy@1 / MRR per matcher)
  2. Aksharantar word-level eval          (needs data/aksharantar/test_pairs.tsv)
  3. Edit-cost heatmap -> documentation/figures/edit_cost_heatmap.png
  4. Full-query experiment: expansion on/off + Rocchio on/off
                                           (needs the integrated main: index + ranker)

Parts whose data or teammates' modules aren't present are skipped with a note,
so this is safe to run on the Viraja branch alone or on integrated main.

    python scripts/viraja_results.py
"""

import os
import sys

# Make the dhvani package importable however this script is launched.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

COSTS = os.path.join("dhvani", "query", "edit_costs.json")
AKSH_TEST = os.path.join("data", "aksharantar", "test_pairs.tsv")
HEATMAP_OUT = os.path.join("documentation", "figures", "edit_cost_heatmap.png")


def _section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def _step(title, fn):
    _section(title)
    try:
        fn()
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - we want to keep going and report
        print(f"skipped: {type(exc).__name__}: {exc}")


def names_eval():
    from dhvani.query import names
    names.main([COSTS])


def aksharantar_eval():
    if not os.path.exists(AKSH_TEST):
        print("skipped: run `python -m dhvani.query.aksharantar` first "
              "(downloads hin.zip, writes data/aksharantar/*.tsv)")
        return
    from dhvani.query import evaluate
    evaluate.main([AKSH_TEST, COSTS, "--limit", "1500"])


def heatmap():
    os.makedirs(os.path.dirname(HEATMAP_OUT), exist_ok=True)
    from dhvani.query import heatmap as H
    H.main([COSTS, HEATMAP_OUT])


def full_query_and_rocchio():
    # Needs the integrated repo (Dhrithi's index, Rishit's ranker + metrics).
    from dhvani.query import experiment
    experiment.main()


def main():
    _step("1. 50-name phonetic eval", names_eval)
    _step("2. Aksharantar word-level eval", aksharantar_eval)
    _step("3. Edit-cost heatmap", heatmap)
    _step("4. Full-query experiment (expansion on/off, Rocchio on/off)", full_query_and_rocchio)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
