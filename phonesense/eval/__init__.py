"""Evaluation harness (owner: A). T11.

    from phonesense.eval import evaluate_run, compare, compare_many, cohen_kappa, results_table

Entry point: ``python -m phonesense.eval.cli`` (eval / compare / table / all / agree).
"""

from phonesense.eval.agreement import agreement_report, cohen_kappa, weighted_kappa
from phonesense.eval.evaluate import evaluate_rankings, evaluate_run, resolve_qrels_path
from phonesense.eval.significance import compare, compare_many, holm
from phonesense.eval.tables import results_table

__all__ = [
    "agreement_report",
    "cohen_kappa",
    "compare",
    "compare_many",
    "evaluate_rankings",
    "evaluate_run",
    "holm",
    "resolve_qrels_path",
    "results_table",
    "weighted_kappa",
]
