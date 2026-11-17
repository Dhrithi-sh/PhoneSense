import random

import pytest

from phonesense.eval import evaluate_rankings
from phonesense.eval.metrics import (
    average_precision,
    canonical_metric,
    f1_at_k,
    interpolated_precision_11pt,
    judged_at_k,
    metric_names,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

# Hand-worked example. Ranking d1..d10. Judgments (grade):
#   d1 3, d2 0, d3 2, d4 1 (marginal: NOT relevant at threshold 2), d6 2, d11 3 (not retrieved)
#   d5, d7, d8, d9, d10 are unjudged -> non-relevant.
# Relevant (grade >= 2): d1, d3, d6, d11  -> |rel| = 4
RANKING = [f"d{i}" for i in range(1, 11)]
QRELS = {"d1": 3, "d2": 0, "d3": 2, "d4": 1, "d6": 2, "d11": 3}


def test_precision_recall_f1():
    assert precision_at_k(RANKING, QRELS, 5, 2) == pytest.approx(2 / 5)  # d1, d3
    assert precision_at_k(RANKING, QRELS, 10, 2) == pytest.approx(3 / 10)  # + d6
    assert recall_at_k(RANKING, QRELS, 10, 2) == pytest.approx(3 / 4)
    # F1@10 = 2 * 0.3 * 0.75 / (0.3 + 0.75) = 0.45 / 1.05
    assert f1_at_k(RANKING, QRELS, 10, 2) == pytest.approx(0.4285714286, abs=1e-9)


def test_ap_rr_rprec():
    # relevant at ranks 1, 3, 6: (1/1 + 2/3 + 3/6) / 4 = 2.1666667 / 4
    assert average_precision(RANKING, QRELS, 2) == pytest.approx(0.5416666667, abs=1e-9)
    assert reciprocal_rank(RANKING, QRELS, 2) == 1.0
    assert reciprocal_rank(["d2", "d4", "d3"], QRELS, 2) == pytest.approx(1 / 3)
    # R = 4: top 4 = d1 d2 d3 d4 -> 2 relevant -> 0.5
    assert r_precision(RANKING, QRELS, 2) == 0.5


def test_ndcg_graded():
    # gains 2^g - 1: d1 7, d2 0, d3 3, d4 1, d5 0 (unjudged)
    # DCG@5  = 7/log2(2) + 3/log2(4) + 1/log2(5) = 7 + 1.5 + 0.4306766 = 8.9306766
    # ideal grades from ALL judged docs: 3 3 2 2 1 (0) ->
    # IDCG@5 = 7/1 + 7/log2(3) + 3/log2(4) + 3/log2(5) + 1/log2(6)
    #        = 7 + 4.4165082 + 1.5 + 1.2920297 + 0.3868528 = 14.5953908
    assert ndcg_at_k(RANKING, QRELS, 5) == pytest.approx(8.9306766 / 14.5953908, abs=1e-6)
    assert ndcg_at_k(RANKING, QRELS, 5) == pytest.approx(0.6118833, abs=1e-6)
    assert ndcg_at_k(["d1", "d11", "d3", "d6", "d4"], QRELS, 5) == pytest.approx(1.0)


def test_judged():
    # judged in top 10: d1 d2 d3 d4 d6 -> 0.5; unjudged d5 d7..d10
    assert judged_at_k(RANKING, QRELS, 10) == 0.5
    assert judged_at_k(["x"], QRELS, 10) == 0.0


def test_threshold_one_counts_marginal():
    assert precision_at_k(RANKING, QRELS, 5, threshold=1) == pytest.approx(3 / 5)


def test_k_larger_than_run():
    short = ["d1", "d3"]
    assert precision_at_k(short, QRELS, 10, 2) == pytest.approx(2 / 10)  # divides by k
    assert recall_at_k(short, QRELS, 100, 2) == pytest.approx(2 / 4)
    assert 0 < ndcg_at_k(short, QRELS, 20) < 1


def test_empty_run_all_zeros():
    m = score_query([], QRELS, threshold=2)
    assert set(m) == set(metric_names())
    assert all(v == 0.0 for v in m.values())


def test_no_relevant_docs_excluded_and_unjudged_listed():
    qrels = {"q01": QRELS, "q02": {"a": 1, "b": 0}, "q03": {}}
    per_query, means, excluded, unjudged = evaluate_rankings(
        {"q01": RANKING, "q02": ["a"]}, qrels, {"q01", "q02", "q03", "q04"}, threshold=2
    )
    assert list(per_query) == ["q01"]
    assert excluded == ["q02"] and unjudged == ["q03", "q04"]
    assert means["MAP"] == pytest.approx(0.5416666667)


def test_missing_query_in_run_scores_zero():
    per_query, means, _, _ = evaluate_rankings({}, {"q01": QRELS, "q02": {"x": 3}}, threshold=2)
    assert per_query["q01"]["MAP"] == 0.0 and means["nDCG@10"] == 0.0


def test_11pt():
    # recall 0.25 at rank 1 (P 1), 0.5 at rank 3 (P 2/3), 0.75 at rank 6 (P .5), never 1.0
    pts = interpolated_precision_11pt(RANKING, QRELS, 2)
    assert pts[0] == 1.0 and pts[2] == 1.0 and pts[3] == pytest.approx(2 / 3)
    assert pts[5] == pytest.approx(2 / 3) and pts[7] == 0.5 and pts[8] == 0.0 and pts[10] == 0.0


def test_canonical_names():
    assert canonical_metric("ndcg@10") == "nDCG@10"
    assert canonical_metric("map") == "MAP"
    assert canonical_metric("JUDGED@10") == "judged@10"
    with pytest.raises(ValueError):
        canonical_metric("bogus")


def test_bad_k():
    with pytest.raises(ValueError):
        precision_at_k(RANKING, QRELS, 0, 2)


def test_matches_pytrec_eval():
    """Test-only cross-check. pytrec_eval's ndcg uses LINEAR gains by default, so we compare
    MAP, P@10 and recall only (with relevance_level=2 to match our threshold)."""
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(0)
    docs = [f"D{i}" for i in range(60)]
    qrels, run = {}, {}
    for q in range(15):
        qid = f"q{q:02d}"
        qrels[qid] = {d: rng.choice([0, 1, 2, 3]) for d in rng.sample(docs, 20)}
        ranked = rng.sample(docs, 30)
        run[qid] = {d: float(30 - i) for i, d in enumerate(ranked)}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"map", "P_10", "recall_10"}, relevance_level=2)
    ref = ev.evaluate(run)
    for qid, scores in run.items():
        ranking = sorted(scores, key=lambda d: -scores[d])
        if not any(g >= 2 for g in qrels[qid].values()):
            continue
        assert average_precision(ranking, qrels[qid], 2) == pytest.approx(ref[qid]["map"])
        assert precision_at_k(ranking, qrels[qid], 10, 2) == pytest.approx(ref[qid]["P_10"])
        assert recall_at_k(ranking, qrels[qid], 10, 2) == pytest.approx(ref[qid]["recall_10"])
