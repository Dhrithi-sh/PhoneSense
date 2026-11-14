import math

from dhvani.eval.metrics import (
    average_11_point,
    average_precision,
    dcg_at_k,
    evaluate,
    f1_at_k,
    interpolated_11_point,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    read_qrels,
    read_run,
    recall_at_k,
    write_run,
)


def ranked(n):
    return [f"d{i}" for i in range(1, n + 1)]


# Lecture 8 MAP example. Query 1 has 5 relevant docs, found at ranks
# 1, 3, 6, 9, 10. Query 2 has 3 relevant docs, found at ranks 2, 5, 7.
Q1 = {f"d{r}": 1 for r in (1, 3, 6, 9, 10)}
Q2 = {f"d{r}": 1 for r in (2, 5, 7)}


def test_lecture_average_precision_query_1():
    # (1.0 + 0.67 + 0.5 + 0.44 + 0.5) / 5 = 0.62
    assert round(average_precision(ranked(10), Q1), 2) == 0.62


def test_lecture_average_precision_query_2():
    # (0.5 + 0.4 + 0.43) / 3 = 0.44
    assert round(average_precision(ranked(10), Q2), 2) == 0.44


def test_lecture_map_is_0_53():
    rankings = {"q1": ranked(10), "q2": ranked(10)}
    assert round(mean_average_precision(rankings, {"q1": Q1, "q2": Q2}), 2) == 0.53


def test_lecture_precision_at_5():
    # 5 retrieved, 3 relevant: P@5 = 0.6
    qrels = {"d1": 1, "d2": 1, "d4": 1}
    assert precision_at_k(ranked(5), qrels, 5) == 0.6


def test_lecture_recall_at_5():
    # 6 relevant in total, 3 of them in the top 5: R@5 = 0.5
    qrels = {"d1": 1, "d2": 1, "d4": 1, "d20": 1, "d21": 1, "d22": 1}
    assert recall_at_k(ranked(5), qrels, 5) == 0.5


def test_missed_relevant_docs_lower_ap():
    # One relevant doc at rank 1 and one never retrieved: AP = (1/1) / 2
    assert average_precision(["a", "b"], {"a": 1, "z": 1}) == 0.5


def test_f1_is_harmonic_mean():
    qrels = {"d1": 1, "d2": 1, "d4": 1, "d20": 1, "d21": 1, "d22": 1}
    p, r = 0.6, 0.5
    assert abs(f1_at_k(ranked(5), qrels, 5) - 2 * p * r / (p + r)) < 1e-12


def test_dcg_and_ndcg_by_hand():
    qrels = {"a": 2, "b": 0, "c": 1}
    ranking = ["b", "a", "c"]
    expected_dcg = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert abs(dcg_at_k(ranking, qrels, 3) - expected_dcg) < 1e-12
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert abs(ndcg_at_k(ranking, qrels, 3) - expected_dcg / ideal) < 1e-12
    assert ndcg_at_k(["a", "c", "b"], qrels, 3) == 1.0


def test_interpolated_precision_never_goes_up():
    curve = interpolated_11_point(ranked(10), Q1)
    assert len(curve) == 11
    assert curve[0] == 1.0
    assert all(a >= b for a, b in zip(curve, curve[1:]))


def test_average_11_point_over_queries():
    rankings = {"q1": ranked(10), "q2": ranked(10)}
    curve = average_11_point(rankings, {"q1": Q1, "q2": Q2})
    assert len(curve) == 11 and all(0 <= p <= 1 for p in curve)


def test_queries_without_relevant_docs_are_skipped():
    per_query, means = evaluate({"q1": ranked(10)}, {"q1": Q1, "q_empty": {"d1": 0}}, k=5)
    assert set(per_query) == {"q1"}
    assert set(means) == {"P@5", "R@5", "MAP", "nDCG@5"}


def test_run_and_qrels_files_round_trip(tmp_path):
    run_path = tmp_path / "run.txt"
    write_run(run_path, {"N01": [("doc_a", 0.9), ("doc_b", 0.5)]}, "lnc")
    assert read_run(run_path) == {"N01": ["doc_a", "doc_b"]}
    assert run_path.read_text().splitlines()[0] == "N01 Q0 doc_a 1 0.900000 lnc"

    qrels_path = tmp_path / "qrels.txt"
    qrels_path.write_text("N01 0 doc_a 2\nN01 0 doc_c 1\n")
    assert read_qrels(qrels_path) == {"N01": {"doc_a": 2, "doc_c": 1}}
