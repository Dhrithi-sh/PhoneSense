"""Fusion: RRF and linear scores by hand, tie-breaking, weights, FusionRetriever, tuning."""

import math

import pytest

from phonesense.retrieval.base import Retriever
from phonesense.retrieval.fusion import (
    FusionRetriever,
    minmax_normalise,
    reciprocal_rank_fusion,
    weighted_linear_fusion,
)
from phonesense.retrieval.fusion_tune import (
    evaluable_qids,
    fuse_runs,
    mean_ndcg,
    ndcg_at_k,
    pick_best,
    tune_linear,
    tune_rrf_k,
    weight_grid,
)
from phonesense.schemas import Query, validate_run_name

A = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
B = [("d2", 0.9), ("d4", 0.5), ("d1", 0.1)]


def scores(ranking):
    return dict(ranking)


# ---------------------------------------------------------------------------- RRF


def test_rrf_scores_by_hand():
    """k=60. d1: ranks 1 and 3 -> 1/61 + 1/63; d2: ranks 2 and 1 -> 1/62 + 1/61;
    d3: rank 3 in A only -> 1/63; d4: rank 2 in B only -> 1/62."""
    got = scores(reciprocal_rank_fusion([A, B], k=60))
    assert got["d1"] == pytest.approx(1 / 61 + 1 / 63)
    assert got["d2"] == pytest.approx(1 / 62 + 1 / 61)
    assert got["d3"] == pytest.approx(1 / 63)
    assert got["d4"] == pytest.approx(1 / 62)
    assert [d for d, _ in reciprocal_rank_fusion([A, B])] == ["d2", "d1", "d4", "d3"]


def test_rrf_ignores_the_score_values_and_only_uses_order():
    scaled = [(d, s * 1000 + 7) for d, s in A]
    assert reciprocal_rank_fusion([scaled, B]) == reciprocal_rank_fusion([A, B])


def test_rrf_ties_break_by_doc_id():
    # x and y each get rank 1 once and rank 2 once: identical fused score, so doc_id decides.
    one, two = [("y", 1.0), ("x", 0.5)], [("x", 1.0), ("y", 0.5)]
    out = reciprocal_rank_fusion([one, two])
    assert out[0][1] == pytest.approx(out[1][1])
    assert [d for d, _ in out] == ["x", "y"]
    assert [d for d, _ in reciprocal_rank_fusion([two, one])] == ["x", "y"]


def test_rrf_three_way_tie_is_not_broken_by_float_noise():
    rankings = [[("c", 1), ("b", 1), ("a", 1)], [("a", 1), ("c", 1), ("b", 1)],
                [("b", 1), ("a", 1), ("c", 1)]]
    assert [d for d, _ in reciprocal_rank_fusion(rankings)] == ["a", "b", "c"]


def test_rrf_weights_depth_and_k():
    assert [d for d, _ in reciprocal_rank_fusion([A, B], weights=[1, 0])] == ["d1", "d2", "d3"]
    assert len(reciprocal_rank_fusion([A, B], depth=2)) == 2
    # a large k flattens the head: the rank-1 vs rank-2 gap shrinks from 1/2-1/3 to ~1e-6
    gap = lambda k: (lambda s: s["d1"] - s["d2"])(scores(reciprocal_rank_fusion([A], k=k)))  # noqa: E731
    assert gap(1) == pytest.approx(1 / 2 - 1 / 3)
    assert gap(1000) < gap(60) < gap(1)


def test_rrf_repeated_doc_counts_once_at_its_best_position():
    dup = [("d1", 3.0), ("d1", 2.0), ("d2", 1.0)]
    got = scores(reciprocal_rank_fusion([dup]))
    assert got["d1"] == pytest.approx(1 / 61) and got["d2"] == pytest.approx(1 / 62)


def test_rrf_does_not_modify_inputs_and_handles_empty():
    before = [list(A), list(B)]
    reciprocal_rank_fusion(before)
    assert before == [A, B]
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([]) == []


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"k": -5}, {"weights": [1]}, {"weights": [-1, 2]},
                                    {"weights": [0, 0]}])
def test_rrf_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([A, B], **kwargs)


# ---------------------------------------------------------------------------- linear


def test_minmax_normalise_edge_cases():
    assert minmax_normalise(A) == {"d1": 1.0, "d2": 0.5, "d3": 0.0}
    assert minmax_normalise([("x", 5.0)]) == {"x": 1.0}
    assert minmax_normalise([("x", 2.0), ("y", 2.0)]) == {"x": 1.0, "y": 1.0}
    assert minmax_normalise([]) == {}
    assert minmax_normalise([("n", -4.0), ("p", 4.0)]) == {"n": 0.0, "p": 1.0}


def test_linear_scores_by_hand():
    """norm(A) = d1 1, d2 .5, d3 0;  norm(B) = d2 1, d4 .5, d1 0.  Equal weights:
    d1 = (1+0)/2 = .5, d2 = (.5+1)/2 = .75, d3 = 0, d4 = .25."""
    got = scores(weighted_linear_fusion([A, B]))
    assert got == {"d1": pytest.approx(0.5), "d2": pytest.approx(0.75),
                   "d3": pytest.approx(0.0), "d4": pytest.approx(0.25)}
    # weights 3:1 -> d1 = .75, d2 = (1.5+1)/4 = .625, d4 = .125
    w = scores(weighted_linear_fusion([A, B], [3, 1]))
    assert w["d1"] == pytest.approx(0.75) and w["d2"] == pytest.approx(0.625)
    assert w["d4"] == pytest.approx(0.125)


def test_linear_weights_are_scale_free_and_ties_break_by_doc_id():
    same = scores(weighted_linear_fusion([A, B], [1, 1]))
    assert scores(weighted_linear_fusion([A, B], [5, 5])) == pytest.approx(same)
    tie = weighted_linear_fusion([[("b", 1.0), ("a", 0.0)], [("a", 1.0), ("b", 0.0)]])
    assert [d for d, _ in tie] == ["a", "b"]


# ---------------------------------------------------------------------------- retriever


class Fixed(Retriever):
    def __init__(self, name, ranking):
        self.name, self.ranking, self.fitted, self.asked = name, ranking, False, []

    def fit(self, phones):
        self.fitted = True

    def search(self, query, k=100):
        self.asked.append(k)
        return self.ranking[:k]


Q = Query(qid="q01", text="anything", split="dev")


def test_fusion_retriever_matches_the_function_and_is_a_valid_run_name():
    a, b = Fixed("a", A), Fixed("b", B)
    r = FusionRetriever([a, b], method="rrf", rrf_k=60)
    assert r.search(Q, 10) == reciprocal_rank_fusion([A, B], 60)
    assert r.search(Q, 2) == reciprocal_rank_fusion([A, B], 60, depth=2)
    validate_run_name(r.name + "_dev")
    lin = FusionRetriever([a, b], method="linear", weights=[3, 1])
    assert lin.search(Q, 10) == weighted_linear_fusion([A, B], [3, 1])
    validate_run_name(lin.name + "_dev")


def test_fusion_retriever_asks_children_for_depth_and_fits_them():
    a, b = Fixed("a", A), Fixed("b", B)
    r = FusionRetriever([a, b], depth=50)
    r.fit([])
    assert a.fitted and b.fitted
    r.search(Q, 10)
    assert a.asked == [50] and b.asked == [50]
    c = Fixed("c", A)
    FusionRetriever([c], fit_children=False).fit([])
    assert not c.fitted


def test_fusion_retriever_validates_arguments():
    with pytest.raises(ValueError):
        FusionRetriever([])
    with pytest.raises(ValueError):
        FusionRetriever([Fixed("a", A)], method="borda")
    with pytest.raises(ValueError):
        FusionRetriever([Fixed("a", A), Fixed("b", B)], weights=[1.0])


# ---------------------------------------------------------------------------- tuning


def test_ndcg_by_hand():
    """grades d1=3, d2=0, d3=2. Ranking d2, d3, d1:
    DCG = 0/log2(2) + 3/log2(3) + 7/log2(4);  ideal (3,2): 7/1 + 3/log2(3)."""
    grades = {"d1": 3, "d2": 0, "d3": 2}
    ranking = [("d2", 0), ("d3", 0), ("d1", 0)]
    dcg = 3 / math.log2(3) + 7 / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3)
    assert ndcg_at_k(ranking, grades, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([("d1", 0), ("d3", 0)], grades) == pytest.approx(1.0)
    assert ndcg_at_k([("zz", 0)], grades) == 0.0  # unjudged counts as 0
    assert ndcg_at_k(A, {"d1": 0}) == 0.0  # nothing relevant to find


def test_evaluable_qids_and_mean_ndcg():
    qrels = {"q1": {"d1": 2}, "q2": {"d1": 0}, "q3": {"d1": 3}}
    assert evaluable_qids(["q1", "q2", "q3", "q4"], qrels) == ["q1", "q3"]
    perfect = {"q1": [("d1", 1.0)], "q3": []}
    assert mean_ndcg(perfect, qrels, ["q1", "q3"]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mean_ndcg(perfect, qrels, [])


def test_weight_grid_sums_to_one_and_is_deterministic():
    grid = list(weight_grid(3, 0.25))
    assert all(sum(w) == pytest.approx(1.0) for w in grid)
    assert len(grid) == 15 and grid[0] == (0.0, 0.0, 1.0) and grid[-1] == (1.0, 0.0, 0.0)
    assert list(weight_grid(2, 0.5)) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    with pytest.raises(ValueError):
        list(weight_grid(2, 0.3))


def test_pick_best_prefers_equal_weights_on_ties():
    trials = [{"weights": [1.0, 0.0], "score": 0.8}, {"weights": [0.5, 0.5], "score": 0.8},
              {"weights": [0.2, 0.8], "score": 0.7}]
    assert pick_best(trials)["weights"] == [0.5, 0.5]
    assert pick_best([{"weights": [0.9, 0.1], "score": 0.9}] + trials)["weights"] == [0.9, 0.1]
    with pytest.raises(ValueError):
        pick_best([])


def test_tuning_finds_the_good_system_and_best_rrf_k():
    # The relevant doc (c, z) has the LARGEST id, so a doc_id tie-break can never rescue a bad mix.
    qrels = {"q1": {"c": 3, "b": 0, "a": 0}, "q2": {"z": 3, "y": 0, "x": 0}}
    good = {"q1": [("c", 1.0), ("b", 0.5), ("a", 0.1)], "q2": [("z", 1.0), ("y", 0.5), ("x", 0.1)]}
    bad = {"q1": [("a", 1.0), ("b", 0.5), ("c", 0.1)], "q2": [("x", 1.0), ("y", 0.5), ("z", 0.1)]}
    qids = ["q1", "q2"]
    equal = mean_ndcg(fuse_runs([good, bad], qids, "linear"), qrels, qids)
    best = pick_best(tune_linear([good, bad], qrels, qids, step=0.25))
    assert best["weights"][0] > best["weights"][1]
    assert best["score"] == pytest.approx(1.0) and equal < best["score"]
    rrf = tune_rrf_k([good, bad], qrels, qids, k_values=(1, 60))
    assert [t["rrf_k"] for t in rrf] == [1, 60]
    fused = fuse_runs([good, bad], qids, "linear", [1.0, 0.0])
    assert fused["q1"][0][0] == "c"
    with pytest.raises(ValueError):
        fuse_runs([good], qids, "borda")
