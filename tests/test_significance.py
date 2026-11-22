import random

import pytest

from phonesense.eval.agreement import agreement_report, cohen_kappa, weighted_kappa
from phonesense.eval.significance import (
    compare,
    compare_many,
    holm,
    randomization_test,
    t_test,
)


def _pq(values):
    return {f"q{i:02d}": {"nDCG@10": v} for i, v in enumerate(values)}


def test_identical_runs_p_one():
    vals = [random.Random(1).random() for _ in range(30)]
    r = compare(_pq(vals), _pq(vals), "nDCG@10", permutations=2000, seed=0)
    assert r["p_randomization"] == pytest.approx(1.0)
    assert r["p_ttest"] == 1.0 and r["tied"] == 30 and r["mean_diff"] == 0.0


def test_uniformly_better_is_significant():
    rng = random.Random(2)
    base = [rng.uniform(0.2, 0.6) for _ in range(30)]
    better = [b + 0.1 + rng.uniform(0, 0.05) for b in base]
    r = compare(_pq(base), _pq(better), "nDCG@10", permutations=10_000, seed=0)
    assert r["p_randomization"] < 0.01 and r["p_ttest"] < 0.01
    assert r["better"] == 30 and r["worse"] == 0 and r["mean_diff"] > 0.1


def test_reproducible_with_seed():
    rng = random.Random(3)
    a = [rng.random() for _ in range(25)]
    b = [x + rng.uniform(-0.1, 0.12) for x in a]
    p1 = randomization_test(b, a, 5000, seed=42)
    assert p1 == randomization_test(b, a, 5000, seed=42)
    assert 0 < p1 <= 1
    assert randomization_test(b, a, 5000) == randomization_test(b, a, 5000)  # config seed


def test_holm_hand_example():
    # sorted: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; max(0.06, 0.04 * 1) = 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_compare_many_adds_holm():
    rng = random.Random(4)
    base = [rng.random() for _ in range(20)]
    systems = {"a": _pq([x + 0.2 for x in base]), "b": _pq(base)}
    rows = compare_many(_pq(base), systems, "nDCG@10", permutations=2000, seed=0)
    assert rows["a"]["p_randomization_holm"] >= rows["a"]["p_randomization"]
    assert rows["b"]["p_randomization_holm"] == pytest.approx(1.0)


def test_t_test_degenerate():
    assert t_test([1.0], [0.0]) == (0.0, 1.0)


# ------------------------------------------------------------------ agreement


def test_kappa_perfect_and_chance():
    a = [1, 0, 1, 1, 0]
    assert cohen_kappa(a, a) == 1.0
    x = [1, 0] * 50
    y = [1, 1, 0, 0] * 25
    assert cohen_kappa(x, y) == pytest.approx(0.0, abs=1e-12)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0


def test_kappa_textbook_example():
    # Manning et al., IIR table 8.2: 400 docs; yes/yes 300, yes/no 20, no/yes 10, no/no 70.
    # P(A) = 370/400 = 0.925; P(E) = 0.8 * 0.775 + 0.2 * 0.225 = 0.665
    # kappa = (0.925 - 0.665) / (1 - 0.665) = 0.776119
    a = [1] * 300 + [1] * 20 + [0] * 10 + [0] * 70
    b = [1] * 300 + [0] * 20 + [1] * 10 + [0] * 70
    assert cohen_kappa(a, b) == pytest.approx(0.776119, abs=1e-6)


def test_weighted_kappa():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    # with two categories linear weighted kappa = Cohen's kappa:
    # p_o = 3/4, p_e = .5 * .25 + .5 * .75 = .5 -> 0.5
    assert weighted_kappa([0, 0, 1, 1], [0, 1, 1, 1], categories=(0, 1)) == pytest.approx(0.5)
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)
    # off by one costs less than off by three
    near = weighted_kappa([0, 1, 2, 3, 3, 0], [1, 1, 2, 3, 2, 0])
    far = weighted_kappa([0, 1, 2, 3, 3, 0], [3, 1, 2, 3, 0, 0])
    assert near > far
    with pytest.raises(ValueError):
        weighted_kappa([0, 4], [0, 1])


def test_weighted_kappa_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = random.Random(5)
    a = [rng.choice(range(4)) for _ in range(200)]
    b = [x if rng.random() < 0.6 else rng.choice(range(4)) for x in a]
    assert weighted_kappa(a, b) == pytest.approx(
        metrics.cohen_kappa_score(a, b, weights="linear", labels=[0, 1, 2, 3])
    )


def test_agreement_report(fixtures):
    human = {"q01": {"A": 3, "B": 0, "C": 2}, "q02": {"A": 1}}
    llm = {"q01": {"A": 2, "B": 0, "C": 1, "D": 3}, "q02": {"A": 1}}
    rep = agreement_report(human, llm)
    assert rep["n_shared"] == 4 and rep["n_only_a"] == 0 and rep["n_only_b"] == 1
    assert rep["exact_agreement"] == pytest.approx(0.5)
    # binary (>= 2): human 1 0 1 0, llm 1 0 0 0 -> 3/4 agree
    assert rep["binary_agreement"] == pytest.approx(0.75)
    assert rep["confusion"][3][2] == 1 and rep["confusion"][2][1] == 1
    assert (
        agreement_report(fixtures / "sample_qrels.txt", fixtures / "sample_qrels.txt")[
            "kappa_weighted"
        ]
        == 1.0
    )
