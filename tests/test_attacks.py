"""T20: attack simulator."""

import copy

import pytest

from phonesense import io
from phonesense.attacks.simulator import (
    attack_size,
    inject,
    measure,
    pick_targets,
    rank_of,
)
from phonesense.schemas import Review
from phonesense.trust.baselines import bayesian_average, raw_average
from phonesense.trust.toy import make_toy_reviews
from phonesense.trust.trust import compute_trust

TARGET = "B0TOY00003"


@pytest.fixture
def toy():
    return make_toy_reviews()


def fakes(reviews, attacked):
    return attacked[len(reviews):]


def test_returns_new_list_and_leaves_input_alone(toy):
    before = copy.deepcopy(toy)
    out = inject(toy, TARGET, "bad_mouth", 7, seed=1)
    assert toy == before and out is not toy
    assert len(out) == len(toy) + 7 and out[:len(toy)] == toy


def test_fake_reviews_follow_the_contract(toy, tmp_path):
    out = inject(toy, TARGET, "ballot_stuff", 3, seed=1, camouflage=2)
    fake = fakes(toy, out)
    assert all(f.is_synthetic and f.user_id.startswith("SYN_") and not f.verified for f in fake)
    assert all(Review.from_dict(f.to_dict()) == f for f in fake)  # valid review_id etc.
    assert len({f.review_id for f in out}) == len(out)
    path = io.write_reviews(tmp_path / "reviews_attack.jsonl", out)
    assert io.read_reviews(path, allow_synthetic=True) == out
    with pytest.raises(ValueError):
        io.read_reviews(path)  # the real-data reader refuses synthetic reviews


def test_one_new_account_per_attack_review(toy):
    fake = fakes(toy, inject(toy, TARGET, "bad_mouth", 3, seed=1))
    assert [f.rating for f in fake] == [1, 1, 1] and {f.doc_id for f in fake} == {TARGET}
    assert len({f.user_id for f in fake}) == 3


def test_seeds(toy):
    a = inject(toy, TARGET, "bad_mouth", 4, seed=1, camouflage=3)
    assert a == inject(toy, TARGET, "bad_mouth", 4, seed=1, camouflage=3)
    b = inject(toy, TARGET, "bad_mouth", 4, seed=2, camouflage=3)
    users = {f.user_id for f in fakes(toy, a)}
    assert users.isdisjoint(f.user_id for f in fakes(toy, b))


def test_direction_of_attack(toy):
    raw = raw_average(toy)
    assert raw_average(inject(toy, TARGET, "bad_mouth", 10, seed=1))[TARGET] < raw[TARGET]
    assert raw_average(inject(toy, TARGET, "ballot_stuff", 10, seed=1))[TARGET] > raw[TARGET]


def test_zero_and_invalid_inputs(toy):
    assert inject(toy, TARGET, "bad_mouth", 0, seed=1) == toy
    with pytest.raises(ValueError):
        inject(toy, TARGET, "praise", 1, seed=1)
    with pytest.raises(ValueError):
        inject(toy, "B0NOTAPHONE", "bad_mouth", 1, seed=1)
    with pytest.raises(ValueError):
        inject(toy, TARGET, "bad_mouth", -1, seed=1)


def test_camouflage_copies_consensus_and_buys_reliability(toy):
    out = inject(toy, TARGET, "bad_mouth", 5, seed=1, camouflage=4)
    fake = fakes(toy, out)
    assert len(fake) == 5 * (4 + 1)
    raw = raw_average(toy)
    cover = [f for f in fake if f.doc_id != TARGET]
    assert len(cover) == 20 and all(f.rating == round(raw[f.doc_id]) for f in cover)
    sybil = fake[-1].user_id
    plain = compute_trust(inject(toy, TARGET, "bad_mouth", 5, seed=1)).reliability[sybil]
    assert compute_trust(out).reliability[sybil] > plain


def test_attack_size(toy):
    n = sum(r.doc_id == TARGET for r in toy)
    assert attack_size(toy, TARGET, 0) == 0
    assert attack_size(toy, TARGET, 100) == n
    assert attack_size(toy, TARGET, 1) == 1  # rounds up
    assert attack_size(toy, "B0NOTAPHONE", 50) == 0


def test_pick_targets(toy):
    raw = raw_average(toy)
    good = pick_targets(toy, "bad_mouth", 3, seed=1)
    weak = pick_targets(toy, "ballot_stuff", 3, seed=1)
    assert good == pick_targets(toy, "bad_mouth", 3, seed=1)
    assert len(set(good)) == 3
    assert min(raw[d] for d in good) > max(raw[d] for d in weak)
    with pytest.raises(ValueError):
        pick_targets(toy, "bad_mouth", 3, seed=1, min_reviews=10_000)


def test_rank_of_breaks_ties_by_doc_id():
    scores = {"b": 1.0, "a": 1.0, "c": 2.0}
    assert [rank_of(scores, d) for d in "cab"] == [1, 2, 3]


def test_measure(toy):
    target = pick_targets(toy, "bad_mouth", 1, seed=1)[0]
    attacked = inject(toy, target, "bad_mouth", attack_size(toy, target, 20), seed=1)
    methods = {"raw": raw_average, "bayesian": bayesian_average,
               "trust": lambda rs: compute_trust(rs).quality}
    m = measure(toy, attacked, target, methods)
    assert set(m) == set(methods)
    for row in m.values():
        assert row["score_after"] < row["score_before"]
        assert row["rank_after"] >= row["rank_before"]
    before = {k: fn(toy) for k, fn in methods.items()}
    assert measure(toy, attacked, target, methods, before=before) == m
