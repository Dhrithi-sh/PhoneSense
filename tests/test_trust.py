"""T19: properties of the reviewer-trust score."""

import copy
import math
from collections import Counter

import pytest

from phonesense.schemas import Review, make_review_id
from phonesense.trust.baselines import bayesian_average, raw_average
from phonesense.trust.toy import make_toy_reviews
from phonesense.trust.trust import compute_trust, trust_scores

TS = 1_700_000_000_000


def rev(doc_id: str, user_id: str, rating: int) -> Review:
    return Review(make_review_id(user_id, doc_id, TS), doc_id, user_id, rating, "", "", TS, 0,
                  False)


def bad_mouth(reviews, target, n):
    return reviews + [rev(target, f"sybil{i}", 1) for i in range(n)]


@pytest.fixture
def toy():
    return make_toy_reviews()


def test_first_iteration_is_bayesian_average(toy):
    res = compute_trust(toy, max_iterations=1)
    bay = bayesian_average(toy, prior_weight=10)
    assert all(math.isclose(res.quality[d], bay[d], abs_tol=1e-9) for d in bay)


def test_infinite_reviewer_prior_reduces_to_bayesian(toy):
    res = compute_trust(toy, prior_strength=1e12, prior_reliability=1.0)
    bay = bayesian_average(toy, prior_weight=10)
    assert all(math.isclose(res.quality[d], bay[d], abs_tol=1e-6) for d in bay)


def test_converges_and_stays_in_range(toy):
    res = compute_trust(toy)
    assert res.converged and res.iterations < 50
    assert all(1.0 <= q <= 5.0 for q in res.quality.values())
    assert all(0.0 <= r <= 1.0 for r in res.reliability.values())
    unit = trust_scores(toy)
    assert set(unit) == {r.doc_id for r in toy}
    assert all(0.0 <= v <= 1.0 for v in unit.values())


def test_contrarian_is_less_reliable_than_agreer(toy):
    raw = raw_average(toy)
    docs = sorted(raw)[:8]
    extra = [rev(d, "contrarian", 1 if raw[d] >= 3 else 5) for d in docs]
    extra += [rev(d, "agreer", round(raw[d])) for d in docs]
    rel = compute_trust(toy + extra).reliability
    assert rel["contrarian"] < 0.5 < rel["agreer"]


def test_new_account_sits_just_below_the_prior(toy):
    raw = raw_average(toy)
    target = max(raw, key=raw.get)
    rel = compute_trust(bad_mouth(toy, target, 1)).reliability["sybil0"]
    assert 0.25 < rel < 0.5  # one disagreeing review: below the 0.5 prior, not zero


def test_no_mutation_and_deterministic(toy):
    before = copy.deepcopy(toy)
    a = trust_scores(toy)
    assert toy == before
    assert a == trust_scores(toy)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_bad_mouthing_hurts_trust_less_than_raw_and_bayesian(toy, n):
    counts = Counter(r.doc_id for r in toy)
    raw0 = raw_average(toy)
    target = max((d for d in raw0 if counts[d] >= 20), key=raw0.get)
    attacked = bad_mouth(toy, target, n)
    drop_raw = raw0[target] - raw_average(attacked)[target]
    drop_bay = bayesian_average(toy)[target] - bayesian_average(attacked)[target]
    drop_trust = compute_trust(toy).quality[target] - compute_trust(attacked).quality[target]
    assert 0 < drop_trust < drop_bay < drop_raw


def test_empty_input_raises():
    with pytest.raises(ValueError):
        compute_trust([])
