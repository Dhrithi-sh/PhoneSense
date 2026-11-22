"""T19: reputation baselines on hand-computed cases."""

import math

import pytest

from phonesense.schemas import Review, make_review_id
from phonesense.trust.baselines import (
    bayesian_average,
    global_mean,
    helpful_weighted_average,
    median_rating,
    raw_average,
    to_unit,
)
from phonesense.trust.toy import make_toy_reviews

TS = 1_700_000_000_000


def rev(doc_id: str, user_id: str, rating: int, helpful: int = 0) -> Review:
    return Review(make_review_id(user_id, doc_id, TS), doc_id, user_id, rating, "", "", TS,
                  helpful, True)


HAND = [rev("a", "u1", 5), rev("a", "u2", 4), rev("a", "u3", 1, helpful=9), rev("b", "u1", 2)]


def test_hand_computed():
    assert raw_average(HAND) == {"a": 10 / 3, "b": 2.0}
    assert median_rating(HAND) == {"a": 4.0, "b": 2.0}
    m = global_mean(HAND)
    assert m == 3.0
    assert math.isclose(bayesian_average(HAND, prior_weight=10)["a"], (10 * m + 10) / 13)
    w3 = 1 + math.log(10)
    assert math.isclose(helpful_weighted_average(HAND)["a"], (5 + 4 + w3 * 1) / (2 + w3))


def test_identical_ratings_agree_everywhere():
    rs = [rev("a", f"u{i}", 4) for i in range(5)]
    for fn in (raw_average, bayesian_average, helpful_weighted_average, median_rating):
        assert math.isclose(fn(rs)["a"], 4.0)


def test_bayesian_pulls_small_phones_to_the_mean():
    reviews = make_toy_reviews() + [rev("lonely", "ux", 5)]
    assert raw_average(reviews)["lonely"] == 5.0
    assert abs(bayesian_average(reviews)["lonely"] - global_mean(reviews)) < 0.2


def test_to_unit_and_empty_input():
    assert to_unit({"a": 1.0, "b": 3.0, "c": 5.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}
    with pytest.raises(ValueError):
        global_mean([])
