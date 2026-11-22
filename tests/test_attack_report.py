"""T25: the attack report behind the demo's attack tab."""

import pytest

from phonesense.attacks.report import attack_report, clean_scores
from phonesense.attacks.simulator import pick_targets
from phonesense.trust.toy import make_toy_reviews

METHODS = ["raw", "bayesian", "helpful", "median", "trust"]


@pytest.fixture(scope="module")
def toy():
    return make_toy_reviews()


@pytest.fixture(scope="module")
def before(toy):
    return clean_scores(toy)


def by_method(report):
    return {r["method"]: r for r in report.rows}


def test_zero_attack_changes_nothing(toy, before):
    target = pick_targets(toy, "bad_mouth", 1, seed=1)[0]
    rep = attack_report(toy, target, "bad_mouth", 0, before=before)
    assert rep.n_fake == 0 and rep.n_cover == 0 and rep.sybil_reliability is None
    assert [r["method"] for r in rep.rows] == METHODS
    assert all(r["rank_gain"] == 0 and r["score_gain"] == pytest.approx(0) for r in rep.rows)


@pytest.mark.parametrize("kind", ["bad_mouth", "ballot_stuff"])
def test_attack_gains_are_positive_for_the_attacker(toy, before, kind):
    target = pick_targets(toy, kind, 1, seed=1)[0]
    rows = by_method(attack_report(toy, target, kind, 50, before=before))
    assert rows["raw"]["rank_gain"] > 0 and rows["raw"]["score_gain"] > 0
    assert rows["trust"]["score_gain"] < rows["raw"]["score_gain"]


def test_camouflage_raises_sybil_reliability(toy, before):
    target = pick_targets(toy, "bad_mouth", 1, seed=1)[0]
    plain = attack_report(toy, target, "bad_mouth", 20, before=before)
    camo = attack_report(toy, target, "bad_mouth", 20, camouflage=4, before=before)
    assert camo.n_fake == plain.n_fake and camo.n_cover == 4 * plain.n_fake
    assert camo.sybil_reliability > plain.sybil_reliability


def test_leaderboards_and_before_scores(toy, before):
    target = pick_targets(toy, "ballot_stuff", 1, seed=1)[0]
    rep = attack_report(toy, target, "ballot_stuff", 100, top_n=5)
    assert set(rep.leaderboards) == set(METHODS)
    assert all(len(board) == 5 for board in rep.leaderboards.values())
    for board in rep.leaderboards.values():
        scores = [s for _, s in board]
        assert scores == sorted(scores, reverse=True)
    assert rep.rows == attack_report(toy, target, "ballot_stuff", 100, top_n=5,
                                     before=before).rows
