from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.speedups import (
    default_min_match,
    high_idf_terms,
    overlap_at_k,
    search_index_elimination,
    soft_and_candidates,
)
from dhvani.rank.vsm import query_vector, search


# --- index elimination -------------------------------------------------------

def test_low_idf_words_are_dropped():
    idx = SampleIndex.load()
    qvec = query_vector(exact_query("बारिश में अलर्ट"), idx)
    kept, idf = high_idf_terms(qvec, idx, min_idf=0.3)
    # में is in 15 of 20 articles (idf 0.125), so it's skipped.
    assert "में" not in kept
    assert {"बारिश", "अलर्ट"} <= set(kept)


def test_rarest_term_is_kept_even_if_all_are_common():
    idx = SampleIndex.load()
    qvec = query_vector(exact_query("में ने"), idx)
    kept, idf = high_idf_terms(qvec, idx, min_idf=5.0)
    assert kept == [max(qvec, key=lambda t: idf[t])]


def test_soft_and_needs_enough_terms():
    idx = SampleIndex.load()
    both = soft_and_candidates(idx, ["दिल्ली", "बारिश"], 2)
    either = soft_and_candidates(idx, ["दिल्ली", "बारिश"], 1)
    assert both < either
    assert "jagran_1001" in both and "amarujala_3001" not in both


def test_min_match_follows_the_lecture_example():
    assert default_min_match(4) == 3
    assert default_min_match(2) == 1
    assert default_min_match(1) == 1


def test_scores_fewer_articles_than_full_search():
    idx = SampleIndex.load()
    results, stats = search_index_elimination(exact_query("दिल्ली में बारिश का अलर्ट"), idx, k=3)
    assert stats["scored"] < stats["full_candidates"]
    assert set(stats["terms_dropped"]) >= {"में"}
    assert results[0][0] == "jagran_1001"


def test_top_k_stays_close_to_exact():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली में बारिश का अलर्ट")
    fast, _ = search_index_elimination(q, idx, k=3)
    assert overlap_at_k(fast, search(q, idx, k=3), k=3) >= 2 / 3


def test_relaxes_when_too_strict():
    idx = SampleIndex.load()
    # Nothing contains both कोहली and बिहार, but the search still returns k.
    results, stats = search_index_elimination(exact_query("कोहली बिहार"), idx, k=2, min_match=2)
    assert len(results) == 2
    assert stats["min_match"] == 1


def test_unknown_words_give_nothing():
    results, stats = search_index_elimination(exact_query("xyzabc"), SampleIndex.load())
    assert results == [] and stats["scored"] == 0


def test_overlap_at_k():
    a = [("x", 1, {}), ("y", 1, {}), ("z", 1, {})]
    b = [("x", 1, {}), ("q", 1, {}), ("z", 1, {})]
    assert overlap_at_k(a, b, k=3) == 2 / 3
    assert overlap_at_k(a, [], k=3) == 1.0


# --- champion lists ----------------------------------------------------------

from dhvani.rank.speedups import ChampionLists, search_champions  # noqa: E402


def test_champion_list_is_at_most_r_and_ordered_by_weight():
    idx = SampleIndex.load()
    champs = ChampionLists(idx, r=2)
    lst = champs.champions("बारिश")
    assert len(lst) == 2
    from dhvani.rank.speedups import _doc_tf
    from dhvani.rank.vsm import log_tf
    weight = {d: log_tf(tf) / idx.doc_norm[d] for d, tf in _doc_tf(idx, "बारिश").items()}
    assert weight[lst[0]] >= weight[lst[1]]
    assert all(weight[d] <= weight[lst[-1]] for d in weight if d not in lst)


def test_big_r_gives_the_exact_ranking():
    idx = SampleIndex.load()
    q = exact_query("मौसम बारिश")
    fast, stats = search_champions(q, idx, ChampionLists(idx, r=1000), k=5)
    exact = search(q, idx, k=5)
    assert [d for d, _, _ in fast] == [d for d, _, _ in exact]
    assert not stats["fell_back"]


def test_small_r_scores_fewer_articles():
    idx = SampleIndex.load()
    q = exact_query("मौसम बारिश दिल्ली")
    fast, stats = search_champions(q, idx, ChampionLists(idx, r=2), k=3)
    assert stats["scored"] < stats["full_candidates"]
    assert overlap_at_k(fast, search(q, idx, k=3), k=3) >= 2 / 3


def test_falls_back_when_champions_are_too_few():
    idx = SampleIndex.load()
    q = exact_query("मौसम")
    results, stats = search_champions(q, idx, ChampionLists(idx, r=1), k=4)
    assert stats["fell_back"]
    assert len(results) == 4


def test_static_scores_pull_an_article_into_the_list():
    idx = SampleIndex.load()
    plain = ChampionLists(idx, r=1).champions("बारिश")
    loser = next(d for d, _, _ in idx.postings("बारिश", "body") if d not in plain)
    boosted = ChampionLists(idx, r=1, static_scores={loser: 10.0}).champions("बारिश")
    assert boosted == [loser]


# --- recent-news tiers -------------------------------------------------------

from dhvani.rank.speedups import RecencyTiers, search_tiered  # noqa: E402


def test_newest_articles_land_in_tier_0():
    idx = SampleIndex.load()
    tiers = RecencyTiers(idx, tier_days=(1, 2, None))
    # Newest article is 5 Oct 21:30; jagran_1001 (5 Oct 08:10) is under a day old,
    # livehindustan_4001 (3 Oct 07:45) is more than 2 days old.
    assert tiers.tier_of["jagran_1001"] == 0
    assert tiers.tier_of["livehindustan_4001"] == 2
    assert sum(tiers.size(t) for t in range(3)) == idx.N


def test_stops_at_the_first_tier_with_enough_results():
    idx = SampleIndex.load()
    tiers = RecencyTiers(idx, tier_days=(1, 2, None))
    results, stats = search_tiered(exact_query("बारिश"), idx, tiers, k=1)
    assert stats["tiers_used"] == [0]
    assert stats["scored"] < stats["full_candidates"]
    assert tiers.tier_of[results[0][0]] == 0


def test_drops_to_older_tiers_when_needed():
    idx = SampleIndex.load()
    tiers = RecencyTiers(idx, tier_days=(1, 2, None))
    results, stats = search_tiered(exact_query("बारिश"), idx, tiers, k=5)
    assert stats["tiers_used"] == [0, 1, 2]
    assert len(results) == 5


def test_all_tiers_give_the_exact_ranking():
    idx = SampleIndex.load()
    q = exact_query("मौसम बारिश")
    tiers = RecencyTiers(idx, tier_days=(1, 2, None))
    fast, _ = search_tiered(q, idx, tiers, k=100)
    exact = search(q, idx, k=100)
    assert [d for d, _, _ in fast] == [d for d, _, _ in exact]


# --- cluster pruning ---------------------------------------------------------

from dhvani.rank.speedups import ClusterPruning, search_clusters  # noqa: E402


def test_about_sqrt_n_leaders_and_every_article_in_one_cluster():
    idx = SampleIndex.load()
    cp = ClusterPruning(idx)
    assert len(cp.leaders) == round(idx.N ** 0.5)          # 20 articles -> 4 leaders
    members = [d for m in cp.members.values() for d in m]
    assert sorted(members) == sorted(idx.meta)             # each article exactly once
    assert all(cp.leader_of[l] == l for l in cp.leaders)


def test_followers_join_their_most_similar_leader():
    idx = SampleIndex.load()
    cp = ClusterPruning(idx)
    for doc_id, leader in cp.leader_of.items():
        if doc_id in cp.leaders:
            continue
        sims = {l: sum(w * cp.vectors[l].get(t, 0.0) for t, w in cp.vectors[doc_id].items()) for l in cp.leaders}
        assert sims[leader] == max(sims.values())


def test_same_seed_same_clusters():
    idx = SampleIndex.load()
    assert ClusterPruning(idx, seed=3).leaders == ClusterPruning(idx, seed=3).leaders


def test_all_leaders_gives_the_exact_ranking():
    idx = SampleIndex.load()
    cp = ClusterPruning(idx)
    q = exact_query("दिल्ली बारिश")
    fast, stats = search_clusters(q, idx, cp, k=5, b=len(cp.leaders))
    assert overlap_at_k(fast, search(q, idx, k=5), k=5) == 1.0
    assert stats["leaders_used"] == len(cp.leaders)


def test_cluster_search_scores_at_most_the_full_candidates():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    _results, stats = search_clusters(q, idx, ClusterPruning(idx), k=2, b=1)
    assert stats["method"] == "cluster pruning"
    assert 0 < stats["scored"] <= stats["full_candidates"]
    assert 1 <= stats["leaders_used"] <= stats["leaders"]


# --- impact-ordered postings -------------------------------------------------

from dhvani.rank.speedups import ImpactOrdered, search_impact  # noqa: E402


def test_impact_lists_are_sorted_by_weight():
    idx = SampleIndex.load()
    imp = ImpactOrdered(idx)
    for term in ("बारिश", "दिल्ली"):
        weights = [w for _d, w in imp.postings(term)]
        assert weights == sorted(weights, reverse=True)
        assert {d for d, _w in imp.postings(term)} == {d for z in ("headline", "body") for d, _tf, _p in idx.postings(term, z)}


def test_reading_everything_gives_the_exact_ranking():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    fast, stats = search_impact(q, idx, ImpactOrdered(idx), k=5, max_docs=None, min_share=0.0)
    exact = search(q, idx, k=5)
    assert [d for d, _, _ in fast] == [d for d, _, _ in exact]
    assert stats["postings_read"] == stats["postings_total"]


def test_max_docs_stops_each_list_early():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    _results, stats = search_impact(q, idx, ImpactOrdered(idx), k=5, max_docs=1)
    assert stats["postings_read"] <= len(query_vector(q, idx))
    assert stats["postings_read"] < stats["postings_total"]


def test_weight_floor_stops_reading_low_weight_articles():
    idx = SampleIndex.load()
    imp = ImpactOrdered(idx)
    q = exact_query("बारिश")
    _results, stats = search_impact(q, idx, imp, k=5, max_docs=None, min_share=0.99)
    best = imp.postings("बारिश")[0][1]
    assert stats["postings_read"] == sum(1 for _d, w in imp.postings("बारिश") if w >= 0.99 * best)
