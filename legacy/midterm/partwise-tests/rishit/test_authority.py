from dhvani.rank.authority import link_graph, originals, pagerank, static_scores
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.scoring import rank
from dhvani.rank.speedups import ChampionLists


def test_pagerank_sums_to_one():
    pr = pagerank(link_graph(SampleIndex.load()))
    assert abs(sum(pr.values()) - 1.0) < 1e-9


def test_more_incoming_links_means_higher_pagerank():
    # a and b both link to c; nothing links to a or b.
    pr = pagerank({"a": ["c"], "b": ["c"], "c": []})
    assert pr["c"] > pr["a"] and abs(pr["a"] - pr["b"]) < 1e-12


def test_dangling_articles_keep_the_total_at_one():
    pr = pagerank({"a": [], "b": [], "c": ["a"]})
    assert abs(sum(pr.values()) - 1.0) < 1e-9
    assert pr["a"] > pr["b"]


def test_two_node_cycle_is_even():
    pr = pagerank({"a": ["b"], "b": ["a"]})
    assert abs(pr["a"] - 0.5) < 1e-9


def test_links_outside_the_index_and_self_links_are_ignored():
    idx = SampleIndex.load()
    idx.meta["jagran_1001"]["links"] = ["nbt_2001", "not_crawled", "jagran_1001"]
    assert link_graph(idx)["jagran_1001"] == ["nbt_2001"]


def test_most_linked_sample_article_ranks_first():
    pr = pagerank(link_graph(SampleIndex.load()))
    assert max(pr, key=pr.get) == "nbt_2001"   # linked from jagran_1001 and aajtak_5001


def test_first_to_publish_is_the_original():
    idx = SampleIndex.load()
    assert originals(idx) == {"aajtak_5004"}   # amarujala_3004 has dup_of = aajtak_5004


def test_static_score_is_in_range_and_built_from_parts():
    idx = SampleIndex.load()
    g, parts = static_scores(idx)
    assert set(g) == set(idx.meta)
    assert all(0.0 <= v <= 1.0 for v in g.values())
    p = parts["aajtak_5004"]
    assert abs(g["aajtak_5004"] - (0.5 * p["recency"] + 0.3 * p["pagerank"] + 0.2 * p["original"])) < 1e-12


def test_original_now_beats_its_newer_copy():
    idx = SampleIndex.load()
    q = exact_query("रेल लाइन उद्घाटन")
    only_recency = [d for d, _, _ in rank(q, idx, k=2)]
    g, _ = static_scores(idx)
    with_authority = [d for d, _, _ in rank(q, idx, k=2, static=g)]
    assert only_recency[0] == "amarujala_3004"      # the newer copy used to win
    assert with_authority[0] == "aajtak_5004"       # the original wins with authority


def test_without_static_the_net_score_is_unchanged():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    assert rank(q, idx, k=5) == rank(q, idx, k=5, static=None)


def test_champion_lists_accept_the_static_score():
    idx = SampleIndex.load()
    g, _ = static_scores(idx)
    assert ChampionLists(idx, r=2, static_scores=g).champions("बारिश")
