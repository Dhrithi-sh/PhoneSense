from dhvani.rank.collapse import cluster_id, clusters, collapse_duplicates, collapse_pool
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import search


def test_copy_and_original_share_a_cluster():
    idx = SampleIndex.load()
    assert cluster_id("amarujala_3004", idx) == "aajtak_5004"
    assert cluster_id("aajtak_5004", idx) == "aajtak_5004"
    assert sorted(clusters(idx)["aajtak_5004"]) == ["aajtak_5004", "amarujala_3004"]


def test_wire_story_shows_once_with_also_in():
    idx = SampleIndex.load()
    results = search(exact_query("रेल लाइन उद्घाटन"), idx, k=10)
    ids = [d for d, _, _ in results]
    assert "aajtak_5004" in ids and "amarujala_3004" in ids     # both copies before collapsing
    collapsed = collapse_duplicates(results, idx)
    kept = [d for d, _, _ in collapsed]
    assert len({"aajtak_5004", "amarujala_3004"} & set(kept)) == 1
    top = next(e for d, _, e in collapsed if d in {"aajtak_5004", "amarujala_3004"})
    assert len(top["also_in"]) == 1 and len(top["duplicates"]) == 1


def test_best_ranked_copy_is_the_one_kept():
    idx = SampleIndex.load()
    results = search(exact_query("रेल लाइन उद्घाटन"), idx, k=10)
    first_of_cluster = next(d for d, _, _ in results if d in {"aajtak_5004", "amarujala_3004"})
    assert first_of_cluster in [d for d, _, _ in collapse_duplicates(results, idx)]


def test_articles_without_copies_are_untouched():
    idx = SampleIndex.load()
    results = search(exact_query("कोहली शतक"), idx, k=5)
    collapsed = collapse_duplicates(results, idx)
    assert [d for d, _, _ in collapsed] == [d for d, _, _ in results]
    assert all(e["also_in"] == [] for _, _, e in collapsed)


def test_cut_to_k_after_collapsing():
    idx = SampleIndex.load()
    results = search(exact_query("उद्घाटन रेल"), idx, k=collapse_pool(2))
    assert len(collapse_duplicates(results, idx, k=2)) <= 2


def test_collapse_keeps_term_scores_separate_for_plain_rankers():
    idx = SampleIndex.load()
    _, _, explain = collapse_duplicates(search(exact_query("रेल लाइन"), idx, k=5), idx)[0]
    assert "also_in" in explain and "also_in" not in explain["terms"]
