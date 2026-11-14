from dhvani.rank.feedback import article_vector, feedback_query
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex


def test_article_vectors_weight_rare_words_above_common_ones():
    idx = SampleIndex.load()
    vec = article_vector(idx, "jagran_1001")
    common = [t for t in vec if idx.df(t) > idx.N // 2]
    rare = [t for t in vec if idx.df(t) == 1]
    assert common and rare
    assert max(vec[t] for t in common) < max(vec[t] for t in rare)


def test_feedback_adds_prf_terms_and_leaves_the_input_alone():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    before = [dict(t, expansions=list(t["expansions"])) for t in q["tokens"]]
    expanded, feedback = feedback_query(q, idx, docs=2, terms=3)
    assert feedback and len(feedback) <= 2
    added = expanded["tokens"][len(q["tokens"]):]
    assert 0 < len(added) <= 3 and all(t["expansions"][0][2] == "prf" for t in added)
    assert q["tokens"] == before


def test_no_results_means_no_feedback():
    idx = SampleIndex.load()
    q = exact_query("zzzz")
    assert feedback_query(q, idx) == (q, [])
