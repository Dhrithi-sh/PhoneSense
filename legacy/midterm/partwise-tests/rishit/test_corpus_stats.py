from dhvani.eval.corpus_stats import stop_words, stopword_experiment, term_stats, top_terms, zipf_fit
from dhvani.eval.experiments import SAMPLE_DIR, read_queries
from dhvani.eval.metrics import read_qrels
from dhvani.rank.sample_index import SampleIndex


def sample():
    return read_queries(SAMPLE_DIR / "queries.tsv"), read_qrels(SAMPLE_DIR / "qrels.txt")


# --- stop words, idf, Zipf ---------------------------------------------------

def test_term_stats_match_the_index():
    idx = SampleIndex.load()
    stats = term_stats(idx)
    assert set(stats) == set(idx.vocab)
    assert stats["मौसम"]["df"] == idx.df("मौसम")
    assert all(s["cf"] >= s["df"] for s in stats.values())


def test_function_words_come_out_as_stop_words():
    stop = stop_words(term_stats(SampleIndex.load()), n=8)
    assert {"में", "का", "की", "के"} <= set(stop)


def test_frequent_terms_have_the_lowest_idf():
    rows = top_terms(term_stats(SampleIndex.load()), n=5)
    idfs = [idf for *_, idf in rows]
    assert idfs == sorted(idfs)
    assert rows[0][0] == "में"


def test_zipf_slope_is_negative():
    slope, _ = zipf_fit(term_stats(SampleIndex.load()))
    assert slope < 0


def test_stopword_experiment_runs_all_three_setups():
    queries, qrels = sample()
    idx = SampleIndex.load()
    out = stopword_experiment(idx, queries, qrels, stop_words(term_stats(idx), 8))
    assert set(out) == {"no idf (lnc.lnc)", "idf (lnc.ltc)", "stop words removed"}
    assert all("MAP" in m for m in out.values())
