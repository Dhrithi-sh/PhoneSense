"""UI engine: toggles, fallbacks and notes. No Streamlit, no network, no model download."""

import importlib.util

import pytest

from phonesense.io import read_phones
from phonesense.llm.client import LLMClient, LLMError
from phonesense.retrieval.hyde import HydeGenerator, HydeRetriever
from phonesense.ui.engine import NoDataError, SearchEngine, find_phones_file

HAS_DENSE = importlib.util.find_spec("phonesense.retrieval.dense") is not None
HAS_FUSION = importlib.util.find_spec("phonesense.retrieval.fusion") is not None


@pytest.fixture
def engine(fixtures):
    return SearchEngine(read_phones(fixtures / "phones.jsonl"))


def ids(response):
    return [r.doc_id for r in response.results]


def use_fake_llm(engine, tmp_path, text="compact flagship excellent camera", fail=False):
    def backend(prompt, model, temperature):
        if fail:
            raise LLMError("GEMINI_API_KEY is not set")
        return text

    client = LLMClient("gemini", "m", backend=backend, log_dir=tmp_path)
    engine._hyde = HydeRetriever(HydeGenerator(client), engine.tfidf)


def test_default_is_tfidf_and_results_are_displayable(engine):
    r = engine.search("compact camera")
    assert r.used == ["tfidf"] and r.notes == []
    top = r.results[0]
    assert top.rank == 1 and top.title and top.score > 0
    assert "USD" in top.price_text and "ratings" in top.rating_text


def test_empty_query_gives_no_results(engine):
    assert engine.search("   ").results == []


def test_no_system_selected_falls_back_to_tfidf_with_a_note(engine):
    r = engine.search("compact camera", {"tfidf": False, "rocchio": False})
    assert r.used == ["tfidf"] and any("No system selected" in n for n in r.notes)


def test_rocchio_toggle_runs_rocchio(engine):
    r = engine.search("compact camera", {"rocchio": True})
    assert r.used == ["rocchio"] and r.results


def test_k_limits_the_results(engine):
    # "camera" and "battery" occur in different phones, so there are at least two matches
    assert len(engine.search("camera battery", {"tfidf": True}, k=5).results) >= 2
    assert len(engine.search("camera battery", {"tfidf": True}, k=1).results) == 1


def test_hyde_toggle_uses_the_llm(engine, tmp_path):
    use_fake_llm(engine, tmp_path)
    r = engine.search("compact camera", {"hyde": True})
    assert r.used == ["hyde"] and r.results and r.notes == []


def test_hyde_without_a_key_is_skipped_with_a_note_and_retried_later(engine, tmp_path):
    use_fake_llm(engine, tmp_path, fail=True)
    r = engine.search("compact camera", {"hyde": True})
    assert r.used == ["tfidf"] and any("HyDE" in n and "GEMINI_API_KEY" in n for n in r.notes)
    assert engine.unavailable_reason("hyde") is None  # LLM errors may be transient: not cached


def test_two_systems_without_fusion_use_the_first_and_say_so(engine):
    r = engine.search("compact camera", {"tfidf": True, "rocchio": True, "fusion": False})
    # the reason is "fusion is off", or "not in this build" on a branch without the fusion code
    assert r.used == ["tfidf"] and any("2 systems selected" in n for n in r.notes)
    if HAS_FUSION:
        assert any("fusion is off" in n for n in r.notes)


def test_bad_fusion_method_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.search("x", {"tfidf": True}, fusion_method="borda")


@pytest.mark.skipif(not HAS_FUSION, reason="fusion module not in this build")
@pytest.mark.parametrize("method", ["rrf", "linear"])
def test_fusion_merges_the_selected_systems(engine, method):
    toggles = {"tfidf": True, "rocchio": True, "fusion": True}
    r = engine.search("compact camera", toggles, fusion_method=method)
    assert r.used == ["tfidf", "rocchio"] and r.notes == []
    assert ids(r) == sorted(set(ids(r)), key=ids(r).index)  # no duplicates
    assert [x.rank for x in r.results] == list(range(1, len(r.results) + 1))


@pytest.mark.skipif(HAS_FUSION, reason="only meaningful when fusion is missing")
def test_missing_fusion_module_is_reported_not_raised(engine):
    assert "not in this build" in engine.unavailable_reason("fusion")
    r = engine.search("compact camera", {"tfidf": True, "rocchio": True, "fusion": True})
    assert r.used == ["tfidf"] and any("not in this build" in n for n in r.notes)


@pytest.mark.skipif(HAS_DENSE, reason="only meaningful when dense is missing")
def test_missing_dense_module_is_skipped_with_a_note(engine):
    assert "not in this build" in engine.unavailable_reason("dense")
    r = engine.search("compact camera", {"dense": True})
    assert r.used == ["tfidf"] and any("Dense" in n for n in r.notes)


@pytest.mark.skipif(not HAS_DENSE, reason="dense module not in this build")
def test_dense_toggle_with_an_offline_encoder(engine, monkeypatch, tmp_path):
    from phonesense.retrieval import dense

    monkeypatch.setattr(dense, "encoder_from_config", lambda: dense.HashingEncoder())
    monkeypatch.setitem(__import__("phonesense.config", fromlist=["x"]).get_config(), "dense",
                        {"cache_dir": str(tmp_path / "cache")})
    r = engine.search("compact camera", {"dense": True})
    assert r.used == ["dense"] and r.results


@pytest.mark.skipif(not (HAS_DENSE and HAS_FUSION), reason="needs dense and fusion")
def test_all_systems_fused(engine, monkeypatch, tmp_path):
    from phonesense.retrieval import dense

    monkeypatch.setattr(dense, "encoder_from_config", lambda: dense.HashingEncoder())
    monkeypatch.setitem(__import__("phonesense.config", fromlist=["x"]).get_config(), "dense",
                        {"cache_dir": str(tmp_path / "cache")})
    use_fake_llm(engine, tmp_path)
    toggles = {"tfidf": True, "rocchio": True, "hyde": True, "dense": True, "fusion": True}
    r = engine.search("compact camera", toggles)
    assert r.used == ["tfidf", "rocchio", "hyde", "dense"] and r.results and r.notes == []


def test_find_phones_file_prefers_the_environment_variable(monkeypatch, fixtures):
    monkeypatch.setenv("PHONESENSE_PHONES", str(fixtures / "phones.jsonl"))
    assert find_phones_file() == (fixtures / "phones.jsonl", False)
    monkeypatch.setenv("PHONESENSE_PHONES", str(fixtures / "missing.jsonl"))
    with pytest.raises(NoDataError, match="does not exist"):
        find_phones_file()


def test_find_phones_file_explains_what_to_do_when_there_is_no_data(monkeypatch, tmp_path):
    monkeypatch.delenv("PHONESENSE_PHONES", raising=False)
    monkeypatch.setattr("phonesense.ui.engine.get_path", lambda key: tmp_path / key)
    with pytest.raises(NoDataError, match="build_data.py"):
        find_phones_file()


# ---------------------------------------------------------------------------- score breakdown


def test_single_system_breakdown_matches_the_result(engine):
    r = engine.search("compact camera battery", {"rocchio": True})
    assert r.results
    for res in r.results:
        assert [p.key for p in res.breakdown] == ["rocchio"]
        part = res.breakdown[0]
        assert part.label == "Rocchio feedback" and part.rank == res.rank
        assert part.score == pytest.approx(res.score)


def test_breakdown_marks_phones_a_system_did_not_return(engine):
    ranking = [("B0TEST0001", 0.9), ("B0TEST0002", 0.5)]
    only_one = {"tfidf": [("B0TEST0001", 2.0)], "rocchio": ranking}
    rows = engine._results(ranking, only_one)
    assert [(p.key, p.rank) for p in rows[0].breakdown] == [("tfidf", 1), ("rocchio", 1)]
    second = {p.key: p for p in rows[1].breakdown}
    assert second["tfidf"].rank is None and second["tfidf"].score is None
    assert second["rocchio"].rank == 2 and second["rocchio"].score == 0.5
    assert engine._results(ranking)[0].breakdown == ()  # no systems given: empty breakdown


@pytest.mark.skipif(not HAS_FUSION, reason="fusion module not in this build")
def test_fused_breakdown_lists_every_selected_system(engine):
    toggles = {"tfidf": True, "rocchio": True, "fusion": True}
    for res in engine.search("compact camera battery", toggles).results:
        assert [p.key for p in res.breakdown] == ["tfidf", "rocchio"]


# ---------------------------------------------------------------------------- BM25 toggle

HAS_BM25 = importlib.util.find_spec("phonesense.retrieval.bm25") is not None


@pytest.mark.skipif(not HAS_BM25, reason="BM25 module not in this build")
def test_bm25_toggle_runs_the_real_retriever_with_the_tuned_settings(engine):
    from phonesense.config import get_config

    r = engine.search("compact camera battery", {"bm25": True})
    assert r.used == ["bm25"] and r.notes == [] and r.results
    tuned = get_config().get("bm25", {}).get("tuned")
    bm25 = engine._bm25
    if tuned:
        assert (bm25.k1, bm25.b, bm25.name) == (tuned["k1"], tuned["b"], "bm25_tuned")
    assert bm25.index is engine.tfidf.index  # shares our index


@pytest.mark.skipif(not HAS_BM25, reason="BM25 module not in this build")
def test_bm25_falls_back_to_the_textbook_defaults_without_tuned_settings(engine, monkeypatch):
    import phonesense.ui.engine as eng

    monkeypatch.setattr(eng, "get_config", lambda: {"bm25": {"default": {"k1": 1.5, "b": 0.5}}})
    engine.search("compact camera battery", {"bm25": True})
    assert (engine._bm25.k1, engine._bm25.b, engine._bm25.name) == (1.5, 0.5, "bm25_default")


@pytest.mark.skipif(not HAS_BM25, reason="BM25 module not in this build")
def test_bm25_and_tfidf_can_be_fused_and_both_appear_in_the_breakdown(engine):
    if importlib.util.find_spec("phonesense.retrieval.fusion") is None:
        pytest.skip("fusion module not in this build")
    toggles = {"tfidf": True, "bm25": True, "fusion": True}
    r = engine.search("compact camera battery", toggles)
    assert r.used == ["tfidf", "bm25"]
    assert all([p.key for p in res.breakdown] == ["tfidf", "bm25"] for res in r.results)


@pytest.mark.skipif(HAS_BM25, reason="only meaningful while BM25 is missing")
def test_missing_bm25_is_reported_not_raised(engine):
    assert "not in this build" in engine.unavailable_reason("bm25")
    r = engine.search("compact camera", {"bm25": True})
    assert r.used == ["tfidf"] and any("BM25" in n for n in r.notes)


# ---------------------------------------------------------------------------- priors

HAS_RERANK = importlib.util.find_spec("phonesense.graph.rerank") is not None
needs_rerank = pytest.mark.skipif(not HAS_RERANK, reason="prior reranking not in this build")

IDS = ["B0TEST0001", "B0TEST0002", "B0TEST0003"]


def engine_with_priors(fixtures, **priors):
    return SearchEngine(read_phones(fixtures / "phones.jsonl"), priors=priors)


@needs_rerank
def test_prior_with_lambda_one_ranks_by_the_prior_alone(fixtures):
    # prefer the phone that TF-IDF ranks last for "camera battery": 0003 > 0002 > 0001
    prior = {"B0TEST0001": 0.0, "B0TEST0002": 0.5, "B0TEST0003": 1.0}
    eng = engine_with_priors(fixtures, pagerank=lambda: prior)
    plain = ids(eng.search("compact camera battery", {"tfidf": True}, k=3))
    r = eng.search("compact camera battery", {"tfidf": True, "pagerank": True}, 3, prior_weight=1.0)
    ranked = ids(r)
    assert ranked == [d for d in sorted(prior, key=prior.get, reverse=True) if d in plain]
    assert r.used == ["tfidf"] and r.notes == []


@needs_rerank
def test_prior_with_lambda_zero_keeps_the_order_but_still_reports_the_prior(fixtures):
    eng = engine_with_priors(fixtures, pagerank=lambda: {d: 0.3 for d in IDS})
    plain = eng.search("compact camera battery", {"tfidf": True})
    r = eng.search("compact camera battery", {"tfidf": True, "pagerank": True}, prior_weight=0.0)
    assert ids(r) == ids(plain)
    row = r.results[0].breakdown[-1]
    assert row.kind == "prior" and row.key == "pagerank" and row.rank is None
    assert row.score == pytest.approx(0.3) and "lambda=0" in row.label


@needs_rerank
def test_two_priors_are_averaged_and_both_listed(fixtures):
    seen = []

    def loader(name, value):
        def load():
            seen.append(name)
            return {d: value for d in IDS}
        return load

    eng = engine_with_priors(fixtures, pagerank=loader("pagerank", 1.0), trust=loader("trust", 0.0))
    toggles = {"tfidf": True, "pagerank": True, "trust": True}
    r = eng.search("compact camera battery", toggles, prior_weight=0.5)
    assert [p.key for p in r.results[0].breakdown if p.kind == "prior"] == ["pagerank", "trust"]
    eng.search("compact camera battery", toggles, prior_weight=0.5)
    assert sorted(seen) == ["pagerank", "trust"]  # each prior is computed once, then kept


@needs_rerank
def test_a_failing_prior_is_skipped_with_a_note(fixtures):
    def broken():
        raise FileNotFoundError("graph_edges.tsv not found")

    eng = engine_with_priors(fixtures, pagerank=broken)
    plain = ids(eng.search("compact camera battery", {"tfidf": True}))
    r = eng.search("compact camera battery", {"tfidf": True, "pagerank": True})
    assert ids(r) == plain and any("PageRank prior skipped" in n for n in r.notes)


@needs_rerank
def test_priors_never_add_phones(fixtures):
    eng = engine_with_priors(fixtures, trust=lambda: {d: 1.0 for d in IDS})
    base = set(ids(eng.search("camera", {"tfidf": True}, k=20)))
    got = set(ids(eng.search("camera", {"tfidf": True, "trust": True}, k=20, prior_weight=1.0)))
    assert got == base


def test_prior_weight_must_be_a_fraction(engine):
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            engine.search("compact camera", {"tfidf": True}, prior_weight=bad)


@pytest.mark.skipif(HAS_RERANK, reason="only meaningful while prior reranking is missing")
def test_priors_are_unavailable_without_the_rerank_module(engine):
    for key in ("pagerank", "trust"):
        assert "not in this build" in engine.unavailable_reason(key)
    r = engine.search("compact camera", {"tfidf": True, "pagerank": True})
    assert r.used == ["tfidf"] and any("PageRank prior skipped" in n for n in r.notes)


# ---------------------------------------------------------------------------- robustness


def test_an_unexpected_error_in_one_system_is_a_note_not_a_crash(engine):
    class Boom:
        def search(self, query, k=100):
            raise RuntimeError("model download failed")

    engine.rocchio = Boom()
    r = engine.search("compact camera", {"tfidf": True, "rocchio": True, "fusion": False})
    assert r.results and any("Rocchio feedback skipped: RuntimeError" in n for n in r.notes)
    only = engine.search("compact camera", {"rocchio": True})
    assert only.used == ["tfidf"] and any("Falling back to TF-IDF" in n for n in only.notes)
    assert engine.unavailable_reason("rocchio") is None  # a runtime failure is retried next time


# ---------------------------------------------------------------------------- tuned fusion


@pytest.mark.skipif(not HAS_FUSION, reason="fusion module not in this build")
def test_fusion_uses_the_dev_tuned_rrf_constant_from_the_config(engine, monkeypatch):
    import phonesense.ui.engine as eng
    from phonesense.retrieval.fusion import reciprocal_rank_fusion

    a = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    b = [("d3", 0.9), ("d2", 0.5), ("d1", 0.1)]
    monkeypatch.setattr(eng, "get_config", lambda: {"fusion": {"rrf_k": 5}})
    assert engine._fuse([a, b], "rrf", 10, ["tfidf", "rocchio"]) == reciprocal_rank_fusion(
        [a, b], 5, depth=10
    )
    monkeypatch.setattr(eng, "get_config", lambda: {})  # no config: the published default
    assert engine._fuse([a, b], "rrf", 10, ["tfidf", "rocchio"]) == reciprocal_rank_fusion(
        [a, b], 60, depth=10
    )


@pytest.mark.skipif(not HAS_FUSION, reason="fusion module not in this build")
def test_linear_fusion_uses_tuned_weights_only_for_the_bm25_dense_pair(engine, monkeypatch):
    import phonesense.ui.engine as eng
    from phonesense.retrieval.fusion import weighted_linear_fusion

    a = [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)]
    b = [("d3", 0.9), ("d2", 0.5), ("d1", 0.1)]
    cfg = {"fusion": {"tuned": {"linear": {"weights": [0.9, 0.1]}}}}
    monkeypatch.setattr(eng, "get_config", lambda: cfg)
    assert engine._fuse([a, b], "linear", 10, ["bm25", "dense"]) == weighted_linear_fusion(
        [a, b], [0.9, 0.1], depth=10
    )
    assert engine._fuse([a, b], "linear", 10, ["tfidf", "dense"]) == weighted_linear_fusion(
        [a, b], None, depth=10
    )  # other pairs: equal weights


# ---------------------------------------------------------------------------- filters and phrase

HAS_FILTERS = importlib.util.find_spec("phonesense.retrieval.filters") is not None
HAS_POSITIONAL = importlib.util.find_spec("phonesense.retrieval.positional") is not None


def test_describe_filter_is_readable():
    from phonesense.ui.engine import describe_filter

    leaf = {"field": "price", "op": "<", "value": 300}
    assert describe_filter(leaf) == "price < 300"
    both = {"and": [leaf, {"field": "has_5g", "op": "==", "value": True}]}
    assert describe_filter(both) == "price < 300 AND has_5g == True"
    nested = {"or": [both, {"not": {"field": "os", "op": "==", "value": "ios"}}]}
    assert describe_filter(nested) == "(price < 300 AND has_5g == True) OR NOT (os == ios)"


@pytest.mark.skipif(not HAS_FILTERS, reason="filters module not in this build")
def test_filters_keep_only_phones_that_satisfy_the_constraint_in_the_query(engine):
    # fixture: Nova 299.99 USD and 5G; Pebble 449 USD; the third phone has no price
    plain = engine.search("5G phone under $300", {"tfidf": True})
    on = engine.search("5G phone under $300", {"tfidf": True, "filters": True})
    assert "price < 300" in on.filter_summary and "has_5g == True" in on.filter_summary
    assert on.n_allowed == 1 and ids(on) == ["B0TEST0001"]
    assert plain.filter_summary is None and plain.n_allowed is None


@pytest.mark.skipif(not HAS_FILTERS, reason="filters module not in this build")
def test_an_unknown_value_never_passes_a_filter(engine):
    r = engine.search("phone under $500", {"tfidf": True, "filters": True})
    assert "B0TEST0003" not in ids(r)  # price unknown
    assert engine.search("phone under $500", {"filters": True}).n_allowed == 2


@pytest.mark.skipif(not HAS_FILTERS, reason="filters module not in this build")
def test_a_query_without_a_constraint_is_not_filtered_and_says_so(engine):
    r = engine.search("great camera", {"tfidf": True, "filters": True})
    assert r.filter_summary is None and r.n_allowed is None
    assert any("No numeric constraint found" in n for n in r.notes)


@pytest.mark.skipif(not HAS_FILTERS, reason="filters module not in this build")
def test_a_filter_nothing_passes_gives_an_empty_page_with_a_note(engine):
    r = engine.search("phone under $10", {"tfidf": True, "filters": True})
    assert r.results == [] and any("No phone passes the filter" in n for n in r.notes)


@pytest.mark.skipif(not HAS_FILTERS, reason="filters module not in this build")
def test_the_filter_applies_to_every_selected_system(engine):
    toggles = {"tfidf": True, "rocchio": True, "fusion": True, "filters": True}
    if importlib.util.find_spec("phonesense.retrieval.fusion") is None:
        toggles.pop("fusion")
    for r in (engine.search("5G phone under $300", toggles),):
        assert set(ids(r)) <= {"B0TEST0001"}


@pytest.mark.skipif(HAS_FILTERS, reason="only meaningful while the filters module is missing")
def test_missing_filters_are_reported_not_raised(engine):
    r = engine.search("5G phone under $300", {"tfidf": True, "filters": True})
    assert r.results and any("Spec filters skipped" in n for n in r.notes)


needs_phrase = pytest.mark.skipif(
    not (HAS_POSITIONAL and HAS_BM25), reason="phrase boost needs BM25 and positional"
)


@needs_phrase
def test_phrase_boost_wraps_bm25_and_is_labelled(engine):
    r = engine.search('"compact phone" camera', {"bm25": True, "phrase": True})
    assert r.used == ["bm25"] and r.results and r.notes == []
    assert r.results[0].breakdown[0].label == "BM25 (tuned) + phrase boost"
    assert engine._phrase.name.endswith("+phrase")
    plain = engine.search('"compact phone" camera', {"bm25": True})
    assert plain.results[0].breakdown[0].label == "BM25 (tuned)"


@needs_phrase
def test_phrase_boost_without_bm25_says_to_switch_bm25_on(engine):
    r = engine.search("compact camera", {"tfidf": True, "phrase": True})
    assert r.used == ["tfidf"] and any("switch BM25 on" in n for n in r.notes)
