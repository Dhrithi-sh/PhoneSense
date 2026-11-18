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
