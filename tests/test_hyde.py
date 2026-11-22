"""HyDE: prompt, listing cleanup, generator logging, and the HyDE weight in Rocchio."""

import json

import pytest

from phonesense.index import Analyzer
from phonesense.llm.client import LLMClient
from phonesense.retrieval.hyde import HydeGenerator, HydeRetriever, build_prompt, clean_listing
from phonesense.retrieval.rocchio import RocchioRetriever, rocchio_update
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Query


def q(text: str) -> Query:
    return Query(qid="q01", text=text, split="dev")


def test_prompt_contains_the_query_once_and_demands_units():
    p = build_prompt('phone with a "big"   battery')
    assert p.count("phone with a 'big' battery") == 1
    assert "mAh" in p and "Output only the listing" in p
    with pytest.raises(ValueError):
        build_prompt("   ")


def test_clean_listing_strips_markdown_and_caps_length():
    raw = "```\n**Nova X5**\n- 5000mAh battery\n* 8GB RAM\n\n## Desc\nGreat. 5-star\n```"
    assert clean_listing(raw) == "Nova X5\n5000mAh battery\n8GB RAM\nDesc\nGreat. 5-star"
    assert len(clean_listing("x" * 5000, max_chars=100)) == 100
    assert clean_listing("") == ""


def test_generator_logs_the_call_and_replays_it(tmp_path):
    calls = []

    def backend(prompt, model, temperature):
        calls.append(prompt)
        return "**Pixel**\n- 12MP camera for night photos"

    gen = HydeGenerator(LLMClient("gemini", "m", backend=backend, log_dir=tmp_path))
    a = gen.generate("great night camera")
    b = gen.generate("great night camera")
    assert a == b == "Pixel\n12MP camera for night photos"
    assert len(calls) == 1
    rec = json.loads((tmp_path / "hyde.jsonl").read_text().splitlines()[0])
    assert rec["purpose"] == "hyde" and "great night camera" in rec["prompt"]


def test_rocchio_update_hyde_term_by_hand():
    """q={a:1}, no feedback docs, h={a:0.6, b:0.8}, delta=0.5: a = 1 + 0.3 = 1.3, b = 0.4."""
    out = rocchio_update({"a": 1.0}, [], [], 1.0, 0.75, 0.0, hypothetical={"a": 0.6, "b": 0.8},
                         delta=0.5)
    assert out == {"a": pytest.approx(1.3), "b": pytest.approx(0.4)}


@pytest.fixture
def collection(make_phone):
    return [
        make_phone("d1", "camera lens zoom"),
        make_phone("d2", "camera lens zoom"),
        make_phone("d3", "macro lens close up"),
        make_phone("d4", "battery charger"),
    ]


def make_hyde(collection, tmp_path, text="macro lens close up", **kw):
    client = LLMClient("gemini", "m", backend=lambda p, m, t: text, log_dir=tmp_path)
    r = HydeRetriever(
        HydeGenerator(client), TfidfRetriever(analyzer=Analyzer(stem=False)), m=2, **kw
    )
    r.fit(collection)
    return r


def test_hyde_text_pulls_in_matching_documents(collection, tmp_path):
    plain = RocchioRetriever(TfidfRetriever(analyzer=Analyzer(stem=False)), m=2, beta=0.0)
    plain.fit(collection)
    assert [d for d, _ in plain.search(q("camera"), 10)] == ["d1", "d2"]

    ranked = [d for d, _ in make_hyde(collection, tmp_path).search(q("camera"), 10)]
    assert ranked[:2] == ["d1", "d2"]  # the original query keeps its own weight
    assert "d3" in ranked and "d4" not in ranked


def test_empty_generation_equals_plain_rocchio(collection, tmp_path):
    hyde = make_hyde(collection, tmp_path, text="")
    rocchio = RocchioRetriever(TfidfRetriever(analyzer=Analyzer(stem=False)), m=2, n_terms=30)
    rocchio.fit(collection)
    assert hyde.search(q("camera"), 10) == rocchio.search(q("camera"), 10)


def test_delta_zero_ignores_the_hypothetical_text(collection, tmp_path):
    hyde = make_hyde(collection, tmp_path, delta=0.0)
    rocchio = RocchioRetriever(TfidfRetriever(analyzer=Analyzer(stem=False)), m=2, n_terms=30)
    rocchio.fit(collection)
    assert hyde.search(q("camera"), 10) == rocchio.search(q("camera"), 10)


def test_name_is_a_valid_run_name_prefix(collection, tmp_path):
    from phonesense.schemas import validate_run_name

    validate_run_name(make_hyde(collection, tmp_path).name + "_dev")
    assert make_hyde(collection, tmp_path).name.startswith("hyde_tfidf")
