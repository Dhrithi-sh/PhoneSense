"""Buying guide: grounding (tags, figures), prompt contents, logging and the engine hook."""

import json

import pytest

from phonesense.io import read_phones
from phonesense.llm.buying_guide import (
    BuyingGuide,
    build_prompt,
    clean_text,
    phone_facts,
    unverified_figures,
)
from phonesense.llm.client import LLMClient, LLMError
from phonesense.ui.engine import SearchEngine


@pytest.fixture
def phones(fixtures):
    return read_phones(fixtures / "phones.jsonl")


def guide_with(answer, tmp_path, max_phones=5):
    seen = []

    def backend(prompt, model, temperature):
        seen.append(prompt)
        return answer

    client = LLMClient("gemini", "m", backend=backend, log_dir=tmp_path)
    return BuyingGuide(client, max_phones), seen


def test_facts_use_human_units_and_name_unknowns(phones):
    nova, _pebble, bare = phones
    facts = phone_facts("P1", nova)
    assert facts.startswith("[P1] Nova X5 5G Smartphone 128GB")
    for text in ("price: $299.99", "RAM 8 GB", "battery 5000 mAh", "screen 6.5 inch",
                 "main camera 50 MP", "5G yes", "unknown specs: none"):
        assert text in facts
    unknown = phone_facts("P3", bare)
    assert "price: unknown" in unknown and "no ratings" in unknown
    assert "RAM" in unknown.split("unknown specs:")[1]
    assert "known specs: storage 32 GB" in unknown


def test_prompt_has_query_all_phones_and_the_only_use_facts_rule(phones):
    prompt, facts = build_prompt('compact "great" camera', phones)
    assert "compact 'great' camera" in prompt and "Use ONLY the facts" in prompt
    for tag in ("[P1]", "[P2]", "[P3]"):
        assert tag in facts and tag in prompt
    with pytest.raises(ValueError):
        build_prompt("x", [])
    with pytest.raises(ValueError):
        build_prompt("  ", phones)


@pytest.mark.parametrize(
    "guide",
    [
        "It costs $299.99 and has a 5000 mAh battery.",
        "Comes with 8 GB of RAM and 128 GB storage, 6.5 inch screen, 50 MP camera.",
        "Priced $449 with 3200 mAh.",
    ],
)
def test_real_figures_are_not_flagged(guide, phones):
    facts = build_prompt("x", phones)[1]
    assert unverified_figures(guide, facts) == []


def test_invented_figures_are_flagged_once_and_in_order(phones):
    facts = build_prompt("x", phones)[1]
    guide = "It has a 200 MP camera, 6000 mAh, and again 200 MP. It costs $299.99."
    assert unverified_figures(guide, facts) == ["200 MP", "6000 mAh"]


def test_tags_are_never_read_as_figures(phones):
    assert unverified_figures("[P1] beats [P2] and [P10].", build_prompt("x", phones)[1]) == []


def test_write_maps_tags_to_titles_and_reports_nothing_when_grounded(phones, tmp_path):
    answer = "[P1] is the best fit at $299.99 with 5000 mAh. [P2] costs $449.00."
    g, seen = guide_with(answer, tmp_path)
    result = g.write("compact camera", phones)
    assert result.text == ("**Nova X5 5G Smartphone 128GB** is the best fit at $299.99 with "
                           "5000 mAh. **Pebble Mini compact phone** costs $449.00.")
    assert result.doc_ids == ["B0TEST0001", "B0TEST0002"] and result.warnings == []
    assert "compact camera" in seen[0]


def test_unlisted_tags_are_removed_and_reported(phones, tmp_path):
    g, _ = guide_with("[P1] is good. [P7] is better.", tmp_path)
    result = g.write("q", phones)
    assert "[P7]" not in result.text and "P7" not in result.text.replace("[P7]", "")
    assert any("[P7]" in w for w in result.warnings) and result.doc_ids == ["B0TEST0001"]


def test_invented_figures_and_missing_phones_become_warnings(phones, tmp_path):
    g, _ = guide_with("A phone with a 200 MP camera is great.", tmp_path)
    result = g.write("q", phones)
    assert any("200 MP" in w for w in result.warnings)
    assert any("does not name any" in w for w in result.warnings) and result.doc_ids == []


def test_only_the_first_max_phones_are_offered(phones, tmp_path):
    g, seen = guide_with("[P1] fits.", tmp_path, max_phones=2)
    g.write("q", phones)
    assert "[P2]" in seen[0] and "[P3]" not in seen[0]
    with pytest.raises(ValueError):
        BuyingGuide(LLMClient("gemini", "m", backend=lambda *a: "", log_dir=tmp_path), 0)


def test_call_is_logged_and_replayed(phones, tmp_path):
    g, seen = guide_with("[P1] fits.", tmp_path)
    g.write("q", phones)
    g.write("q", phones)
    assert len(seen) == 1
    rec = json.loads((tmp_path / "guide.jsonl").read_text().splitlines()[0])
    assert rec["purpose"] == "guide" and "Use ONLY the facts" in rec["prompt"]


def test_clean_text_strips_markdown():
    assert clean_text("```\n## Guide\n**[P1]** is fine.\n```") == "Guide\n[P1] is fine."


def test_engine_hook_passes_the_displayed_phones_in_order(fixtures, tmp_path):
    engine = SearchEngine(read_phones(fixtures / "phones.jsonl"))
    g, seen = guide_with("[P1] is the pick.", tmp_path)
    engine._guide = g
    results = engine.search("compact camera battery", {"tfidf": True}).results
    guide = engine.write_guide("compact camera battery", results)
    assert guide.doc_ids == [results[0].doc_id]
    positions = [seen[0].index(r.title) for r in results]
    assert positions == sorted(positions)  # offered in the order they are displayed


def test_engine_hook_propagates_llm_errors(fixtures, tmp_path):
    def broken(prompt, model, temperature):
        raise LLMError("GEMINI_API_KEY is not set")

    engine = SearchEngine(read_phones(fixtures / "phones.jsonl"))
    engine._guide = BuyingGuide(LLMClient("gemini", "m", backend=broken, log_dir=tmp_path))
    results = engine.search("compact camera battery", {"tfidf": True}).results
    with pytest.raises(LLMError):
        engine.write_guide("compact camera battery", results)
