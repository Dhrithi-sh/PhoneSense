"""LLM judge: prompt content, answer parsing and the re-ask on a bad answer (fake LLM)."""

import json

import pytest

from phonesense.io import read_phones, read_queries
from phonesense.llm.client import LLMClient
from phonesense.llm.judge import JudgeError, LLMJudge, build_prompt, parse_grade, phone_summary


@pytest.fixture
def phones(fixtures):
    return read_phones(fixtures / "phones.jsonl")


@pytest.fixture
def queries(fixtures):
    return read_queries(fixtures / "queries.jsonl")


def judge_with(answers, tmp_path):
    seen = []

    def backend(prompt, model, temperature):
        seen.append(prompt)
        return answers[min(len(seen), len(answers)) - 1]

    return LLMJudge(LLMClient("gemini", "m", backend=backend, log_dir=tmp_path)), seen


def test_prompt_shows_query_filters_notes_and_phone_but_not_ratings(queries, phones):
    p = build_prompt(queries[0], phones[0])  # fixture q01 has filters and intent notes
    assert "phone under $300 with good battery and 5G" in p
    assert '"field": "price"' in p and "Battery >= 4500 mAh" in p
    assert phones[0].title in p and "battery_mah=5000" in p
    assert "avg_rating" not in p and "rating_count" not in p
    assert '{"grade": <0|1|2|3>' in p


def test_unknown_specs_are_marked_unknown(phones):
    summary = phone_summary(phones[2])  # fixture phone 3 has mostly null specs
    for text in ("ram_gb=unknown", "Price: unknown", "Brand: unknown"):
        assert text in summary


def test_type_prefix_is_not_leaked_into_the_prompt(phones):
    from phonesense.schemas import Query

    q = Query(qid="q99", text="iPhone 13", split="dev", intent_notes="type=model. Only this model.")
    p = build_prompt(q, phones[0])
    assert "type=model" not in p and "Only this model." in p


@pytest.mark.parametrize(
    "answer, expected",
    [
        ('{"grade": 3, "reason": "fits"}', (3, "fits")),
        ('```json\n{"grade": 0, "reason": "a case"}\n```', (0, "a case")),
        ('Sure! {"grade": 2, "reason": "ok"} Hope that helps', (2, "ok")),
        ('grade: 1', (1, "")),
    ],
)
def test_parse_grade_accepts_common_shapes(answer, expected):
    assert parse_grade(answer) == expected


@pytest.mark.parametrize(
    "answer",
    ['{"grade": 4}', '{"grade": 2.5}', '{"grade": "high"}', '{"grade": true}', "no idea", "", "{}"],
)
def test_parse_grade_rejects_bad_answers(answer):
    with pytest.raises(JudgeError):
        parse_grade(answer)


def test_judge_returns_a_judgment(queries, phones, tmp_path):
    judge, seen = judge_with(['{"grade": 3, "reason": "fits"}'], tmp_path)
    j = judge.judge(queries[0], phones[0])
    assert (j.qid, j.doc_id, j.grade, j.reason) == ("q01", phones[0].doc_id, 3, "fits")
    assert len(seen) == 1


def test_bad_answer_is_asked_again_once(queries, phones, tmp_path):
    judge, seen = judge_with(["I think it is good", '{"grade": 2, "reason": "ok"}'], tmp_path)
    assert judge.judge(queries[0], phones[0]).grade == 2
    assert len(seen) == 2 and "not valid" in seen[1]
    assert len((tmp_path / "judge.jsonl").read_text().splitlines()) == 2


def test_two_bad_answers_raise(queries, phones, tmp_path):
    judge, _ = judge_with(["nope", "still nope"], tmp_path)
    with pytest.raises(JudgeError):
        judge.judge(queries[0], phones[0])


def test_same_pair_is_replayed(queries, phones, tmp_path):
    judge, seen = judge_with([json.dumps({"grade": 1, "reason": "r"})], tmp_path)
    judge.judge(queries[0], phones[0])
    judge.judge(queries[0], phones[0])
    assert len(seen) == 1
