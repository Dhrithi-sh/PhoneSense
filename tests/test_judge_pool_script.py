"""scripts/judge_pool.py: model assignment, interleaving, saving progress (fake LLM, no network)."""

import importlib.util
import json
from pathlib import Path

import pytest

from phonesense.io import read_qrels
from phonesense.llm import client as client_mod
from phonesense.llm.client import LLMError
from phonesense.llm.judge import assign_models, interleave
from phonesense.llm.pooling import write_pool


def test_assign_models_is_round_robin_over_sorted_ids_and_one_model_per_query():
    got = assign_models(["q03", "q01", "q02", "q01"], ["a", "b"])
    assert got == {"q01": "a", "q02": "b", "q03": "a"}
    assert set(assign_models(["q1"], ["only"]).values()) == {"only"}
    with pytest.raises(ValueError):
        assign_models(["q1"], [])


def test_interleave_alternates_and_keeps_every_item():
    assert interleave([[1, 2, 3], ["a"], ["x", "y"]]) == [1, "a", "x", 2, "y", 3]
    assert interleave([]) == [] and interleave([[], []]) == []


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "judge_pool.py"
    spec = importlib.util.spec_from_file_location("judge_pool_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup(tmp_path, fixtures, monkeypatch):
    script = _load_script()
    paths = {"queries": fixtures / "queries.jsonl", "qrels_llm": tmp_path / "llm.txt",
             "results_dir": tmp_path / "results", "llm_logs": tmp_path / "logs"}
    monkeypatch.setattr(script, "get_path", lambda key: paths[key])
    monkeypatch.setattr(client_mod, "get_path", lambda key: paths[key])
    write_pool(tmp_path / "pool.tsv", {"q01": {"B0TEST0001": ["x"], "B0TEST0002": ["x"]},
                                       "q02": {"B0TEST0001": ["x"]}})
    return script, tmp_path, fixtures, monkeypatch


def _args(tmp_path, fixtures, *extra):
    return ["--split", "dev", "--pool", str(tmp_path / "pool.tsv"),
            "--phones", str(fixtures / "phones.jsonl"), "--out", str(tmp_path / "llm.txt"),
            "--provider", "gemini", *extra]


def test_models_split_the_queries_and_alternate_calls(setup):
    script, tmp_path, fixtures, monkeypatch = setup
    calls = []

    def backend_factory(retries):
        def call(prompt, model, temperature):
            calls.append(model)
            return json.dumps({"grade": 3 if model == "m1" else 1, "reason": "r"})

        return call

    monkeypatch.setitem(client_mod.BACKENDS, "gemini", backend_factory)
    assert script.main(_args(tmp_path, fixtures, "--models", "m1,m2")) == 0
    qrels = read_qrels(tmp_path / "llm.txt")
    assert qrels["q01"] == {"B0TEST0001": 3, "B0TEST0002": 3}  # q01 -> m1
    assert qrels["q02"] == {"B0TEST0001": 1}  # q02 -> m2
    assert calls == ["m1", "m2", "m1"]  # alternating, not all of m1 first
    mapping = json.loads((tmp_path / "results" / "judge_models_dev.json").read_text())
    assert mapping["q01"] == "m1" and mapping["q02"] == "m2"


def test_a_failing_call_saves_the_grades_so_far_and_returns_2(setup, capsys):
    script, tmp_path, fixtures, monkeypatch = setup
    state = {"n": 0}

    def backend_factory(retries):
        def call(prompt, model, temperature):
            state["n"] += 1
            if state["n"] > 2:
                raise LLMError("daily token limit reached")
            return '{"grade": 2, "reason": "ok"}'

        return call

    monkeypatch.setitem(client_mod.BACKENDS, "gemini", backend_factory)
    assert script.main(_args(tmp_path, fixtures, "--models", "m1,m2")) == 2
    saved = read_qrels(tmp_path / "llm.txt")
    assert sum(len(d) for d in saved.values()) == 2  # two judged before the quota ran out
    err = capsys.readouterr().err
    assert "stopped after 2/3" in err and "unjudged" in err


def test_rerun_after_a_failure_replays_the_logged_calls(setup):
    script, tmp_path, fixtures, monkeypatch = setup
    sent = []

    def backend_factory(retries):
        def call(prompt, model, temperature):
            sent.append(1)
            if len(sent) == 3:
                raise LLMError("quota")
            return '{"grade": 2, "reason": "ok"}'

        return call

    monkeypatch.setitem(client_mod.BACKENDS, "gemini", backend_factory)
    assert script.main(_args(tmp_path, fixtures, "--models", "m1,m2")) == 2
    first_round = len(sent)
    assert script.main(_args(tmp_path, fixtures, "--models", "m1,m2")) == 0
    assert len(sent) == first_round + 1  # only the one missing pair is sent again
    assert sum(len(d) for d in read_qrels(tmp_path / "llm.txt").values()) == 3
