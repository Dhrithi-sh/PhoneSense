"""LLMClient: logging, replay, offline mode and validation, with a fake backend (no network)."""

import json

import pytest

from phonesense.llm.client import LLMClient, LLMError, call_key, load_env


class Fake:
    """Backend that records its calls and answers ``echo:<prompt>``."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, model, temperature):
        self.calls.append((prompt, model, temperature))
        return f"echo:{prompt}"


@pytest.fixture
def client(tmp_path):
    fake = Fake()
    return LLMClient("gemini", "m1", backend=fake, log_dir=tmp_path), fake


def test_call_is_logged_with_prompt_and_response(client, tmp_path):
    c, fake = client
    assert c.complete("hello", purpose="hyde") == "echo:hello"
    rec = json.loads((tmp_path / "hyde.jsonl").read_text().splitlines()[0])
    assert rec["prompt"] == "hello" and rec["response"] == "echo:hello"
    assert rec["provider"] == "gemini" and rec["model"] == "m1" and rec["purpose"] == "hyde"
    assert rec["key"] == call_key("gemini", "m1", 0.0, "hello")
    assert "key" in rec and "KEY" not in json.dumps(rec).upper().replace('"KEY"', "")


def test_identical_call_is_replayed_not_resent(client, tmp_path):
    c, fake = client
    c.complete("hello", purpose="hyde")
    c.complete("hello", purpose="hyde")
    assert len(fake.calls) == 1 and c.calls_made == 1 and c.calls_replayed == 1
    assert len((tmp_path / "hyde.jsonl").read_text().splitlines()) == 1


def test_replay_works_across_clients_and_purposes(tmp_path):
    LLMClient("gemini", "m1", backend=Fake(), log_dir=tmp_path).complete("q", purpose="hyde")
    second = Fake()
    again = LLMClient("gemini", "m1", backend=second, log_dir=tmp_path)
    assert again.complete("q", purpose="judge") == "echo:q"
    assert second.calls == []


def test_changing_model_temperature_or_prompt_makes_a_new_call(tmp_path):
    fake = Fake()
    a = LLMClient("gemini", "m1", backend=fake, log_dir=tmp_path)
    a.complete("q", purpose="hyde")
    a.complete("q", purpose="hyde", temperature=0.7)
    a.complete("q2", purpose="hyde")
    LLMClient("gemini", "m2", backend=fake, log_dir=tmp_path).complete("q", purpose="hyde")
    assert len(fake.calls) == 4


def test_offline_replays_but_never_calls(tmp_path):
    LLMClient("gemini", "m1", backend=Fake(), log_dir=tmp_path).complete("q", purpose="hyde")
    off = LLMClient("gemini", "m1", backend=Fake(), log_dir=tmp_path, offline=True)
    assert off.complete("q", purpose="hyde") == "echo:q"
    with pytest.raises(LLMError, match="offline"):
        off.complete("never asked", purpose="hyde")


def test_bad_purpose_and_provider_are_rejected(client):
    c, _ = client
    for bad in ("", "../x", "a b"):
        with pytest.raises(ValueError):
            c.complete("q", purpose=bad)
    with pytest.raises(ValueError):
        LLMClient("nope", "m")


def test_missing_key_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("phonesense.llm.client.load_env", lambda *a, **k: None)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMClient("gemini", "m1", log_dir=tmp_path).complete("q", purpose="hyde")


def test_load_env_reads_keys_but_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_KEY=abc\nEMPTY=\nBAR_KEY='quoted'\nSET_KEY=from-file\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAR_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.setenv("SET_KEY", "from-env")
    load_env(env)
    import os

    assert os.environ["FOO_KEY"] == "abc" and os.environ["BAR_KEY"] == "quoted"
    assert os.environ["SET_KEY"] == "from-env" and "EMPTY" not in os.environ
    for k in ("FOO_KEY", "BAR_KEY"):
        monkeypatch.delenv(k, raising=False)
