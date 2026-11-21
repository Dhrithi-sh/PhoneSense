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


# ---------------------------------------------------------------------------- mistral and throttle


def test_mistral_is_a_registered_provider_with_its_own_key_variable():
    from phonesense.llm.client import BACKENDS, KEY_VARS

    assert "mistral" in BACKENDS and KEY_VARS["mistral"] == "MISTRAL_API_KEY"
    assert LLMClient("mistral", "m", backend=lambda *a: "x").provider == "mistral"


def test_mistral_backend_reads_the_chat_completion_and_sends_the_key(monkeypatch):
    from phonesense.llm import client as mod

    seen = {}

    def fake_post(url, headers, body, retries):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setenv("MISTRAL_API_KEY", "secret-key")
    monkeypatch.setattr(mod, "_post_json", fake_post)
    assert mod.mistral_backend()("prompt", "mistral-small-latest", 0.0) == "hello"
    assert seen["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret-key"
    assert seen["body"]["messages"] == [{"role": "user", "content": "prompt"}]


def test_mistral_backend_rejects_an_unexpected_response(monkeypatch):
    from phonesense.llm import client as mod

    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr(mod, "_post_json", lambda *a: {"error": "nope"})
    with pytest.raises(LLMError, match="unexpected Mistral response"):
        mod.mistral_backend()("p", "m", 0.0)


def test_retry_delay_honours_retry_after_and_falls_back_to_backoff():
    import urllib.error
    from email.message import Message

    from phonesense.llm.client import _retry_delay

    def err(value):
        headers = Message()
        if value is not None:
            headers["Retry-After"] = value
        return urllib.error.HTTPError("http://x", 429, "Too Many", headers, None)

    assert _retry_delay(err("7"), 0) == 7.0
    assert _retry_delay(err("500"), 0) == 60.0  # capped
    assert _retry_delay(err(None), 2) == 4.0  # 2 ** attempt
    assert _retry_delay(err("soon"), 1) == 2.0  # not a number


def test_throttle_waits_between_real_calls_only(tmp_path):
    now, sleeps = [100.0], []

    def clock():
        return now[0]

    def sleep(seconds):
        sleeps.append(round(seconds, 3))
        now[0] += seconds

    c = LLMClient("mistral", "m", backend=lambda p, m, t: p, log_dir=tmp_path,
                  min_interval=1.0, clock=clock, sleep=sleep)
    c.complete("a", purpose="hyde")  # first call: no wait
    now[0] += 0.25
    c.complete("b", purpose="hyde")  # 0.25 s since the last call: wait 0.75 s
    c.complete("a", purpose="hyde")  # replayed from the log: no call, no wait
    now[0] += 5.0
    c.complete("c", purpose="hyde")  # long enough since the last call: no wait
    assert sleeps == [0.75] and c.calls_made == 3 and c.calls_replayed == 1


def test_default_interval_comes_from_the_config_for_mistral_only(tmp_path):
    assert LLMClient("mistral", "m", backend=lambda *a: "x", log_dir=tmp_path).min_interval > 0
    assert LLMClient("gemini", "m", backend=lambda *a: "x", log_dir=tmp_path).min_interval == 0


# ---------------------------------------------------------------------------- groq


def test_groq_is_registered_and_throttled_by_default():
    from phonesense.llm.client import BACKENDS, KEY_VARS

    assert "groq" in BACKENDS and KEY_VARS["groq"] == "GROQ_API_KEY"
    assert LLMClient("groq", "m", backend=lambda *a: "x").min_interval > 0


def test_groq_backend_sends_low_reasoning_effort_only_for_gpt_oss(monkeypatch):
    from phonesense.llm import client as mod

    bodies = []

    def fake_post(url, headers, body, retries):
        bodies.append((url, headers["Authorization"], body))
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setenv("GROQ_API_KEY", "k-groq")
    monkeypatch.setattr(mod, "_post_json", fake_post)
    call = mod.groq_backend()
    assert call("p", "openai/gpt-oss-120b", 0.0) == "ok"
    assert call("p", "qwen/qwen3.8-27b", 0.0) == "ok"
    (url, auth, first), (_, _, second) = bodies
    assert url == "https://api.groq.com/openai/v1/chat/completions" and auth == "Bearer k-groq"
    assert first["reasoning_effort"] == "low" and "reasoning_effort" not in second


def test_groq_backend_rejects_an_unexpected_response(monkeypatch):
    from phonesense.llm import client as mod

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(mod, "_post_json", lambda *a: {"error": "nope"})
    with pytest.raises(LLMError, match="unexpected Groq response"):
        mod.groq_backend()("p", "m", 0.0)


def test_every_request_carries_a_user_agent(monkeypatch):
    import urllib.request

    from phonesense.llm import client as mod

    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    mod._post_json("http://x", {"Authorization": "Bearer k"}, {}, 0)
    assert seen["ua"].startswith("phonesense/")
