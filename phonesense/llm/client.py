"""Provider-agnostic LLM client that logs every call (owner: B).

Used by the HyDE query expansion (T15) and the relevance judge (T09). Two rules from GUIDE.md
section 7 shape the design:

* **Every call is logged** to ``<paths.llm_logs>/<purpose>.jsonl`` (one JSON object per call:
  time, provider, model, purpose, temperature, prompt, response). These logs are committed so
  results are reproducible. API keys are never written to a log.
* **Calls are replayed, not repeated.** A call is keyed by a hash of (provider, model,
  temperature, prompt). If an earlier log has the same key, its response is returned and
  nothing is sent. Re-running an experiment is therefore free and gives identical numbers.
  ``offline=True`` turns a replay miss into an error, so a fresh clone can reproduce results
  without any API key.

API keys come from the environment or the git-ignored ``.env`` (see ``.env.example``), never
from code. Backends use only the standard library (``urllib``), so there is no extra dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from phonesense.config import get_config, get_path, repo_root

Backend = Callable[[str, str, float], str]  # (prompt, model, temperature) -> response text

KEY_VARS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY", "groq": "GROQ_API_KEY"}


class LLMError(RuntimeError):
    """The provider could not answer (missing key, offline replay miss, HTTP failure)."""


def load_env(path: str | Path | None = None) -> None:
    """Read ``KEY=VALUE`` lines of ``.env`` into ``os.environ`` (existing variables win)."""
    path = Path(path) if path is not None else repo_root() / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def call_key(provider: str, model: str, temperature: float, prompt: str) -> str:
    """Stable identity of a call, used to replay logged responses."""
    payload = json.dumps([provider, model, round(temperature, 6), prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before retrying: the server's ``Retry-After`` (capped at 60) when it
    sends one, else exponential backoff (1, 2, 4, ...)."""
    header = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(float(header), 60.0) if header is not None else float(2**attempt)
    except ValueError:
        return float(2**attempt)


def _post_json(url: str, headers: dict[str, str], body: dict, retries: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    headers = {"User-Agent": "phonesense/0.1 (course project)", **headers}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(_retry_delay(exc, attempt))
                continue
            raise LLMError(f"HTTP {exc.code} from {url.split('?')[0]}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise LLMError(f"could not reach {url.split('?')[0]}: {exc.reason}") from exc
    raise LLMError("unreachable")  # pragma: no cover


def _api_key(provider: str) -> str:
    load_env()
    var = KEY_VARS[provider]
    key = os.environ.get(var, "").strip()
    if not key:
        raise LLMError(f"{var} is not set: copy .env.example to .env and fill it in")
    return key


def openai_backend(retries: int = 3) -> Backend:
    def call(prompt: str, model: str, temperature: float) -> str:
        out = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {_api_key('openai')}", "Content-Type": "application/json"},
            {
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            retries,
        )
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected OpenAI response: {str(out)[:200]}") from exc

    return call


def gemini_backend(retries: int = 3) -> Backend:
    def call(prompt: str, model: str, temperature: float) -> str:
        out = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": _api_key("gemini"), "Content-Type": "application/json"},
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
            retries,
        )
        try:
            parts = out["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Gemini response: {str(out)[:200]}") from exc

    return call


def mistral_backend(retries: int = 3) -> Backend:
    """Mistral La Plateforme chat completions (same message format as OpenAI)."""

    def call(prompt: str, model: str, temperature: float) -> str:
        out = _post_json(
            "https://api.mistral.ai/v1/chat/completions",
            {"Authorization": f"Bearer {_api_key('mistral')}", "Content-Type": "application/json"},
            {
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            retries,
        )
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Mistral response: {str(out)[:200]}") from exc

    return call


def groq_backend(retries: int = 3) -> Backend:
    """Groq chat completions (OpenAI format). gpt-oss models are reasoning models: ask for low
    reasoning effort so they do not spend the small free daily token budget on thinking."""

    def call(prompt: str, model: str, temperature: float) -> str:
        body: dict = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
        out = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {_api_key('groq')}", "Content-Type": "application/json"},
            body,
            retries,
        )
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Groq response: {str(out)[:200]}") from exc

    return call


BACKENDS: dict[str, Callable[[int], Backend]] = {
    "openai": openai_backend,
    "gemini": gemini_backend,
    "mistral": mistral_backend,
    "groq": groq_backend,
}


class LLMClient:
    """Calls an LLM, logs the call, and replays earlier identical calls.

    ``provider`` / ``model`` / ``temperature`` default to the ``llm:`` block of config.yaml.
    Pass ``backend`` to use a fake (tests); then no key is needed.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        temperature: float | None = None,
        backend: Backend | None = None,
        log_dir: str | Path | None = None,
        offline: bool = False,
        min_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = get_config().get("llm", {})
        self.provider = provider or cfg.get("provider", "gemini")
        if backend is None and self.provider not in BACKENDS:
            raise ValueError(f"unknown provider {self.provider!r}; use one of {sorted(BACKENDS)}")
        models = cfg.get("models", {})
        self.model = model or models.get(self.provider) or cfg.get("model")
        if not self.model:
            raise ValueError(f"no model configured for provider {self.provider!r}")
        default_temp = cfg.get("temperature", 0.0)
        self.temperature = float(default_temp if temperature is None else temperature)
        self.offline = offline
        self.log_dir = Path(log_dir) if log_dir is not None else get_path("llm_logs")
        self._backend = backend
        self._retries = int(cfg.get("max_retries", 3))
        # free tiers cap requests per second: wait at least this long between real calls
        intervals = cfg.get("min_interval_seconds", {})
        self.min_interval = float(
            min_interval if min_interval is not None else intervals.get(self.provider, 0.0)
        )
        self._clock, self._sleep, self._last_call = clock, sleep, None
        self._replay: dict[str, str] | None = None
        self.calls_made = 0
        self.calls_replayed = 0

    # ------------------------------------------------------------------ replay

    def _load_replay(self) -> dict[str, str]:
        replay: dict[str, str] = {}
        for path in sorted(self.log_dir.glob("*.jsonl")) if self.log_dir.is_dir() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                replay[rec["key"]] = rec["response"]
        return replay

    def _append_log(self, purpose: str, key: str, prompt: str, response: str, temp: float) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "key": key,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": self.provider,
            "model": self.model,
            "purpose": purpose,
            "temperature": temp,
            "prompt": prompt,
            "response": response,
        }
        with (self.log_dir / f"{purpose}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ public

    def complete(self, prompt: str, purpose: str, temperature: float | None = None) -> str:
        """Return the model's answer to ``prompt``; ``purpose`` names the log file (e.g. "hyde")."""
        if not purpose or not purpose.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"purpose must be a simple name, got {purpose!r}")
        temp = self.temperature if temperature is None else float(temperature)
        key = call_key(self.provider, self.model, temp, prompt)
        if self._replay is None:
            self._replay = self._load_replay()
        if key in self._replay:
            self.calls_replayed += 1
            return self._replay[key]
        if self.offline:
            raise LLMError(f"offline: no logged response for this {purpose} call")
        if self._backend is None:
            self._backend = BACKENDS[self.provider](self._retries)
        if self.min_interval > 0 and self._last_call is not None:
            wait = self.min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()
        response = self._backend(prompt, self.model, temp)
        self.calls_made += 1
        self._replay[key] = response
        self._append_log(purpose, key, prompt, response, temp)
        return response
