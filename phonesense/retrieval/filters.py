"""Boolean spec filters (T06): evaluator, natural-language constraint parser, wrapper.

**Evaluator.** :func:`matches` evaluates a filter in the CONTRACT.md 4.4 syntax (validated
by the existing ``schemas.validate_filter``) with three-valued (Kleene) logic, exactly as the
contract specifies:

* a leaf whose field is ``null`` on the phone is UNKNOWN, for every op (``!=`` included);
* ``and``: FALSE if any child is FALSE, else UNKNOWN if any is UNKNOWN, else TRUE;
  ``or``: TRUE if any child is TRUE, else UNKNOWN if any is UNKNOWN, else FALSE;
  ``not``: swaps TRUE/FALSE and keeps UNKNOWN;
* a phone passes only if the root is TRUE.

So **a null spec never satisfies a constraint, not even under ``not``**: a phone with unknown
RAM fails both ``ram_gb >= 8`` and ``not(ram_gb >= 8)``. This favours precision over recall:
phones with missing specs (common in this data) disappear from filtered result lists, which
the report has to mention when discussing filtered runs.

**Parser.** :func:`parse_constraints` turns common spec phrases into a filter and returns the
rest of the query for the ranker. It works on :meth:`Analyzer.normalize` output, so units are
joined exactly as in the index (``128 GB`` -> ``128gb``, ``5,000 mAh`` -> ``5000mah``,
``6.5 inch`` -> ``6.5in``). Patterns (all case-insensitive):

==========================================  =========================================
query phrase                                filter
==========================================  =========================================
under / below / less than / cheaper than /  ``price < X``  (``up to``, ``max``: ``<=``)
up to / max ``$X`` (``$`` optional)
over / above / more than / at least ``$X``  ``price > X`` / ``price >= X``
``$X-Y``, ``$X to $Y``, between X and Y     ``X <= price <= Y``
``8gb ram``, ``ram 8gb``, at least 6gb ram  ``ram_gb >= X``
``128gb`` [storage/rom/internal], ``1tb``   ``storage_gb >= X`` (TB x 1024)
bare ``Ngb`` with N < 16                    ``ram_gb >= N`` (small values are RAM)
``5000mah``                                 ``battery_mah >= X``
``6.5in`` [screen/display]                  ``|screen_in - X| <= 0.2`` (config)
at least / over / under ``6.5in``           ``screen_in >= X`` / ``<= X``
compact / small (phone/screen)              ``screen_in <= 6.2`` (config)
``5g``                                      ``has_5g == true``
android / ios / iphone                      ``os == android`` / ``os == ios``
==========================================  =========================================

Not parsed (left to the ranker, no numeric constraint): "budget", "cheap", "big battery",
"good camera", brands. "iphone" sets ``os == ios`` but stays in the residual text (it is also
a title word); "android", "ios", "5g" and the matched spec phrases are removed. Limits: one
value per spec (the first match wins), no negation ("not samsung"), no camera megapixels.

**Filter source.** :func:`resolve_filter`: a query's own ``filters`` (written in
queries.jsonl by teammate B) wins; otherwise the parser is used. The source is recorded per
query (``"queries.jsonl"``, ``"parser"`` or ``"none"``).

**Wrapper.** :class:`FilteredRetriever` wraps any retriever. ``mode="pre"`` passes the
allowed docs into the scoring loop (retrievers with ``score_text(text, k, allowed=...)``:
TF-IDF, BM25, BM25F, phrase boost), so filtered docs never enter the accumulators and the
surviving docs keep their unfiltered scores. ``mode="post"`` retrieves ``post_depth`` (1000)
docs and drops the disallowed ones, for retrievers that cannot take an allowed set (e.g.
dense retrieval). Either way, if fewer than k docs pass, fewer are returned (no padding).

The midterm had metadata filters on news (``dhvani/rank/filters.py``: source/section/date
applied before the top-k); this module is written new for phone specs, with the same
"filter before picking the top k" idea.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from phonesense.index import Analyzer, InvertedIndex
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.schemas import Phone, Query, validate_filter

TOP_LEVEL = ("price", "avg_rating", "rating_count", "brand")
UNKNOWN = None  # third truth value


def filters_config() -> dict[str, Any]:
    """The ``filters:`` section of config.yaml."""
    from phonesense.config import get_config

    return dict(get_config().get("filters") or {})


# ------------------------------------------------------------------ evaluator


def field_value(phone: Phone | Mapping[str, Any], field: str) -> Any:
    """The phone's value for a filter field (top-level or ``specs``); ``None`` if unknown."""
    if isinstance(phone, Mapping):
        return phone.get(field) if field in TOP_LEVEL else (phone.get("specs") or {}).get(field)
    return getattr(phone, field) if field in TOP_LEVEL else getattr(phone.specs, field)


def _leaf(value: Any, op: str, target: Any) -> bool | None:
    if value is None:
        return UNKNOWN
    if op == "in":
        return value in target
    return {
        "==": value == target,
        "!=": value != target,
        "<": value < target,
        "<=": value <= target,
        ">": value > target,
        ">=": value >= target,
    }[op]


def evaluate(phone: Phone | Mapping[str, Any], node: Mapping[str, Any]) -> bool | None:
    """Three-valued result of a (valid) filter node: True, False or None (UNKNOWN)."""
    if "field" in node:
        return _leaf(field_value(phone, node["field"]), node["op"], node["value"])
    ((key, val),) = node.items()
    if key == "not":
        r = evaluate(phone, val)
        return None if r is None else not r
    results = [evaluate(phone, child) for child in val]
    if key == "and":
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if any(r is True for r in results):  # "or"
        return True
    return None if any(r is None for r in results) else False


def matches(phone: Phone | Mapping[str, Any], filter_json: Mapping[str, Any] | None) -> bool:
    """True iff the phone passes the filter (root TRUE). ``None`` filter = everything passes."""
    if filter_json is None:
        return True
    validate_filter(filter_json)
    return evaluate(phone, filter_json) is True


def allowed_docs(phones: Iterable[Phone], filter_json: Mapping[str, Any] | None) -> set[str]:
    """doc_ids of the phones that pass ``filter_json`` (all of them if it is ``None``)."""
    phones = list(phones)
    if filter_json is None:
        return {p.doc_id for p in phones}
    validate_filter(filter_json)
    return {p.doc_id for p in phones if evaluate(p, filter_json) is True}


# ------------------------------------------------------------------ parser

_NUM = r"(\d+(?:\.\d+)?)"
_END = r"(?![a-z0-9.])"
_PRICE_RANGE = [
    re.compile(rf"\$\s*{_NUM}\s*(?:-|to)\s*\$?\s*{_NUM}{_END}"),
    re.compile(rf"between\s*\$?\s*{_NUM}\s*(?:and|-)\s*\$?\s*{_NUM}{_END}"),
    re.compile(rf"{_NUM}\s*(?:-|to)\s*{_NUM}\s*(?:dollars|usd|bucks){_END}"),
]
_PRICE_MAX = re.compile(
    rf"(under|below|less than|cheaper than|up to|upto|max(?:imum)?|within|<)\s*\$?\s*{_NUM}"
    rf"(?:\s*(?:dollars|usd|bucks))?{_END}"
)
_PRICE_MIN = re.compile(
    rf"(over|above|more than|at least|min(?:imum)?|>)\s*\$\s*{_NUM}(?:\s*(?:dollars|usd))?{_END}"
)
_RAM = [
    re.compile(rf"(?:at least\s*|min(?:imum)?\s*(?:of\s*)?)?{_NUM}gb\s*(?:of\s*)?ram{_END}"),
    re.compile(rf"ram\s*(?:of\s*)?(?:at least\s*)?{_NUM}gb{_END}"),
]
_STORAGE = re.compile(
    rf"(?:at least\s*)?{_NUM}(gb|tb)(?:\s*(?:of\s*)?(?:storage|rom|internal(?: storage)?))?{_END}"
)
_BATTERY = re.compile(rf"(?:at least\s*|over\s*)?{_NUM}mah(?:\s*battery)?{_END}")
_SCREEN = re.compile(
    rf"(at least|over|above|under|below|at most|up to)?\s*{_NUM}in(?:\s*(?:screen|display))?{_END}"
)
_COMPACT = re.compile(r"\b(compact|small)(?:\s*(?:sized?|screen|display))?\b")
_5G = re.compile(r"\b5g\b")
_ANDROID = re.compile(r"\bandroid\b")
_IOS = re.compile(r"\b(ios)\b")
_IPHONE = re.compile(r"\biphone\b")
_FILLER = {
    "with",
    "and",
    "a",
    "an",
    "the",
    "of",
    "for",
    "that",
    "has",
    "having",
    "in",
    "at",
    "least",
    "phone's",
    "or",
}


def _leaf_node(field: str, op: str, value: Any) -> dict[str, Any]:
    if isinstance(value, float) and value.is_integer() and field in ("battery_mah",):
        value = int(value)
    return {"field": field, "op": op, "value": value}


def parse_constraints(query_text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse spec constraints from free text: ``(filter_json | None, residual_text)``.

    The residual is the normalized query with the matched phrases (and dangling filler
    words) removed; it goes to the ranker. See the module docstring for the patterns.
    """
    cfg = filters_config()
    compact_max = float(cfg.get("compact_max_screen_in", 6.2))
    tol = float(cfg.get("screen_tolerance_in", 0.2))
    small_ram = float(cfg.get("small_ram_gb", 16))
    text = Analyzer.normalize(query_text)
    leaves: list[dict[str, Any]] = []
    seen: set[str] = set()

    def cut(m: re.Match[str]) -> None:
        nonlocal text
        text = text[: m.start()] + " " + text[m.end() :]

    def add(field: str, op: str, value: Any) -> None:
        leaves.append(_leaf_node(field, op, value))
        seen.add(field)

    for rx in _PRICE_RANGE:
        m = rx.search(text)
        if m:
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            add("price", ">=", lo)
            leaves.append(_leaf_node("price", "<=", hi))
            cut(m)
            break
    if "price" not in seen and (m := _PRICE_MAX.search(text)):
        op = "<=" if m.group(1) in ("up to", "upto", "max", "maximum") else "<"
        add("price", op, float(m.group(2)))
        cut(m)
    if "price" not in seen and (m := _PRICE_MIN.search(text)):
        op = ">=" if m.group(1) in ("at least", "min", "minimum") else ">"
        add("price", op, float(m.group(2)))
        cut(m)
    for rx in _RAM:
        if m := rx.search(text):
            add("ram_gb", ">=", float(m.group(1)))
            cut(m)
            break
    spans = []
    for m in _STORAGE.finditer(text):
        gb = float(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
        if gb < small_ram and m.group(2) == "gb" and "storage" not in m.group(0):
            if "ram_gb" not in seen:
                add("ram_gb", ">=", gb)
                spans.append(m.span())
        elif "storage_gb" not in seen:
            add("storage_gb", ">=", gb)
            spans.append(m.span())
    for a, b in sorted(spans, reverse=True):  # cut from the end so offsets stay valid
        text = text[:a] + " " + text[b:]
    if m := _BATTERY.search(text):
        add("battery_mah", ">=", float(m.group(1)))
        cut(m)
    if m := _SCREEN.search(text):
        x, qual = float(m.group(2)), m.group(1)
        if qual in ("at least", "over", "above"):
            add("screen_in", ">=", x)
        elif qual in ("under", "below", "at most", "up to"):
            add("screen_in", "<=", x)
        else:
            add("screen_in", ">=", round(x - tol, 4))
            leaves.append(_leaf_node("screen_in", "<=", round(x + tol, 4)))
        cut(m)
    elif m := _COMPACT.search(text):
        add("screen_in", "<=", compact_max)
        cut(m)
    if m := _5G.search(text):
        add("has_5g", "==", True)
        text = _5G.sub(" ", text)
    if _ANDROID.search(text):
        add("os", "==", "android")
        text = _ANDROID.sub(" ", text)
    elif _IOS.search(text) or _IPHONE.search(text):
        add("os", "==", "ios")
        text = _IOS.sub(" ", text)

    residual = " ".join(w for w in text.replace("$", " ").split() if w not in _FILLER)
    if not leaves:
        return None, residual
    filt = leaves[0] if len(leaves) == 1 else {"and": leaves}
    validate_filter(filt)
    return filt, residual


def resolve_filter(query: Query) -> tuple[dict[str, Any] | None, str, str]:
    """``(filter, text_for_ranker, source)``; source = queries.jsonl | parser | none."""
    if query.filters is not None:
        return query.filters, query.text, "queries.jsonl"
    filt, residual = parse_constraints(query.text)
    if filt is None:
        return None, query.text, "none"
    return filt, residual or query.text, "parser"


# ------------------------------------------------------------------ wrapper


class FilteredRetriever(Retriever):
    """Restrict any retriever to the phones that pass the query's filter.

    ``base`` must already be fitted (or call :meth:`fit`). ``phones`` provides the spec values;
    in ``pre`` mode ``base`` needs an ``index`` and ``score_text(text, k, allowed=...)``.
    After searching, :attr:`log` holds per query the filter, its source, the ranker text and
    the number of allowed docs (see :meth:`write_log`).
    """

    def __init__(
        self,
        base: Retriever,
        phones: Iterable[Phone] | None = None,
        mode: str = "pre",
        post_depth: int | None = None,
        resolver: Callable[[Query], tuple[dict | None, str, str]] = resolve_filter,
        name: str | None = None,
    ) -> None:
        if mode not in ("pre", "post"):
            raise ValueError(f"mode must be 'pre' or 'post', got {mode!r}")
        if mode == "pre" and not hasattr(base, "score_text"):
            raise TypeError(f"{type(base).__name__} cannot pre-filter; use mode='post'")
        self.base, self.mode = base, mode
        self.post_depth = int(post_depth or filters_config().get("post_depth", 1000))
        self.resolver = resolver
        self._name = name
        self.phones: list[Phone] = list(phones) if phones is not None else []
        self.log: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, set[str]] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name or f"{self.base.name}+filters"

    def fit(self, phones: list[Phone]) -> None:
        self.phones = list(phones)
        self._cache.clear()
        self.base.fit(phones)

    def _allowed(self, filt: dict[str, Any]) -> set[str]:
        key = json.dumps(filt, sort_keys=True)
        if key not in self._cache:
            self._cache[key] = allowed_docs(self.phones, filt)
        return self._cache[key]

    def search(self, query: Query, k: int = 100) -> Ranking:
        filt, text, source = self.resolver(query)
        entry: dict[str, Any] = {"source": source, "filter": filt, "ranker_text": text}
        self.log[query.qid] = entry
        q = replace(query, text=text, filters=None)
        if filt is None:
            entry["n_allowed"] = len(self.phones)
            return list(self.base.search(q, k))
        if not self.phones:
            raise RuntimeError("FilteredRetriever needs phones (constructor or fit())")
        allowed = self._allowed(filt)
        entry["n_allowed"] = len(allowed)
        if self.mode == "pre":
            index: InvertedIndex = self.base.index  # type: ignore[attr-defined]
            ids = {index.doc_index(d) for d in allowed}
            return list(self.base.score_text(text, k, allowed=ids))  # type: ignore[attr-defined]
        out = [(d, s) for d, s in self.base.search(q, max(self.post_depth, k)) if d in allowed]
        return out[:k]

    def write_log(self, path: str | Path) -> Path:
        """Write :attr:`log` as JSON (e.g. ``runs/<run_name>.filters.json``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.log, indent=2) + "\n", encoding="utf-8")
        return path
