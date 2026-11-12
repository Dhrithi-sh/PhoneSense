"""Dataclasses and validators for every shared record type.

This module is the executable form of CONTRACT.md. If you change one, change the other
(plus tests/fixtures/) in the same PR.

Every record type has:
    Record.from_dict(d)  -> validated instance (raises SchemaError on any violation)
    record.to_dict()     -> plain dict, ready for JSON
TSV/TREC record types also have ``from_fields`` / ``to_line`` helpers used by phonesense.io.

Validation is strict: wrong types are rejected (no silent coercion of "8" -> 8, and booleans
are never accepted as numbers), unknown keys are rejected, and unknown values must be
``null`` (None), not "", "N/A" or 0.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------------------
# Constants (mirrored in CONTRACT.md)
# --------------------------------------------------------------------------------------

MAX_RUN_DEPTH = 100  # max docs per query in a run file
RELEVANCE_THRESHOLD = 2  # binary metrics: grade >= 2 is relevant
VALID_GRADES = (0, 1, 2, 3)
SPLITS = ("dev", "test")
RUN_SPLITS = ("dev", "test", "all")  # "all" = runs over both splits (e.g. pooling)
OS_VALUES = ("android", "ios", "other")
MIN_RELEASE_YEAR, MAX_RELEASE_YEAR = 2000, 2030
# Timestamps are milliseconds since the Unix epoch; anything below this is almost
# certainly seconds by mistake (1e11 ms ~= March 1973).
MIN_TIMESTAMP_MS = 100_000_000_000

QID_RE = re.compile(r"^q\d{2}$")
TOKEN_RE = re.compile(r"^\S+$")  # ids used in whitespace-separated files
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+\-_]*_(dev|test|all)$")


class SchemaError(ValueError):
    """Raised when a record does not satisfy the contract."""


# --------------------------------------------------------------------------------------
# Primitive checkers
# --------------------------------------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (isinstance(v, int | float)) and not isinstance(v, bool)


def _req_str(d: dict, key: str, where: str, *, nonempty: bool = True) -> str:
    v = d[key]
    if not isinstance(v, str):
        raise SchemaError(f"{where}.{key}: expected str, got {type(v).__name__}")
    if nonempty and not v.strip():
        raise SchemaError(f"{where}.{key}: must be a non-empty string")
    return v


def _opt_str(d: dict, key: str, where: str) -> str | None:
    v = d[key]
    if v is None:
        return None
    if not isinstance(v, str):
        raise SchemaError(f"{where}.{key}: expected str or null, got {type(v).__name__}")
    if not v.strip():
        raise SchemaError(f"{where}.{key}: use null for unknown, not an empty string")
    return v


def _token(v: Any, where: str) -> str:
    if not isinstance(v, str) or not TOKEN_RE.match(v):
        raise SchemaError(f"{where}: expected a non-empty string without whitespace, got {v!r}")
    return v


def _req_int(d: dict, key: str, where: str, *, lo: int | None = None, hi: int | None = None):
    v = d[key]
    if not _is_int(v):
        raise SchemaError(f"{where}.{key}: expected int, got {type(v).__name__}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise SchemaError(f"{where}.{key}: {v} out of range [{lo}, {hi}]")
    return v


def _finite(v: Any, where: str) -> float:
    if not _is_number(v):
        raise SchemaError(f"{where}: expected a number, got {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise SchemaError(f"{where}: must be finite, got {v!r}")
    return f


def _opt_number(
    d: dict, key: str, where: str, *, lo: float | None = None, hi: float | None = None,
    positive: bool = False,
) -> float | None:
    v = d[key]
    if v is None:
        return None
    f = _finite(v, f"{where}.{key}")
    if positive and f <= 0:
        raise SchemaError(f"{where}.{key}: must be > 0 (use null for unknown), got {v!r}")
    if (lo is not None and f < lo) or (hi is not None and f > hi):
        raise SchemaError(f"{where}.{key}: {v!r} out of range [{lo}, {hi}]")
    return f


def _opt_int(
    d: dict, key: str, where: str, *, lo: int | None = None, hi: int | None = None,
    positive: bool = False,
) -> int | None:
    v = d[key]
    if v is None:
        return None
    if not _is_int(v):
        raise SchemaError(f"{where}.{key}: expected int or null, got {type(v).__name__}")
    if positive and v <= 0:
        raise SchemaError(f"{where}.{key}: must be > 0 (use null for unknown), got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise SchemaError(f"{where}.{key}: {v} out of range [{lo}, {hi}]")
    return v


def _req_bool(d: dict, key: str, where: str) -> bool:
    v = d[key]
    if not isinstance(v, bool):
        raise SchemaError(f"{where}.{key}: expected bool, got {type(v).__name__}")
    return v


def _check_keys(d: Any, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected an object, got {type(d).__name__}")
    missing = required - d.keys()
    if missing:
        raise SchemaError(f"{where}: missing keys {sorted(missing)} (use null for unknown)")
    extra = d.keys() - required - optional
    if extra:
        raise SchemaError(f"{where}: unknown keys {sorted(extra)}")


def validate_qid(qid: Any, where: str = "qid") -> str:
    if not isinstance(qid, str) or not QID_RE.match(qid):
        raise SchemaError(f"{where}: expected 'qNN' (e.g. 'q01'), got {qid!r}")
    return qid


def validate_run_name(name: Any, where: str = "run_name") -> str:
    if not isinstance(name, str) or not RUN_NAME_RE.match(name):
        raise SchemaError(
            f"{where}: expected '<method>[_<variant>]_<dev|test|all>' with no whitespace, "
            f"got {name!r}"
        )
    return name


def run_name_split(name: str) -> str:
    """'bm25_k1.2_b0.75_dev' -> 'dev'."""
    return validate_run_name(name).rsplit("_", 1)[1]


# --------------------------------------------------------------------------------------
# 4.1 Phones
# --------------------------------------------------------------------------------------

SPEC_FIELDS = (
    "ram_gb", "storage_gb", "battery_mah", "screen_in",
    "main_camera_mp", "os", "has_5g", "release_year",
)


@dataclass(frozen=True)
class Specs:
    ram_gb: float | None = None
    storage_gb: float | None = None
    battery_mah: int | None = None
    screen_in: float | None = None
    main_camera_mp: float | None = None
    os: str | None = None
    has_5g: bool | None = None
    release_year: int | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "specs") -> Specs:
        _check_keys(d, set(SPEC_FIELDS), set(), where)
        os_v = d["os"]
        if os_v is not None and os_v not in OS_VALUES:
            raise SchemaError(f"{where}.os: expected one of {OS_VALUES} or null, got {os_v!r}")
        has_5g = d["has_5g"]
        if has_5g is not None and not isinstance(has_5g, bool):
            raise SchemaError(f"{where}.has_5g: expected bool or null, got {has_5g!r}")
        return cls(
            ram_gb=_opt_number(d, "ram_gb", where, positive=True, hi=64),
            storage_gb=_opt_number(d, "storage_gb", where, positive=True, hi=4096),
            battery_mah=_opt_int(d, "battery_mah", where, positive=True, hi=20000),
            screen_in=_opt_number(d, "screen_in", where, positive=True, hi=12),
            main_camera_mp=_opt_number(d, "main_camera_mp", where, positive=True, hi=400),
            os=os_v,
            has_5g=has_5g,
            release_year=_opt_int(
                d, "release_year", where, lo=MIN_RELEASE_YEAR, hi=MAX_RELEASE_YEAR
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhoneFields:
    title: str
    features: list[str]
    description: str

    @classmethod
    def from_dict(cls, d: Any, where: str = "fields") -> PhoneFields:
        _check_keys(d, {"title", "features", "description"}, set(), where)
        feats = d["features"]
        if not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
            raise SchemaError(f"{where}.features: expected a list of strings")
        return cls(
            title=_req_str(d, "title", where),
            features=list(feats),
            description=_req_str(d, "description", where, nonempty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "features": list(self.features),
                "description": self.description}


def build_text(fields: PhoneFields) -> str:
    """Canonical searchable text: title, each feature, description, joined by newlines."""
    parts = [fields.title, *fields.features, fields.description]
    return "\n".join(p for p in parts if p.strip())


PHONE_KEYS = {
    "doc_id", "title", "brand", "price", "currency", "avg_rating", "rating_count",
    "specs", "text", "fields",
}


@dataclass(frozen=True)
class Phone:
    doc_id: str
    title: str
    brand: str | None
    price: float | None
    currency: str
    avg_rating: float | None
    rating_count: int
    specs: Specs
    text: str
    fields: PhoneFields

    @classmethod
    def from_dict(cls, d: Any, where: str = "phone") -> Phone:
        _check_keys(d, PHONE_KEYS, set(), where)
        currency = _req_str(d, "currency", where)
        if not CURRENCY_RE.match(currency):
            raise SchemaError(f"{where}.currency: expected ISO 4217 code like 'USD'")
        return cls(
            doc_id=_token(d["doc_id"], f"{where}.doc_id"),
            title=_req_str(d, "title", where),
            brand=_opt_str(d, "brand", where),
            price=_opt_number(d, "price", where, positive=True),
            currency=currency,
            avg_rating=_opt_number(d, "avg_rating", where, lo=1.0, hi=5.0),
            rating_count=_req_int(d, "rating_count", where, lo=0),
            specs=Specs.from_dict(d["specs"], f"{where}.specs"),
            text=_req_str(d, "text", where),
            fields=PhoneFields.from_dict(d["fields"], f"{where}.fields"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "title": self.title, "brand": self.brand,
            "price": self.price, "currency": self.currency, "avg_rating": self.avg_rating,
            "rating_count": self.rating_count, "specs": self.specs.to_dict(),
            "text": self.text, "fields": self.fields.to_dict(),
        }


# --------------------------------------------------------------------------------------
# 4.2 Reviews
# --------------------------------------------------------------------------------------


def make_review_id(user_id: str, doc_id: str, timestamp: int) -> str:
    """Deterministic review id: first 16 hex chars of sha1('user_id|doc_id|timestamp')."""
    return hashlib.sha1(f"{user_id}|{doc_id}|{timestamp}".encode()).hexdigest()[:16]


REVIEW_KEYS = {"review_id", "doc_id", "user_id", "rating", "title", "text", "timestamp",
               "helpful_votes", "verified"}


@dataclass(frozen=True)
class Review:
    review_id: str
    doc_id: str
    user_id: str
    rating: int
    title: str
    text: str
    timestamp: int
    helpful_votes: int
    verified: bool
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, d: Any, where: str = "review") -> Review:
        _check_keys(d, REVIEW_KEYS, {"is_synthetic"}, where)
        is_syn = d.get("is_synthetic", False)
        if not isinstance(is_syn, bool):
            raise SchemaError(f"{where}.is_synthetic: expected bool")
        r = cls(
            review_id=_token(d["review_id"], f"{where}.review_id"),
            doc_id=_token(d["doc_id"], f"{where}.doc_id"),
            user_id=_token(d["user_id"], f"{where}.user_id"),
            rating=_req_int(d, "rating", where, lo=1, hi=5),
            title=_req_str(d, "title", where, nonempty=False),
            text=_req_str(d, "text", where, nonempty=False),
            timestamp=_req_int(d, "timestamp", where, lo=MIN_TIMESTAMP_MS),
            helpful_votes=_req_int(d, "helpful_votes", where, lo=0),
            verified=_req_bool(d, "verified", where),
            is_synthetic=is_syn,
        )
        expected = make_review_id(r.user_id, r.doc_id, r.timestamp)
        if r.review_id != expected:
            raise SchemaError(
                f"{where}.review_id: {r.review_id!r} != make_review_id(...) = {expected!r}"
            )
        return r

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# 4.4 Boolean filters
# --------------------------------------------------------------------------------------

# field -> value kind. Phone top-level fields plus every spec field.
FILTER_FIELDS: dict[str, str] = {
    "price": "number",
    "avg_rating": "number",
    "rating_count": "number",
    "brand": "string",
    "ram_gb": "number",
    "storage_gb": "number",
    "battery_mah": "number",
    "screen_in": "number",
    "main_camera_mp": "number",
    "release_year": "number",
    "os": "string",
    "has_5g": "bool",
}
FILTER_OPS = ("==", "!=", "<", "<=", ">", ">=", "in")
_OPS_BY_KIND = {
    "number": set(FILTER_OPS),
    "string": {"==", "!=", "in"},
    "bool": {"==", "!="},
}


def _check_filter_value(kind: str, v: Any, where: str) -> None:
    ok = {
        "number": _is_number(v) and math.isfinite(float(v)),
        "string": isinstance(v, str) and bool(v),
        "bool": isinstance(v, bool),
    }[kind]
    if not ok:
        raise SchemaError(f"{where}: value {v!r} is not a valid {kind}")


def validate_filter(f: Any, where: str = "filters") -> dict:
    """Validate a Boolean filter expression (CONTRACT.md 4.4). Returns it unchanged."""
    if not isinstance(f, dict):
        raise SchemaError(f"{where}: expected an object, got {type(f).__name__}")
    if "field" in f or "op" in f or "value" in f:
        _check_keys(f, {"field", "op", "value"}, set(), where)
        fld, op, val = f["field"], f["op"], f["value"]
        if fld not in FILTER_FIELDS:
            raise SchemaError(
                f"{where}.field: unknown field {fld!r}; allowed {sorted(FILTER_FIELDS)}"
            )
        if op not in FILTER_OPS:
            raise SchemaError(f"{where}.op: unknown op {op!r}; allowed {FILTER_OPS}")
        kind = FILTER_FIELDS[fld]
        if op not in _OPS_BY_KIND[kind]:
            raise SchemaError(f"{where}: op {op!r} not allowed for {kind} field {fld!r}")
        if op == "in":
            if not isinstance(val, list) or not val:
                raise SchemaError(f"{where}.value: 'in' needs a non-empty list")
            for i, item in enumerate(val):
                _check_filter_value(kind, item, f"{where}.value[{i}]")
        else:
            _check_filter_value(kind, val, f"{where}.value")
        if fld == "os":
            vals = val if isinstance(val, list) else [val]
            bad = [v for v in vals if v not in OS_VALUES]
            if bad:
                raise SchemaError(f"{where}.value: os must be one of {OS_VALUES}, got {bad}")
        return f
    if len(f) != 1:
        raise SchemaError(
            f"{where}: a node must have exactly one of 'and'/'or'/'not', or be a leaf "
            f"with field/op/value; got keys {sorted(f)}"
        )
    (key, val), = f.items()
    if key in ("and", "or"):
        if not isinstance(val, list) or not val:
            raise SchemaError(f"{where}.{key}: expected a non-empty list")
        for i, child in enumerate(val):
            validate_filter(child, f"{where}.{key}[{i}]")
    elif key == "not":
        validate_filter(val, f"{where}.not")
    else:
        raise SchemaError(f"{where}: unknown operator {key!r}; expected and/or/not")
    return f


# --------------------------------------------------------------------------------------
# 4.3 Queries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    split: str
    filters: dict | None = None
    intent_notes: str | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "query") -> Query:
        _check_keys(d, {"qid", "text", "split"}, {"filters", "intent_notes"}, where)
        split = d["split"]
        if split not in SPLITS:
            raise SchemaError(f"{where}.split: expected one of {SPLITS}, got {split!r}")
        filters = d.get("filters")
        if filters is not None:
            validate_filter(filters, f"{where}.filters")
        notes = d.get("intent_notes")
        if notes is not None and not isinstance(notes, str):
            raise SchemaError(f"{where}.intent_notes: expected str or null")
        return cls(
            qid=validate_qid(d["qid"], f"{where}.qid"),
            text=_req_str(d, "text", where),
            split=split,
            filters=filters,
            intent_notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# 4.5 Qrels
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Qrel:
    qid: str
    doc_id: str
    grade: int

    @classmethod
    def from_fields(cls, parts: list[str], where: str = "qrel") -> Qrel:
        if len(parts) != 4:
            raise SchemaError(f"{where}: expected 4 fields 'qid 0 doc_id grade', got {len(parts)}")
        qid, it, doc_id, grade_s = parts
        if it != "0":
            raise SchemaError(f"{where}: second field must be '0', got {it!r}")
        if grade_s not in {str(g) for g in VALID_GRADES}:
            raise SchemaError(f"{where}: grade must be one of {VALID_GRADES}, got {grade_s!r}")
        return cls(validate_qid(qid, f"{where}.qid"), _token(doc_id, f"{where}.doc_id"),
                   int(grade_s))

    def validate(self) -> Qrel:
        validate_qid(self.qid)
        _token(self.doc_id, "qrel.doc_id")
        if not _is_int(self.grade) or self.grade not in VALID_GRADES:
            raise SchemaError(f"qrel.grade: must be one of {VALID_GRADES}, got {self.grade!r}")
        return self

    def to_line(self) -> str:
        return f"{self.qid} 0 {self.doc_id} {self.grade}"

    @property
    def relevant(self) -> bool:
        return self.grade >= RELEVANCE_THRESHOLD


# --------------------------------------------------------------------------------------
# 4.6 Runs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEntry:
    qid: str
    doc_id: str
    rank: int
    score: float
    run_name: str

    @classmethod
    def from_fields(cls, parts: list[str], where: str = "run") -> RunEntry:
        if len(parts) != 6:
            raise SchemaError(
                f"{where}: expected 6 fields 'qid Q0 doc_id rank score run_name', got {len(parts)}"
            )
        qid, q0, doc_id, rank_s, score_s, name = parts
        if q0 != "Q0":
            raise SchemaError(f"{where}: second field must be 'Q0', got {q0!r}")
        try:
            rank = int(rank_s)
            score = float(score_s)
        except ValueError as e:
            raise SchemaError(f"{where}: bad rank/score: {e}") from None
        return cls(qid, doc_id, rank, score, name).validate(where)

    def validate(self, where: str = "run") -> RunEntry:
        validate_qid(self.qid, f"{where}.qid")
        _token(self.doc_id, f"{where}.doc_id")
        if not _is_int(self.rank) or self.rank < 1:
            raise SchemaError(f"{where}.rank: must be an int >= 1, got {self.rank!r}")
        _finite(self.score, f"{where}.score")
        validate_run_name(self.run_name, f"{where}.run_name")
        return self

    def to_line(self) -> str:
        # repr(float) is the shortest string that round-trips exactly.
        return (f"{self.qid} Q0 {self.doc_id} {self.rank} {float(self.score)!r} "
                f"{self.run_name}")


def validate_run(entries: list[RunEntry]) -> None:
    """Whole-run checks (CONTRACT.md 4.6). Entries must be grouped by query, rank order."""
    names = {e.run_name for e in entries}
    if len(names) > 1:
        raise SchemaError(f"run: all lines must share one run_name, got {sorted(names)}")
    seen_qids: set[str] = set()
    prev_qid: str | None = None
    for e in entries:
        e.validate()
        if e.qid != prev_qid:
            if e.qid in seen_qids:
                raise SchemaError(f"run: lines for {e.qid} are not contiguous")
            seen_qids.add(e.qid)
            prev_qid, expected_rank, prev_score, docs = e.qid, 1, math.inf, set()
        if e.rank != expected_rank:
            raise SchemaError(
                f"run: {e.qid} rank {e.rank} where {expected_rank} expected "
                "(ranks start at 1 with no gaps)"
            )
        if e.rank > MAX_RUN_DEPTH:
            raise SchemaError(f"run: {e.qid} has more than {MAX_RUN_DEPTH} docs")
        if e.doc_id in docs:
            raise SchemaError(f"run: duplicate doc_id {e.doc_id} for {e.qid}")
        if e.score > prev_score:
            raise SchemaError(
                f"run: {e.qid} scores not sorted descending at rank {e.rank} "
                f"({e.score} > {prev_score})"
            )
        docs.add(e.doc_id)
        expected_rank += 1
        prev_score = e.score


# --------------------------------------------------------------------------------------
# 4.8 Graph files (owner: C, may evolve; paths are stable)
# --------------------------------------------------------------------------------------

GRAPH_EDGES_HEADER = ("src_doc_id", "dst_doc_id", "weight")
REVIEWER_TRUST_HEADER = ("user_id", "trust")


@dataclass(frozen=True)
class GraphEdge:
    src_doc_id: str
    dst_doc_id: str
    weight: float

    def validate(self, where: str = "edge") -> GraphEdge:
        _token(self.src_doc_id, f"{where}.src_doc_id")
        _token(self.dst_doc_id, f"{where}.dst_doc_id")
        if self.src_doc_id == self.dst_doc_id:
            raise SchemaError(f"{where}: self-loop on {self.src_doc_id}")
        if _finite(self.weight, f"{where}.weight") <= 0:
            raise SchemaError(f"{where}.weight: must be > 0, got {self.weight!r}")
        return self


@dataclass(frozen=True)
class ReviewerTrust:
    user_id: str
    trust: float

    def validate(self, where: str = "trust") -> ReviewerTrust:
        _token(self.user_id, f"{where}.user_id")
        t = _finite(self.trust, f"{where}.trust")
        if not 0.0 <= t <= 1.0:
            raise SchemaError(f"{where}.trust: must be in [0, 1], got {self.trust!r}")
        return self


# --------------------------------------------------------------------------------------
# 4.9 Results
# --------------------------------------------------------------------------------------


def _metric_dict(d: Any, where: str) -> dict[str, float]:
    if not isinstance(d, dict) or not d:
        raise SchemaError(f"{where}: expected a non-empty object of metric -> number")
    out = {}
    for k, v in d.items():
        if not isinstance(k, str) or not k:
            raise SchemaError(f"{where}: metric names must be non-empty strings")
        out[k] = _finite(v, f"{where}.{k}")
    return out


@dataclass(frozen=True)
class Results:
    run: str
    qrels: str
    split: str
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any, where: str = "results") -> Results:
        _check_keys(d, {"run", "qrels", "split", "metrics", "per_query"}, {"meta"}, where)
        validate_run_name(d["run"], f"{where}.run")
        split = d["split"]
        if split not in RUN_SPLITS:
            raise SchemaError(f"{where}.split: expected one of {RUN_SPLITS}, got {split!r}")
        if run_name_split(d["run"]) != split:
            raise SchemaError(f"{where}.split: {split!r} does not match run name {d['run']!r}")
        metrics = _metric_dict(d["metrics"], f"{where}.metrics")
        pq = d["per_query"]
        if not isinstance(pq, dict) or not pq:
            raise SchemaError(f"{where}.per_query: required, non-empty object qid -> metrics")
        per_query = {}
        for qid, m in pq.items():
            validate_qid(qid, f"{where}.per_query key")
            per_query[qid] = _metric_dict(m, f"{where}.per_query.{qid}")
            if per_query[qid].keys() != metrics.keys():
                raise SchemaError(
                    f"{where}.per_query.{qid}: metric names must equal those in 'metrics'"
                )
        meta = d.get("meta", {})
        if not isinstance(meta, dict):
            raise SchemaError(f"{where}.meta: expected an object")
        return cls(d["run"], _req_str(d, "qrels", where), split, metrics, per_query, meta)

    def to_dict(self) -> dict[str, Any]:
        d = {"run": self.run, "qrels": self.qrels, "split": self.split,
             "metrics": dict(self.metrics),
             "per_query": {q: dict(m) for q, m in self.per_query.items()}}
        if self.meta:
            d["meta"] = dict(self.meta)
        return d
