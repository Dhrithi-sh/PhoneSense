"""LLM relevance judge: grades one (query, phone) pair 0-3 (owner: B, T09).

Grades follow CONTRACT.md: 0 not relevant, 1 marginal, 2 relevant, 3 perfect; for binary
metrics a grade of 2 or more counts as relevant. The judge sees the query, its labelling notes
and hard filters (CONTRACT 4.3/4.4), and a compact view of the phone: title, brand, price,
parsed specs, feature bullets and the start of the description. It does *not* see reviews,
ratings, or which system retrieved the phone, so a system cannot influence its own label.

Judgments are only as good as their agreement with people. Round 2 (T09b) measures Cohen's
kappa between this judge and human labels on a shared sample; until then treat the labels as
unvalidated.

Calls go through :class:`~phonesense.llm.client.LLMClient` (purpose ``judge``), so every
prompt and answer is logged and re-running is free and identical.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from phonesense.llm.client import LLMClient
from phonesense.schemas import VALID_GRADES, Phone, Query

PURPOSE = "judge"

RUBRIC = """\
You are a careful relevance assessor for a smartphone search engine.

Shopper's search: "{query}"
{notes}{filters}
Candidate product:
{phone}

Grade how well this product answers the search:
3 = perfect: clearly what the shopper wants, and every stated requirement is met
2 = relevant: a good answer; minor requirements are unverified or only roughly met
1 = marginal: related (same kind of phone, or only one requirement met) but a poor answer
0 = not relevant: not a phone (case, charger, screen protector, cable, accessory), the wrong \
model, or it clearly breaks a stated requirement

Rules: judge only from the product text above. A spec shown as unknown cannot be assumed to \
satisfy a requirement. Do not reward a product for being popular or expensive.

Answer with JSON only, no other text: {{"grade": <0|1|2|3>, "reason": "<one short sentence>"}}
"""

RETRY_SUFFIX = '\nYour last answer was not valid. Reply with exactly: {"grade": N, "reason": "..."}'


class JudgeError(ValueError):
    """The model's answer could not be parsed into a grade."""


@dataclass(frozen=True)
class Judgment:
    qid: str
    doc_id: str
    grade: int
    reason: str


def _query_notes(query: Query) -> str:
    notes = re.sub(r"^type=\w+\.\s*", "", query.intent_notes or "").strip()
    return f"Labeller notes: {notes}\n" if notes else ""


def _query_filters(query: Query) -> str:
    if not query.filters:
        return ""
    return f"Hard requirements (as JSON): {json.dumps(query.filters, sort_keys=True)}\n"


def phone_summary(phone: Phone, max_desc_chars: int = 600, max_features: int = 8) -> str:
    """Compact text view of a phone for the judge (no reviews, no ratings)."""
    specs = ", ".join(
        f"{k}={'unknown' if v is None else v}" for k, v in phone.specs.to_dict().items()
    )
    price = "unknown" if phone.price is None else f"{phone.price:g} {phone.currency}"
    lines = [
        f"Title: {phone.title}",
        f"Brand: {phone.brand or 'unknown'}",
        f"Price: {price}",
        f"Specs: {specs}",
    ]
    features = [f for f in phone.fields.features if f.strip()][:max_features]
    if features:
        lines.append("Features:")
        lines += [f"- {f.strip()[:200]}" for f in features]
    desc = " ".join(phone.fields.description.split())
    if desc:
        lines.append(f"Description: {desc[:max_desc_chars]}")
    return "\n".join(lines)


def build_prompt(query: Query, phone: Phone) -> str:
    """The exact prompt sent to the LLM (and logged) for this pair."""
    text = " ".join(query.text.split()).replace('"', "'")
    return RUBRIC.format(
        query=text, notes=_query_notes(query), filters=_query_filters(query),
        phone=phone_summary(phone),
    )


def parse_grade(response: str) -> tuple[int, str]:
    """Extract ``(grade, reason)`` from the model's answer; raise :class:`JudgeError` if unusable.

    Accepts a JSON object (possibly wrapped in text or a code fence). Falls back to a
    ``"grade": N`` pattern. A grade outside 0-3, or a non-integer, is an error, never rounded.
    """
    text = response.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "grade" in obj:
            grade = obj["grade"]
            if isinstance(grade, bool) or not isinstance(grade, int) or grade not in VALID_GRADES:
                raise JudgeError(f"grade must be one of {VALID_GRADES}, got {grade!r}")
            return grade, str(obj.get("reason", "")).strip()
    loose = re.search(r'"?grade"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b', text, flags=re.IGNORECASE)
    if loose and loose.group(1) in {str(g) for g in VALID_GRADES}:
        return int(loose.group(1)), ""
    raise JudgeError(f"no valid grade in answer: {text[:120]!r}")


class LLMJudge:
    """Grades (query, phone) pairs with an :class:`LLMClient`; re-asks once on a bad answer."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def judge(self, query: Query, phone: Phone) -> Judgment:
        prompt = build_prompt(query, phone)
        try:
            grade, reason = parse_grade(self.client.complete(prompt, purpose=PURPOSE))
        except JudgeError:
            retry = self.client.complete(prompt + RETRY_SUFFIX, purpose=PURPOSE)
            grade, reason = parse_grade(retry)
        return Judgment(query.qid, phone.doc_id, grade, reason)
