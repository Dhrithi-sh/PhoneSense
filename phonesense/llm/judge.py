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
from collections.abc import Sequence
from dataclasses import dataclass

from phonesense.llm.client import LLMClient
from phonesense.schemas import VALID_GRADES, Phone, Query

PURPOSE = "judge"

# The fixed instructions come FIRST and the query and phone LAST, so every prompt starts with the
# same text. Providers that cache prompt prefixes (Groq does, and does not count cached tokens
# against the rate limit) then only charge for the part that changes.
RUBRIC = """\
Rate how well a phone answers a shopper's search, from 0 to 3.

3 = perfect: clearly what they want, every stated requirement is confirmed in the listing
2 = relevant: a good fit; at most minor requirements are unconfirmed or roughly met
1 = marginal: related, but a key requirement is unconfirmed or only partly met
0 = not relevant: not a phone (case, charger, cable, accessory), the wrong model, or the listing \
clearly breaks a requirement (it states the opposite, for example 4G only for a 5G search)
Judge only from the text given. A spec the listing does not state is unconfirmed, not wrong: \
never give 0 for that alone, but do not give 3 either.
Reply with JSON only: {{"grade": <0|1|2|3>, "reason": "<at most 12 words>"}}

Search: "{query}"
{notes}{filters}Phone:
{phone}
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


def phone_summary(phone: Phone, max_desc_chars: int = 300, max_features: int = 5) -> str:
    """Compact text view of a phone for the judge (no reviews, no ratings). Known specs are
    written out; unknown ones are listed by name so the judge knows they are not stated."""
    spec_items = phone.specs.to_dict().items()
    known = ", ".join(f"{k}={v}" for k, v in spec_items if v is not None)
    unknown = ", ".join(k for k, v in spec_items if v is None)
    specs = (known or "none") + (f"; unknown: {unknown}" if unknown else "")
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
        lines += [f"- {f.strip()[:120]}" for f in features]
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


def assign_models(qids: Sequence[str], models: Sequence[str]) -> dict[str, str]:
    """Give every query to ONE judge model (round robin over the sorted query ids).

    Free plans cap tokens per model per day, so the work is split across models. A query is
    never judged by two models, so all systems are compared under the same judge within a query.
    """
    if not models:
        raise ValueError("need at least one model")
    return {qid: models[i % len(models)] for i, qid in enumerate(sorted(set(qids)))}


def interleave(groups: Sequence[Sequence]) -> list:
    """Round-robin merge of lists (a1, b1, a2, b2, ...): consecutive calls then go to different
    models, so each model's per-minute limit is respected while both run in parallel."""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.extend(g[i] for g in groups if i < len(g))
    return out


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
