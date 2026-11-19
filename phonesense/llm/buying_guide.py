"""Buying-guide generator: a short, grounded summary of the top results (owner: B, T23).

Given the shopper's search and the top phones, an LLM writes a ~150-word guide: best fit,
runner-up, trade-offs, what is unknown. Because an LLM can invent specs, the guide is built to
be checkable:

* the model sees **only** a facts block we build from the phone records (title, brand, price,
  rating, the specs that are known, a few feature lines) and is told to use nothing else;
* phones are referred to by short tags ``[P1]``..``[Pn]``; we map them back to titles and drop
  (and report) any tag that is not in the list;
* every number with a unit that the guide states (``5000 mAh``, ``$299``, ``128 GB``) must also
  appear in the facts block; figures that do not are returned as ``warnings``, for the UI to show.

The check catches invented figures. It does NOT check which phone a figure belongs to (a real
price attributed to the wrong phone passes) nor judgements ("best camera" is opinion), so the UI
labels the guide as machine-written and asks the reader to check figures. Calls go through
:class:`~phonesense.llm.client.LLMClient` (purpose ``guide``): logged, and replayed when the
same request repeats.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from phonesense.llm.client import LLMClient
from phonesense.schemas import Phone

PURPOSE = "guide"
TAG_RE = re.compile(r"\[P(\d+)\]")
# a number followed by a unit we care about, or a price with a leading currency sign
FIGURE_RE = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\d[\d,]*(?:\.\d+)?\s?(?:mah|gb|tb|mp|inch(?:es)?|in\b|hz|usd|w\b|stars?))",
    flags=re.IGNORECASE,
)

PROMPT = """\
You are a careful shopping assistant for smartphones.

A shopper searched for: "{query}"

These are the top results. Use ONLY the facts below; do not add specs, prices or features \
that are not listed, and say "unknown" for anything marked unknown.

{facts}

Write a short buying guide of about 120-150 words in plain prose, no headings, no lists:
1. Say which phone fits the search best and why.
2. Name one runner-up and what it does better or worse.
3. Warn about one trade-off or something unknown that could matter to this shopper.
Refer to phones only by their tag, for example [P2], and quote figures exactly as listed.
"""


def _num(x: float) -> str:
    """8.0 -> '8', 6.5 -> '6.5' (so facts and guide agree on how a figure is written)."""
    return f"{x:g}"


# spec key -> (label, how to write a known value); unknown specs are listed separately by label
SPEC_FORMATS = {
    "ram_gb": ("RAM", lambda v: f"{_num(v)} GB"),
    "storage_gb": ("storage", lambda v: f"{_num(v)} GB"),
    "battery_mah": ("battery", lambda v: f"{v} mAh"),
    "screen_in": ("screen", lambda v: f"{_num(v)} inch"),
    "main_camera_mp": ("main camera", lambda v: f"{_num(v)} MP"),
    "os": ("OS", str),
    "has_5g": ("5G", lambda v: "yes" if v else "no"),
    "release_year": ("release year", str),
}


def phone_facts(tag: str, phone: Phone, max_features: int = 4) -> str:
    """The facts block entry for one phone: known values only, unknowns named explicitly."""
    if phone.price is None:
        price = "unknown"
    elif phone.currency == "USD":
        price = f"${phone.price:,.2f}"
    else:
        price = f"{phone.price:,.2f} {phone.currency}"
    rating = ("no ratings" if phone.avg_rating is None
              else f"{phone.avg_rating:.1f} stars from {phone.rating_count:,} ratings")
    specs = phone.specs.to_dict()
    known = [f"{SPEC_FORMATS[k][0]} {SPEC_FORMATS[k][1](v)}" for k, v in specs.items()
             if v is not None]
    unknown = [SPEC_FORMATS[k][0] for k, v in specs.items() if v is None]
    lines = [
        f"[{tag}] {phone.title}",
        f"  brand: {phone.brand or 'unknown'}; price: {price}; rating: {rating}",
        f"  known specs: {', '.join(known) if known else 'none'}",
        f"  unknown specs: {', '.join(unknown) if unknown else 'none'}",
    ]
    for feature in [f.strip() for f in phone.fields.features if f.strip()][:max_features]:
        lines.append(f"  - {feature[:150]}")
    return "\n".join(lines)


def build_prompt(query_text: str, phones: Sequence[Phone]) -> tuple[str, str]:
    """Return ``(prompt, facts)``; ``facts`` is what the figure check compares against."""
    if not phones:
        raise ValueError("no phones to write a guide about")
    text = " ".join(query_text.split()).replace('"', "'")
    if not text:
        raise ValueError("query text is empty")
    facts = "\n\n".join(phone_facts(f"P{i}", p) for i, p in enumerate(phones, start=1))
    return PROMPT.format(query=text, facts=facts), facts


def _norm(s: str) -> str:
    """Lower-case, drop spaces and thousands commas, and read '8.0' as '8'."""
    return re.sub(r"(\d)\.0+(?!\d)", r"\1", re.sub(r"[\s,]", "", s.lower()))


def unverified_figures(guide: str, facts: str) -> list[str]:
    """Figures with units in ``guide`` that do not occur in ``facts`` (ignoring spaces, commas
    and case), in order of appearance without repeats."""
    haystack = _norm(facts)
    # the tags themselves look like "P1": remove them so they are never read as figures
    body = TAG_RE.sub(" ", guide)
    seen: list[str] = []
    for match in FIGURE_RE.finditer(body):
        fig = match.group(0).strip()
        if _norm(fig) not in haystack and fig not in seen:
            seen.append(fig)
    return seen


@dataclass(frozen=True)
class GuideResult:
    """The guide text (tags replaced by bold titles), the phones it mentions, and any problems."""

    text: str
    doc_ids: list[str]
    warnings: list[str] = field(default_factory=list)


def clean_text(raw: str) -> str:
    """Remove code fences and markdown markers the model sometimes adds."""
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", raw, flags=re.MULTILINE)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text.replace("**", "")).strip()


class BuyingGuide:
    """Writes the guide for the top ``max_phones`` results of a search."""

    def __init__(self, client: LLMClient, max_phones: int = 5) -> None:
        if max_phones < 1:
            raise ValueError("max_phones must be >= 1")
        self.client = client
        self.max_phones = max_phones

    def write(self, query_text: str, phones: Sequence[Phone]) -> GuideResult:
        chosen = list(phones)[: self.max_phones]
        prompt, facts = build_prompt(query_text, chosen)
        text = clean_text(self.client.complete(prompt, purpose=PURPOSE))
        warnings: list[str] = []
        cited: list[str] = []
        bad_tags: list[str] = []

        def swap(match: re.Match) -> str:
            n = int(match.group(1))
            if 1 <= n <= len(chosen):
                phone = chosen[n - 1]
                if phone.doc_id not in cited:
                    cited.append(phone.doc_id)
                return f"**{phone.title}**"
            bad_tags.append(match.group(0))
            return ""

        out = TAG_RE.sub(swap, text)
        out = re.sub(r"[ \t]{2,}", " ", out).strip()
        if bad_tags:
            warnings.append(f"Removed references to phones that were not listed: {bad_tags}")
        extra = unverified_figures(text, facts)
        if extra:
            warnings.append("Figures not found in the phone data (check before trusting): "
                            + ", ".join(extra))
        if not cited:
            warnings.append("The guide does not name any of the listed phones.")
        return GuideResult(out, cited, warnings)
