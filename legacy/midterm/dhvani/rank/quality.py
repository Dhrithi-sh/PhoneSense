"""Page type as a quality signal: push listing pages and horoscopes down.

About 18% of the frozen corpus isn't news articles: section and city pages
the crawler saved as articles ("देवरिया की सबसे ताज़ा खबर", "Guna News, Guna
Samachar", amarujala.com/technology), live-blog index pages, and daily
horoscopes. A listing page stitches together dozens of headlines, so it
contains almost any combination of words and wins far too many queries.

This is Lecture 7's query-independent quality: each article gets a page type,
non-articles have their score multiplied by DEMOTE_FACTOR, and they're placed
after every real article, so they still show up when nothing better matches.
That has to override the query parser's stages: a listing page contains
almost every word somewhere, so it often reaches a stricter stage ("all
words") than the real articles, and demoting it only within its stage left
it on top.

A page is a listing if its URL has no article slug (the site root, /live/
pages, or a short path of at most three parts whose last part has at most
three words and no digits) or its headline looks like one ("की सबसे ताज़ा
खबर", "Samachar", "News in Hindi", "Today Live"). It's a horoscope if the
headline mentions राशिफल, Rashifal, Horoscope or Panchang.

The index doesn't keep URLs, so `python -m dhvani.rank.quality` classifies the
corpus once and writes the non-article ids to dhvani/rank/data/page_types.tsv
(ids only, no text). Without that file, the headline rules alone are used.
"""

import argparse
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

PAGE_TYPES = Path(__file__).resolve().parent / "data" / "page_types.tsv"
CORPUS = Path(__file__).resolve().parents[2] / "data" / "news.jsonl"
DEMOTE_FACTOR = 0.3


def _norm(text):
    return unicodedata.normalize("NFC", text).replace("़", "").lower()   # ताज़ा and ताजा match


_LISTING_HEAD = re.compile(_norm(
    r"सबसे ताजा खबर|ताजा समाचार|मुख्य और ताजा|samachar|news in hindi|hindi news|latest news|"
    r"news today|today live|live updates|breaking news|ब्रेकिंग न्यूज|news, |न्यूज\)"))
_HOROSCOPE_HEAD = re.compile(_norm(r"राशिफल|rashifal|horoscope|panchang|पंचांग"))
_BILINGUAL_TAIL = re.compile(r"\s-\s[a-z]")   # Jagran articles end with "- english words"; listings don't


def listing_url(url):
    path = urlparse(url or "").path.strip("/")
    if not path or "/live/" in f"/{path}/":
        return True
    parts = path.split("/")
    return not re.search(r"\d", path) and len(parts[-1].split("-")) <= 3 and len(parts) <= 3


def classify(headline, url=None):
    """'horoscope', 'listing' or 'article'."""
    head = _norm(headline or "")
    if _HOROSCOPE_HEAD.search(head):
        return "horoscope"
    if _BILINGUAL_TAIL.search(headline or ""):
        return "article"
    if _LISTING_HEAD.search(head) or (url is not None and listing_url(url)):
        return "listing"
    return "article"


def build_page_types(corpus=CORPUS, out=PAGE_TYPES):
    """Classify every article in the corpus and write the non-articles. Returns the counts."""
    counts = {"article": 0, "listing": 0, "horoscope": 0}
    rows = []
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            kind = classify(a.get("headline", ""), a.get("url"))
            counts[kind] += 1
            if kind != "article":
                rows.append((a["doc_id"], kind))
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Pages that aren't news articles, from dhvani/rank/quality.py (ids only).\n")
        for doc_id, kind in sorted(rows):
            f.write(f"{doc_id}\t{kind}\n")
    return counts


@lru_cache(maxsize=1)
def load_page_types(path=PAGE_TYPES):
    out = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                doc_id, _, kind = line.partition("\t")
                out[doc_id] = kind
    return out


def page_type(doc_id, index):
    known = load_page_types()
    if doc_id in known:
        return known[doc_id]        # classified from the corpus file, with the URL
    return classify(getattr(index, "articles", {}).get(doc_id, {}).get("headline", ""))


def demote(results, index, factor=DEMOTE_FACTOR):
    """Multiply non-article scores by `factor` and put non-articles after all articles.

    Articles keep their parser-stage order among themselves, and so do the
    non-articles.
    """
    stage_order, out = {}, []
    for i, (doc_id, score, explain) in enumerate(results):
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        kind = page_type(doc_id, index)
        explain["page_type"] = kind
        if kind != "article":
            score *= factor
        stage_order.setdefault(explain.get("stage"), i)
        out.append((doc_id, score, explain))
    out.sort(key=lambda item: (item[2]["page_type"] != "article", stage_order[item[2].get("stage")], -item[1], item[0]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify the corpus into articles, listings and horoscopes.")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--out", default=PAGE_TYPES)
    args = parser.parse_args(argv)
    counts = build_page_types(args.corpus, args.out)
    total = sum(counts.values())
    print(", ".join(f"{k} {v} ({v / total:.0%})" for k, v in counts.items()) + f" -> {args.out}")


if __name__ == "__main__":
    main()
