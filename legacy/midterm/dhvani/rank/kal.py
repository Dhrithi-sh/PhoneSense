"""Date-aware "kal": कल means both yesterday and tomorrow.

"kal ka mausam" is about tomorrow's weather; "kal ka match" is usually about
yesterday's game. Plain matching treats both the same, so this module works
out which day the query means and nudges results towards articles about that
day.

1. Spot a kal word in the query: कल / kal, or English tomorrow / yesterday.
2. Decide the direction:
   - English "tomorrow" and "yesterday" say it outright.
   - Otherwise look at the other query words. Future cues (होगा, रहेगा,
     forecast, alert, and weather words, since weather news is forecasts)
     mean tomorrow; past cues (हुआ, था, result) mean yesterday.
   - With no cue at all we assume yesterday, because news mostly reports
     what already happened.
3. Re-rank with a small date boost, measured from the newest article (the
   same "now" as recency):
   - yesterday: articles published the day before get the full boost.
   - tomorrow: the newest articles get it if they're written in the future
     tense (forecasts, alerts, "होगा", "रहेगी"); older or past-tense ones
     get less.
The boost multiplies the score, score x (1 + weight x boost), so an article
from the right day moves up among relevant results but an unrelated article
can't jump ahead just because of its date. It's recorded under explain["kal"].
"""

from datetime import datetime

from dhvani.rank.parser import STAGES

KAL_WORDS = {"कल", "kal", "kl"}
ENGLISH_DIRECTION = {"tomorrow": "tomorrow", "yesterday": "yesterday"}

FUTURE_CUES = {
    "होगा", "होगी", "होंगे", "रहेगा", "रहेगी", "रहेंगे", "करेंगे", "करेगा", "करेगी", "आएगा", "जाएगा",
    "hoga", "hogi", "rahega", "rahegi", "karenge",
    "अलर्ट", "चेतावनी", "पूर्वानुमान", "संभावना", "alert", "forecast", "warning", "will",
    # weather news is almost always a forecast
    "मौसम", "बारिश", "तापमान", "mausam", "mosam", "barish", "baarish", "weather", "rain",
}
PAST_CUES = {
    "हुआ", "हुई", "हुए", "था", "थी", "थे", "गया", "गई", "गए", "लिया", "किया",
    "hua", "hui", "tha", "thi", "gaya",
    "नतीजा", "नतीजे", "result", "results", "score", "won", "lost",
}
FUTURE_ENDINGS = ("ेगा", "ेगी", "ेंगे", "ेगे")   # Hindi future verb endings: होगा, रहेगी, करेंगे


def kal_intent(query):
    """'tomorrow', 'yesterday', or None if the query has no kal word."""
    words = [t["surface"].lower() for t in query["tokens"]]
    for w in words:
        if w in ENGLISH_DIRECTION:
            return ENGLISH_DIRECTION[w]
    if not any(w in KAL_WORDS for w in words):
        return None
    future = sum(1 for w in words if w in FUTURE_CUES)
    past = sum(1 for w in words if w in PAST_CUES)
    return "tomorrow" if future > past else "yesterday"


def is_future_tense(text):
    """True if the article talks more about what will happen than what did."""
    words = text.split()
    future = sum(1 for w in words if w.strip("।.,") in FUTURE_CUES or w.strip("।.,").endswith(FUTURE_ENDINGS))
    past = sum(1 for w in words if w.strip("।.,") in PAST_CUES)
    return future > past


def _day(date_str):
    return datetime.fromisoformat(date_str).date() if date_str else None


def kal_boost(intent, meta, text, today):
    """Boost in [0, 1] for one article."""
    published = _day(meta.get("date"))
    if published is None:
        return 0.0
    days_old = (today - published).days
    if intent == "yesterday":
        return 1.0 if days_old == 1 else (0.3 if days_old == 0 else 0.0)
    # tomorrow: fresh articles written in the future tense
    if days_old <= 1 and is_future_tense(text):
        return 1.0 if days_old == 0 else 0.6
    return 0.2 if days_old <= 1 else 0.0


def apply_kal(results, query, index, weight=0.5, texts=None, now=None):
    """Re-rank results for a kal query. Leaves other queries untouched.

    results: [(doc_id, score, explain)] from any ranker.
    texts: doc_id -> article text; defaults to index.articles headline + body.
    """
    intent = kal_intent(query)
    if intent is None or not results:
        return results
    if now is None:
        dates = [m["date"] for m in index.meta.values() if m.get("date")]
        now = max(datetime.fromisoformat(d) for d in dates) if dates else None
    if now is None:
        return results
    today = now.date()
    articles = getattr(index, "articles", {})

    def text_of(doc_id):
        if texts:
            return texts(doc_id) or ""
        a = articles.get(doc_id, {})
        return f"{a.get('headline', '')} {a.get('body', '')}"

    reranked = []
    for doc_id, score, explain in results:
        boost = kal_boost(intent, index.meta[doc_id], text_of(doc_id), today)
        # lnc.ltc and BM25 give a plain {term: contribution} dict; keep the term
        # scores under "terms" so the extra info doesn't look like a term.
        explain = dict(explain) if "terms" in explain else {"terms": dict(explain)}
        explain["kal"] = {"intent": intent, "boost": boost, "weight": weight}
        reranked.append((doc_id, score * (1 + weight * boost), explain))
    # Keep the query parser's order: a stricter stage (e.g. exact phrase)
    # still ranks above a looser one; kal only reorders within a stage.
    def stage_rank(explain):
        stage = explain.get("stage")
        return STAGES.index(stage) if stage in STAGES else 0

    reranked.sort(key=lambda item: (stage_rank(item[2]), -item[1], item[0]))
    return reranked
