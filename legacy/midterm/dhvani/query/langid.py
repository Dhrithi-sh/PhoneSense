"""Per-word language identification for a mixed Hindi / Hinglish / English query.

A Dhvani query can mix three ways of writing the same thing: Devanagari Hindi
(``मौसम``), romanised Hindi a.k.a. Hinglish (``mausam``) and English
(``weather``). We label each *word* on its own, because one query often mixes
them: ``kal ka weather``.

The label is a weight over three readings ``{"hi", "hinglish", "en"}`` and these
feed the query object (format 4). The plan calls out that some words — "main",
"to", "hi" — are valid English words *and* common romanised Hindi, so instead of
forcing a choice we keep both readings with weights and let the ranker sort it
out downstream.

Scope: a lexicon-based identifier. The real coverage comes from a large bundled
English wordlist (``english_words.txt``); the small sets below handle the Hindi
side and the genuinely ambiguous words.
"""

import os

# Devanagari block is U+0900..U+097F (plus extensions we do not need here).
_DEVA_START, _DEVA_END = "ऀ", "ॿ"

# Words that are ordinary English *and* common romanised Hindi. We refuse to
# guess and carry both readings. (Romanised Hindi gloss in the comment.)
AMBIGUOUS = {
    "main",  # English "main"      vs मैं "I"
    "to",    # English "to"        vs तो "then/so"
    "hi",    # English "hi"        vs ही "only/just"
    "is",    # English "is"        vs इस "this"
    "me",    # English "me"        vs में "in"
    "us",    # English "us"        vs उस "that"
    "do",    # English "do"        vs दो "two/give"
    "he",    # English "he"        vs हे "O! (vocative)"
    "the",   # English "the"       vs थे "were"
    "or",    # English "or"        vs और "and" (often typed "or")
}

# Common romanised Hindi words that are *also* ordinary English words
# (function words, time words, loanwords) and so leak into an English wordlist.
# We force these to the Hinglish reading so "kal ka mausam" isn't read as English.
HINDI_PROTECT = set("""
ka ki ke ko se mein par pe ne wala wali hai hain ho hua hui tha thi ja jaa kar kiya karo karna raha rahe rahi gaya gayi
tu tum aap hum wo vo ye yeh in un jo kya kaun kahan kab kaise kaisa kyun kyon kitna kitni
aur ya bhi na nahi nahin mat phir ab abhi bas sab kuch koi
kal aaj aj parso raat din subah shaam saal mahina hafta waqt samay
bazaar bharat dilli sheher gaon naam kaam paani pani aag hawa ghar log aadmi raja rani dev mandir masjid roti chai dil jaan paisa khana mausam barish baarish chunav
delhi mumbai bombay kolkata chennai bengaluru bangalore hyderabad pune nagpur lucknow kanpur patna bhopal indore jaipur surat agra varanasi kashi prayagraj allahabad ranchi raipur noida gurugram gurgaon
bihar jharkhand punjab haryana gujarat rajasthan kerala karnataka maharashtra odisha assam bengal telangana
""".split())

# A large English vocabulary, baked into dhvani/query/english_words.txt (common
# words from wordfreq, minus AMBIGUOUS and HINDI_PROTECT). Loaded once, lazily.
# If the file is missing we fall back to a tiny seed so language ID still runs.
_ENGLISH_SEED = {
    "weather", "tomorrow", "today", "news", "election", "elections", "cricket",
    "match", "rain", "price", "prices", "market", "stock", "result", "school",
    "holiday", "budget", "government", "minister", "police", "flood", "monsoon",
    "forecast", "earthquake", "farmers", "snow", "worried",
}
_ENGLISH_PATH = os.path.join(os.path.dirname(__file__), "english_words.txt")
_english = None


def english_words():
    """The English vocabulary set, loaded once from the bundled file."""
    global _english
    if _english is None:
        try:
            with open(_ENGLISH_PATH, encoding="utf-8") as fh:
                _english = {line.strip() for line in fh if line.strip()}
        except OSError:
            _english = set(_ENGLISH_SEED)
    return _english


def script(word):
    """"devanagari" if the word contains any Devanagari character, else "roman"."""
    for ch in word:
        if _DEVA_START <= ch <= _DEVA_END:
            return "devanagari"
    return "roman"


def classify(word):
    """Return a language-weight dict ``{"hi", "hinglish", "en"}`` for one word.

    - Devanagari  -> certainly Hindi.
    - Ambiguous   -> split evenly between a romanised-Hindi and an English reading.
    - Known English -> mostly English, a little romanised-Hindi mass kept for safety.
    - Anything else in Roman -> treated as romanised Hindi, the common case in a
      Hindi-news engine, with a little English mass kept.

    The small residual mass (0.1) on the losing reading mirrors the hedge in the
    format example (``mosam`` -> hinglish 0.9, en 0.1): it keeps a backup reading
    alive rather than zeroing it out.
    """
    word = word.lower()

    if script(word) == "devanagari":
        return {"hi": 1.0, "hinglish": 0.0, "en": 0.0}

    if word in AMBIGUOUS:
        return {"hi": 0.0, "hinglish": 0.5, "en": 0.5}

    if word in HINDI_PROTECT:
        return {"hi": 0.0, "hinglish": 0.9, "en": 0.1}

    if word in english_words():
        return {"hi": 0.0, "hinglish": 0.1, "en": 0.9}

    return {"hi": 0.0, "hinglish": 0.9, "en": 0.1}
