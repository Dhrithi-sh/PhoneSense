"""A standard Roman spelling for a Devanagari Hindi word.

Every Hindi word gets *one* deterministic Latin spelling, so we can compare a
user's romanised query against a canonical form. This is a transliteration, not
a transcription: the goal is one stable spelling per word, not the single way a
human would type it. Phonetic matching (Soundex, Dhvani-code, edit distance)
bridges the gap between this canonical form and however the user actually spelt
it, so here we only need to be consistent.

The one piece of real Hindi phonology we apply is **final schwa deletion**: the
inherent "a" on the last consonant of a word is silent, so कमल is "kamal", not
"kamala". We do only the *final* "a" drop the plan asks for; medial schwa
deletion is a harder problem we deliberately skip.

Scheme choices (documented so tests and readers agree):
- long vowels double the letter: आ/ा -> "aa", ई/ी -> "ii", ऊ/ू -> "uu".
- aspirates add "h": ख -> "kh", थ -> "th", घ -> "gh".
- anusvara/chandrabindu -> "n"; visarga -> "h".
"""

_INHERENT = "a"

# Independent vowels (start of a syllable).
_INDEP_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ऍ": "e", "ऑ": "o", "ऎ": "e", "ऒ": "o",
}

# Dependent vowel signs (matras) that replace a consonant's inherent "a".
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
    "ॆ": "e", "ॊ": "o",
}

# Consonants carry an inherent "a" unless a matra or the virama follows.
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # Precomposed nukta letters.
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}

# Base + combining nukta (U+093C), when not precomposed.
_NUKTA = "़"
_NUKTA_MAP = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}

_VIRAMA = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
# Before a labial consonant an anusvara is the nasal "m" (भूकंप -> bhukamp),
# otherwise "n" (हिंदी -> hindi). Homorganic nasal assimilation.
_LABIAL_CONS = set("पफबभम") | {"फ़"}
_VISARGA = "ः"

_DEVA_DIGITS = {d: str(i) for i, d in enumerate("०१२३४५६७८९")}


def romanize(word):
    """Return the canonical Roman spelling of a Devanagari ``word``.

    Characters that are not Devanagari (already-Latin letters, punctuation) are
    passed through unchanged, so a word that is already Roman comes back as-is.
    """
    out = []
    # True when the last thing we appended was a consonant's silent inherent "a"
    # that nothing has cancelled; used only to drop a *final* schwa.
    trailing_schwa = False
    i, n = 0, len(word)

    while i < n:
        ch = word[i]

        if ch in _CONSONANTS or ch in _NUKTA_MAP:
            # Resolve the consonant, applying a following combining nukta.
            if ch in _CONSONANTS and not (i + 1 < n and word[i + 1] == _NUKTA):
                base = _CONSONANTS[ch]
            else:
                base = _NUKTA_MAP.get(ch, _CONSONANTS.get(ch, ch))
                if i + 1 < n and word[i + 1] == _NUKTA:
                    i += 1  # consume the nukta
            out.append(base)

            nxt = word[i + 1] if i + 1 < n else ""
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                trailing_schwa = False
                i += 1
            elif nxt == _VIRAMA:
                trailing_schwa = False  # bare consonant, no vowel
                i += 1
            else:
                out.append(_INHERENT)
                trailing_schwa = True

        elif ch in _INDEP_VOWELS:
            out.append(_INDEP_VOWELS[ch])
            trailing_schwa = False
        elif ch in _MATRAS:  # a matra with no consonant before it (rare/malformed)
            out.append(_MATRAS[ch])
            trailing_schwa = False
        elif ch in (_ANUSVARA, _CHANDRABINDU):
            nxt = next((c for c in word[i + 1:] if c in _CONSONANTS), None)
            out.append("m" if nxt in _LABIAL_CONS else "n")
            trailing_schwa = False
        elif ch == _VISARGA:
            out.append("h")
            trailing_schwa = False
        elif ch in _DEVA_DIGITS:
            out.append(_DEVA_DIGITS[ch])
            trailing_schwa = False
        elif ch in (_VIRAMA, _NUKTA, "‌", "‍"):  # stray virama/nukta, ZWNJ, ZWJ
            pass
        else:
            out.append(ch)  # non-Devanagari: pass through
            trailing_schwa = False

        i += 1

    if trailing_schwa and out and out[-1] == _INHERENT:
        out.pop()  # final schwa deletion: कमल -> kamal, not kamala

    return "".join(out)
