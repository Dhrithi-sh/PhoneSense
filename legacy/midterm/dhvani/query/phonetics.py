"""Phonetic keys that put spelling variants into the same bucket.

Two words that sound alike but are spelt differently ("mausam" / "mosam") should
collide on a phonetic key, so we can find candidates fast. We provide two:

1. ``soundex`` — the classic algorithm from the lecture (Manning IIR §3.4),
   run on a Roman string. First letter kept, the rest coded to digits.

2. ``dhvani_code`` — our own Soundex built for Hindi. It works on *either*
   Devanagari or Roman input by first mapping every consonant to a shared
   phonetic class, then dropping the vowels. So मौसम, "mausam" and "mosam" all
   reduce to the same consonant skeleton ``585`` (म=5, स=8, म=5). This is the
   piece that makes matching script-agnostic.
"""

from dhvani.query.langid import script

# ---------------------------------------------------------------------------
# 1. Classic Soundex (lecture version, Manning IIR §3.4)
# ---------------------------------------------------------------------------

_SOUNDEX = {}
for _letters, _digit in [
    ("bfpv", "1"),
    ("cgjkqsxz", "2"),
    ("dt", "3"),
    ("l", "4"),
    ("mn", "5"),
    ("r", "6"),
]:
    for _c in _letters:
        _SOUNDEX[_c] = _digit
# Vowels and h, w, y map to "0" (separators that get removed at the end).
for _c in "aeiouhwy":
    _SOUNDEX[_c] = "0"


def soundex(word):
    """Return the 4-character Soundex code of a Roman ``word`` (e.g. "R163").

    Returns "" for a word with no ASCII letters.
    """
    letters = [c for c in word.lower() if "a" <= c <= "z"]
    if not letters:
        return ""

    codes = [_SOUNDEX.get(c, "0") for c in letters]

    # Step 4: collapse runs of the same digit (over the whole string, first
    # letter included, so "ss" at the front counts once).
    collapsed = [codes[0]]
    for code in codes[1:]:
        if code != collapsed[-1]:
            collapsed.append(code)

    # Step 5: keep the first letter, drop the zeros from the rest.
    tail = [code for code in collapsed[1:] if code != "0"]
    result = letters[0].upper() + "".join(tail)
    return (result + "000")[:4]


# ---------------------------------------------------------------------------
# 2. Dhvani-code: a Hindi-aware consonant skeleton, script-agnostic
# ---------------------------------------------------------------------------

# Phonetic classes. Sounds that commonly swap in romanisation share a class.
# The digit is just a label; what matters is that like sounds collide.
#   1 velars      क ख ग घ क़ ख़ ग़   (k, kh, g, gh, q)
#   2 palatals    च छ ज झ ज़ य       (ch, j, y)
#   3 stops t/d   ट ठ ड ढ त थ द ध   (t, th, d, dh)  + ड़ ढ़
#   4 nasal n     ङ ञ ण न  + anusvara/chandrabindu
#   5 nasal m     म
#   6 r           र
#   7 l           ल ळ
#   8 sibilant/h  स श ष ह
#   9 labials     प फ ब भ व फ़       (p, ph, b, bh, v, w, f)
_DEVA_CLASS = {}
for _chars, _cls in [
    ("कखगघ", "1"), ("चछजझ", "2"), ("यय़", "2"),
    ("टठडढतथदध", "3"),
    ("ङञणन", "4"),
    ("म", "5"), ("र", "6"), ("लळ", "7"),
    ("सशषह", "8"),
    ("पफबभव", "9"),
    # precomposed nukta letters
    ("क़ख़ग़", "1"), ("ज़", "2"), ("ड़ढ़", "3"), ("फ़", "9"),
]:
    for _ch in _chars:
        _DEVA_CLASS[_ch] = _cls

_ANUSVARA, _CHANDRABINDU = "ं", "ँ"
# Consonants before which an anusvara is pronounced as the labial nasal म (class
# 5); before anything else it is the dental/other nasal न (class 4). This makes
# भूकंप code like भूकम्प and हिंदी like हिन्दी.
_LABIALS = set("पफबभम") | {"फ़"}

# Roman phonetic units, longest first so digraphs win over single letters.
_ROMAN_UNITS = [
    ("chh", "2"), ("ch", "2"), ("sh", "8"), ("kh", "1"), ("gh", "1"),
    ("th", "3"), ("dh", "3"), ("ph", "9"), ("bh", "9"), ("jh", "2"),
    ("rh", "6"),
    ("k", "1"), ("q", "1"), ("g", "1"), ("x", "1"),
    ("c", "2"), ("j", "2"), ("y", "2"), ("z", "2"),
    ("t", "3"), ("d", "3"),
    ("n", "4"), ("m", "5"), ("r", "6"), ("l", "7"),
    ("s", "8"), ("h", "8"),
    ("p", "9"), ("b", "9"), ("v", "9"), ("w", "9"), ("f", "9"),
]
# Letters that carry no consonant class (vowels) are simply skipped.


def _collapse(digits):
    """Drop a digit equal to the one before it ("patta" tt -> one 3)."""
    out = []
    for d in digits:
        if not out or out[-1] != d:
            out.append(d)
    return "".join(out)


def _dhvani_devanagari(word):
    digits = []
    for i, ch in enumerate(word):
        if ch in (_ANUSVARA, _CHANDRABINDU):
            # Homorganic nasal: look at the next consonant to decide म vs न.
            nxt = next((c for c in word[i + 1:] if c in _DEVA_CLASS), None)
            digits.append("5" if nxt in _LABIALS else "4")
        elif ch in _DEVA_CLASS:
            digits.append(_DEVA_CLASS[ch])
    return _collapse(digits)


def _dhvani_roman(word):
    word = word.lower()
    digits = []
    i, n = 0, len(word)
    while i < n:
        for unit, cls in _ROMAN_UNITS:
            if word.startswith(unit, i):
                digits.append(cls)
                i += len(unit)
                break
        else:
            i += 1  # vowel or other letter: skip, no consonant class
    return _collapse(digits)


def dhvani_code(word):
    """Return the Dhvani consonant-skeleton key for a Hindi or Hinglish word.

    Devanagari input is coded from its characters; Roman input is coded from its
    phonetic units. Both yield the same key for the same-sounding word, e.g.
    ``dhvani_code("मौसम") == dhvani_code("mausam") == dhvani_code("mosam") == "585"``.
    """
    if script(word) == "devanagari":
        return _dhvani_devanagari(word)
    return _dhvani_roman(word)
