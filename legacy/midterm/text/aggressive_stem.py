"""
Aggressive Hindi stemming.

This stemmer applies a larger suffix inventory than the light stemmer.
It is intentionally recall-oriented: different morphological forms are
collapsed more aggressively so that a query can match more documents.
"""

# Longer suffixes must be checked first. Otherwise a shorter suffix can
# consume part of a longer morphological ending.
AGGRESSIVE_SUFFIXES = sorted(
    {
        "ाइयाँ",
        "ाइयों",
        "ाइयां",
        "ाओंगी",
        "ाओंगे",
        "ाएंगी",
        "ाएंगे",
        "ाइयों",
        "ाइयां",
        "ियाँ",
        "ियों",
        "ियां",
        "ाऊंगी",
        "ाऊंगा",
        "ाएगी",
        "ाएगा",
        "ाइए",
        "ाइया",
        "ाइयाँ",
        "ाइयों",
        "ाएंगी",
        "ाएंगे",
        "ाती",
        "ाते",
        "ाता",
        "ाने",
        "ाना",
        "ाकर",
        "ायी",
        "ाये",
        "ाया",
        "ाइ",
        "ाई",
        "ाए",
        "ाओ",
        "ाऊ",
        "िया",
        "िये",
        "ियो",
        "ियाँ",
        "ियों",
        "ियां",
        "ीं",
        "ों",
        "ें",
        "ाँ",
        "ां",
        "ा",
        "ि",
        "ी",
        "ु",
        "ू",
        "े",
        "ै",
        "ो",
    },
    key=len,
    reverse=True,
)


# Very short stems are usually undesirable. For example, aggressively
# reducing a two- or three-character Hindi word can destroy its meaning.
MIN_STEM_LENGTH = 2


def stem_aggressive(word: str) -> str:
    """
    Aggressively reduce a Hindi word to a shorter stem.

    The longest matching suffix is removed, provided that enough of the
    original word remains.

    Words shorter than the minimum safe length are returned unchanged.
    """
    if not word:
        return word

    for suffix in AGGRESSIVE_SUFFIXES:
        if not word.endswith(suffix):
            continue

        stem = word[: -len(suffix)]

        if len(stem) >= MIN_STEM_LENGTH:
            return stem

    return word