"""
Hindi light stemmer based on:

A. Ramanathan and D. Rao (2003),
"A Lightweight Stemmer for Hindi".

The suffix inventory follows the Devanagari implementation
used by Apache Lucene's HindiStemmer.
"""


# Suffixes are grouped by length.
# Longer suffix groups are checked first.
HINDI_SUFFIX_GROUPS = [
    # 5-character suffixes
    (5, 7, [
        "ाएंगी",
        "ाएंगे",
        "ाऊंगी",
        "ाऊंगा",
        "ाइयाँ",
        "ाइयों",
        "ाइयां",
    ]),

    # 4-character suffixes
    (4, 6, [
        "ाएगी",
        "ाएगा",
        "ाओगी",
        "ाओगे",
        "एंगी",
        "ेंगी",
        "एंगे",
        "ेंगे",
        "ूंगी",
        "ूंगा",
        "ातीं",
        "नाओं",
        "नाएं",
        "ताओं",
        "ताएं",
        "ियाँ",
        "ियों",
        "ियां",
    ]),

    # 3-character suffixes
    (3, 5, [
        "ाकर",
        "ाइए",
        "ाईं",
        "ाया",
        "ेगी",
        "ेगा",
        "ोगी",
        "ोगे",
        "ाने",
        "ाना",
        "ाते",
        "ाती",
        "ाता",
        "तीं",
        "ाओं",
        "ाएं",
        "ुओं",
        "ुएं",
        "ुआं",
    ]),

    # 2-character suffixes
    (2, 4, [
        "कर",
        "ाओ",
        "िए",
        "ाई",
        "ाए",
        "ने",
        "नी",
        "ना",
        "ते",
        "ीं",
        "ती",
        "ता",
        "ाँ",
        "ां",
        "ों",
        "ें",
    ]),

    # 1-character suffixes
    (1, 3, [
        "ो",
        "े",
        "ू",
        "ु",
        "ी",
        "ि",
        "ा",
    ]),
]


def stem(word: str) -> str:
    """
    Apply the Hindi light stemming algorithm.

    Removes the longest matching Hindi inflectional suffix,
    subject to the minimum word-length conditions of the
    Ramanathan and Rao algorithm.

    If no suffix matches, the original word is returned.
    """

    for suffix_length, min_word_length, suffixes in HINDI_SUFFIX_GROUPS:
        if len(word) >= min_word_length:
            for suffix in suffixes:
                if word.endswith(suffix):
                    return word[:-suffix_length]

    return word