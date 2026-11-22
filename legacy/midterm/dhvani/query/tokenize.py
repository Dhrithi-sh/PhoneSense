"""Splitting raw query text into words.

This mirrors the token pattern the shared analyzer uses (Dhrithi's format 2):
letters, combining marks (Devanagari vowel signs) and digits. Plain ``\\w``
would split मौसम apart at the vowel sign, so we use the ``regex`` module's
Unicode property classes instead.

It stands in only for *query* tokenizing until Dhrithi's analyzer is importable;
documents are always tokenized by her analyzer, never by this.
"""

import regex

# [\p{L}\p{M}\p{Nd}]+ : a run of letters, combining marks and decimal digits.
TOKEN = regex.compile(r"[\p{L}\p{M}\p{Nd}]+")


def word_tokens(text):
    """Return the lowercased word tokens of ``text``, in order.

    Lowercasing is a no-op for Devanagari (it is caseless) and folds Roman
    input like "Kal"/"KAL" to one form.
    """
    return [match.group().lower() for match in TOKEN.finditer(text)]
