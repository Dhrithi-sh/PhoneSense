import regex


TOKEN_PATTERN = regex.compile(r"[\p{L}\p{M}\p{Nd}]+")


def tokenize(text: str) -> list[str]:
    """
    Tokenize Hindi, Hinglish, and English text.

    Tokens consist of Unicode letters, combining marks,
    and decimal digits.
    """

    return TOKEN_PATTERN.findall(text)