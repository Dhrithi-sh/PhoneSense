import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)

    # Normalize English/Hinglish case as well as Unicode form.
    text = text.lower()

    # Remove zero-width characters that can split otherwise
    # identical Hindi words.
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    # Standardize common Hindi spelling variation.
    text = text.replace("हिंदी", "हिन्दी")

    # Collapse repeated whitespace.
    text = " ".join(text.split())

    return text