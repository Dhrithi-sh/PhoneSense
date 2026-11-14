"""Small normalizers shared by dedupe and build_phones: brand, model name, price."""

from __future__ import annotations

import math
import re
from typing import Any

from phonesense.data.common import clean_text

# Canonical spelling for brands that appear in many casings/variants. The Boolean filter
# compares brands exactly, so "SAMSUNG", "Samsung Electronics" and "samsung" must agree.
CANONICAL_BRANDS = {
    "apple": "Apple", "samsung": "Samsung", "samsung electronics": "Samsung",
    "google": "Google", "motorola": "Motorola", "moto": "Motorola",
    "motorola mobility": "Motorola", "oneplus": "OnePlus", "one plus": "OnePlus",
    "nokia": "Nokia", "hmd": "Nokia", "lg": "LG", "lg electronics": "LG",
    "xiaomi": "Xiaomi", "redmi": "Xiaomi", "poco": "Xiaomi", "blu": "BLU", "blu products": "BLU",
    "tcl": "TCL", "sony": "Sony", "huawei": "Huawei", "honor": "Honor", "zte": "ZTE",
    "alcatel": "Alcatel", "oppo": "OPPO", "realme": "realme", "vivo": "vivo", "asus": "ASUS",
    "nothing": "Nothing", "ulefone": "Ulefone", "doogee": "DOOGEE", "blackview": "Blackview",
    "oukitel": "OUKITEL", "cubot": "CUBOT", "umidigi": "UMIDIGI", "fairphone": "Fairphone",
    "kyocera": "Kyocera", "htc": "HTC", "blackberry": "BlackBerry", "cat": "CAT",
    "sonim": "Sonim", "lenovo": "Lenovo", "infinix": "Infinix", "tecno": "TECNO",
    "unihertz": "Unihertz", "nuu": "NUU", "nuu mobile": "NUU", "orbic": "Orbic",
}
_GENERIC_BRANDS = {"generic", "unbranded", "unknown", "n/a", "na", "none", "no brand",
                   "other", "brand"}
# Sellers, not manufacturers: never used as a brand.
_RESELLER_RX = re.compile(r"^amazon\b|\brenewed\b|\brefurbished\b|\bwireless\b|\bmobile store\b"
                          r"|\bdeals?\b|\boutlet\b|\bwholesale\b", re.I)
_STORE_PREFIX = re.compile(r"^visit the\s+|\s+store$", re.I)
# A known brand among the first words of the title ("Apple iPhone 12 ...",
# "Simple Mobile Prepaid - Apple iPhone 11").
_TITLE_BRAND_RX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CANONICAL_BRANDS, key=len, reverse=True))
    + r")\b",
    re.I,
)


def _clean_brand(raw: Any) -> str | None:
    b = _STORE_PREFIX.sub("", clean_text(raw)).strip(" -,.")
    if not b or b.lower() in _GENERIC_BRANDS:
        return None
    return CANONICAL_BRANDS.get(b.lower(), b)


def canonical_brand(details_brand: Any, store: Any, title: Any = None) -> str | None:
    """Brand from ``details.Brand``, else a known brand in the first words of the title, else
    the store (unless it is a reseller such as "Amazon Renewed"). Known brands get one
    canonical spelling. None if unknown."""
    b = _clean_brand(details_brand)
    if b:
        return b
    head = " ".join(clean_text(title).split()[:6])
    m = _TITLE_BRAND_RX.search(head)
    if m:
        return CANONICAL_BRANDS[m.group(1).lower()]
    b = _clean_brand(store)
    if b and not _RESELLER_RX.search(b):
        return b
    return None


_PRICE_RX = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)$")


def parse_price(v: Any) -> float | None:
    """'$199.99', '1,099.00', 199.99 -> float > 0. Ranges ('199 - 249'), '—', None -> None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | float):
        f = float(v)
    elif isinstance(v, str):
        m = _PRICE_RX.match(v.strip())
        if not m:
            return None
        f = float(m.group(1).replace(",", ""))
    else:
        return None
    return round(f, 2) if math.isfinite(f) and f > 0 else None


_GENERIC_MODELS = {"smartphone", "smartphones", "cellphone", "cellphones", "phone", "phones",
                   "mobilephone", "unlocked", "android", "galaxy", "iphone", "pixel", "moto",
                   "motorola", "renewed", "refurbished", "5g", "4g", "lte"}


def normalize_model(model: Any, brand: str | None) -> str | None:
    """'Galaxy S23 Ultra' -> 'galaxys23ultra'. None if missing or too generic to dedupe on
    (a generic key like 'smartphone' would merge unrelated phones)."""
    m = clean_text(model).lower()
    if not m:
        return None
    if brand:
        m = re.sub(rf"^{re.escape(brand.lower())}\b", "", m)
    # Drop seller noise but keep year tags: "Galaxy A3 (2016)" != "Galaxy A3 (2017)".
    m = re.sub(r"\b(?:unlocked|renewed|refurbished|smartphone|cell\s*phone)\b", " ", m)
    key = re.sub(r"[^a-z0-9+]", "", m).replace("+", "plus")
    if not key or key in _GENERIC_MODELS:
        return None
    # A family name without a number ("Galaxy Note", "Galaxy S") covers many models.
    if not re.search(r"\d", key):
        return None
    return key


_VARIANT_RX = re.compile(
    r"\b(lite|plus|pro|max|mini|ultra|edge|fe|xl|prime|neo|go|play|power|active|note|"
    r"5g|20[0-3]\d)\b|(\+)",
    re.I,
)


def title_variants(title: Any) -> frozenset[str]:
    """Variant words in a title ("Note 10 Lite" vs "Note 10+"): listings are only merged when
    these agree, because the dataset's Model Name is often coarser than the actual model."""
    out = set()
    for m in _VARIANT_RX.finditer(clean_text(title)):
        out.add((m.group(1) or "plus").lower().replace("+", "plus"))
    return frozenset(out)
