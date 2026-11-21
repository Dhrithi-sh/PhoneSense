"""Stage 3: parse the normalized ``specs`` object (CONTRACT.md 4.1) for each candidate phone.

One small pure function per field. Each field is looked up in priority order
**details -> title -> features**; the first source that yields a valid value wins, and the
source is recorded (for the hand check). Unknown or impossible values are ``None``, never
guessed.

Run: ``python -m phonesense.data.parse_specs``
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from phonesense.data.common import (
    AtomicJsonlWriter,
    Settings,
    iter_jsonl,
    mark_stage_done,
    stage_is_cached,
)
from phonesense.schemas import SPEC_FIELDS, Specs

# Plausibility limits for phones (stricter than the contract's schema limits).
RAM_MAX_GB = 32.0
STORAGE_MAX_GB = 4096.0
BATTERY_MIN_MAH, BATTERY_MAX_MAH = 1000, 10000
SCREEN_MIN_IN, SCREEN_MAX_IN = 3.0, 8.0
CAMERA_MAX_MP = 400.0
YEAR_MIN, YEAR_MAX = 2000, 2030

SOURCES = ("details", "title", "features")

_NUM = r"(\d+(?:[.,]\d+)?)"
_UNIT_GB = {"mb": 1 / 1024, "gb": 1.0, "tb": 1024.0}


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _to_gb(value: str, unit: str | None) -> float | None:
    if unit is None:
        return None
    return round(_num(value) * _UNIT_GB[unit.lower()], 4)


# --------------------------------------------------------------------------------------
# Memory: RAM and storage are parsed together so "8GB+128GB" etc. are never confused.
# --------------------------------------------------------------------------------------

_MEM_TOKEN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(MB|GB|TB)\b", re.I)
# "8GB+128GB", "4GB/64GB", "4/64GB", "8 GB RAM + 256 GB", "6GB+128GB ROM"
_MEM_COMBO = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(MB|GB|TB)?\s*(?:RAM)?\s*[+/]\s*"
    r"(\d+(?:\.\d+)?)\s*(MB|GB|TB)\b(?:\s*(?:ROM|storage))?",
    re.I,
)
# "8GB RAM", "8 GB of RAM", "8GB LPDDR5 RAM", "8GB Memory RAM"
_RAM_AFTER = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(MB|GB)\s*(?:of\s+)?(?:LP)?(?:DDR\d*X?\s*)?"
    r"(?:memory\s+)?RAM\b",
    re.I,
)
# "RAM: 8GB", "RAM 8 GB", "8GB RAM" handled above
_RAM_BEFORE = re.compile(r"\bRAM\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(MB|GB)\b", re.I)
# "128GB storage", "128GB ROM", "128 GB internal memory", "storage: 128GB"
_STORAGE_AFTER = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(MB|GB|TB)\s*(?:of\s+)?(?:internal\s+)?"
    r"(?:storage|ROM|internal\s+memory|memory\s+storage)\b",
    re.I,
)
_STORAGE_BEFORE = re.compile(
    r"\b(?:storage|ROM|internal\s+memory)\s*(?:capacity)?\s*[:\-]?\s*"
    r"(\d+(?:\.\d+)?)\s*(MB|GB|TB)\b",
    re.I,
)
# Expandable storage mentions are not internal storage.
_EXPANDABLE_CTX = re.compile(
    r"(up\s*to|expand\w*|micro\s*-?sd|sd\s*card|tf\s*card|support\w*|extend\w*|"
    r"cloud|data|hotspot|plan|wi-?fi)\W*(?:\w+\W+){0,3}$",
    re.I,
)
_STANDARD_STORAGE = {8, 16, 32, 64, 128, 256, 512, 1024, 2048}


def _valid_ram(v: float | None) -> float | None:
    return v if v is not None and 0 < v <= RAM_MAX_GB else None


def _valid_storage(v: float | None) -> float | None:
    return v if v is not None and 0 < v <= STORAGE_MAX_GB else None


def _is_expandable(text: str, start: int) -> bool:
    return bool(_EXPANDABLE_CTX.search(text[max(0, start - 40):start]))


def parse_memory(text: str) -> tuple[float | None, float | None]:
    """Return ``(ram_gb, storage_gb)`` from free text. Either may be None."""
    if not text:
        return None, None
    ram = storage = None

    m = _MEM_COMBO.search(text)
    if m and not _is_expandable(text, m.start()):
        a_unit = m.group(2) or m.group(4)
        a = _to_gb(m.group(1), a_unit)
        b = _to_gb(m.group(3), m.group(4))
        if a is not None and b is not None and a < b:
            ram, storage = _valid_ram(a), _valid_storage(b)

    ram_span: tuple[int, int] | None = None
    if ram is None:
        for rx in (_RAM_AFTER, _RAM_BEFORE):
            m = rx.search(text)
            if m:
                ram_span = ram_span or m.span()  # explicit RAM is never storage, even if
                ram = _valid_ram(_to_gb(m.group(1), m.group(2)))  # its value is impossible
                if ram is not None:
                    ram_span = m.span()
                    break

    if storage is None:
        for rx in (_STORAGE_AFTER, _STORAGE_BEFORE):
            for m in rx.finditer(text):
                if not _is_expandable(text, m.start()):
                    storage = _valid_storage(_to_gb(m.group(1), m.group(2)))
                    if storage is not None:
                        break
            if storage is not None:
                break

    if storage is None:
        # Generic "128GB" tokens that are not RAM and not expandable storage.
        values = []
        for m in _MEM_TOKEN.finditer(text):
            if ram_span and ram_span[0] <= m.start() < ram_span[1]:
                continue
            if _is_expandable(text, m.start()):
                continue
            v = _to_gb(m.group(1), m.group(2))
            if v is not None and v != ram:
                values.append(v)
        distinct = sorted(set(values))
        if len(distinct) == 1:
            storage = _valid_storage(distinct[0])
        elif len(distinct) == 2:
            small, big = distinct
            # "64GB 4GB" style listings: the small value is RAM, the big one storage.
            if small <= 16 and big >= 16 and big in _STANDARD_STORAGE:
                storage = _valid_storage(big)
                if ram is None:
                    ram = _valid_ram(small)
        # 3+ distinct sizes: ambiguous -> unknown.

    if ram is not None and storage is not None and ram >= storage:
        ram = None  # RAM can't exceed storage; the value is a mislabelled storage size
    return ram, storage


def parse_ram_value(value: str) -> float | None:
    """A details value for a RAM key, e.g. "8 GB", "6144 MB"."""
    m = _MEM_TOKEN.search(value or "")
    return _valid_ram(_to_gb(m.group(1), m.group(2))) if m else None


def parse_storage_value(value: str) -> float | None:
    """A details value for a storage key, e.g. "128 GB", "1 TB"."""
    m = _MEM_TOKEN.search(value or "")
    return _valid_storage(_to_gb(m.group(1), m.group(2))) if m else None


# --------------------------------------------------------------------------------------
# Battery
# --------------------------------------------------------------------------------------

_BATTERY = re.compile(
    r"(?<![\d.,])(\d{1,2}[,.]\d{3}|\d{3,5})\s*(?:-\s*)?(?:mAh|milli\s*amp(?:ere)?\s*hours?)\b",
    re.I,
)


def parse_battery(text: str) -> int | None:
    """"5000mAh", "5,000 mAh", "4500 Milliamp Hours" -> int. Wh values are not converted."""
    for m in _BATTERY.finditer(text or ""):
        v = int(re.sub(r"[,.]", "", m.group(1)))
        if BATTERY_MIN_MAH <= v <= BATTERY_MAX_MAH:
            return v
        return None  # first stated battery value is implausible -> unknown
    return None


# --------------------------------------------------------------------------------------
# Screen
# --------------------------------------------------------------------------------------

_SCREEN_IN = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s*(?:-\s*)?(?:inch(?:es)?\b|in\b\.?|\"|”|″|''|’’)",
    re.I,
)
_SCREEN_CM = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s*(?:-\s*)?(?:cm|centimet(?:er|re)s?)\b", re.I
)
_SCREEN_CTX = re.compile(r"display|screen|amoled|oled|lcd|ips|retina|hd\+?|fhd|inch", re.I)


def _valid_screen(v: float | None) -> float | None:
    return round(v, 2) if v is not None and SCREEN_MIN_IN <= v <= SCREEN_MAX_IN else None


def parse_screen_value(value: str) -> float | None:
    """Details value for a screen key: "6.1 Inches", "6.1", "16.5 cm"."""
    value = value or ""
    m = _SCREEN_CM.search(value)
    if m:
        return _valid_screen(_num(m.group(1)) / 2.54)
    m = _SCREEN_IN.search(value) or re.fullmatch(r"\s*(\d{1,2}(?:\.\d{1,2})?)\s*", value)
    return _valid_screen(_num(m.group(1))) if m else None


def parse_screen(text: str) -> float | None:
    """Free text: '6.7"', '6.1 Inches', '6.5-inch display'."""
    for m in _SCREEN_IN.finditer(text or ""):
        v = _valid_screen(_num(m.group(1)))
        if v is not None:
            return v
    for m in _SCREEN_CM.finditer(text or ""):
        ctx = (text or "")[max(0, m.start() - 30):m.end() + 30]
        if _SCREEN_CTX.search(ctx):
            v = _valid_screen(_num(m.group(1)) / 2.54)
            if v is not None:
                return v
    return None


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------

_CAMERA = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)\s*(?:-\s*)?(?:MP|mega\s*-?pixels?)\b",
                     re.I)
_FRONT = re.compile(r"front|selfie", re.I)
_CAM_SEP = re.compile(r"[,;|+/()\[\]]|\band\b|\bwith\b", re.I)


def parse_camera(text: str) -> float | None:
    """Largest rear camera in MP: "48MP + 12MP" -> 48. Front/selfie values are skipped."""
    text = text or ""
    best = None
    for m in _CAMERA.finditer(text):
        # The clause around the number ("32MP front camera", "selfie: 16MP") decides.
        before = _CAM_SEP.split(text[max(0, m.start() - 25):m.start()])[-1]
        after = _CAM_SEP.split(text[m.end():m.end() + 25])[0]
        if _FRONT.search(before) or _FRONT.search(after):
            continue
        v = _num(m.group(1))
        if 0 < v <= CAMERA_MAX_MP:
            best = v if best is None else max(best, v)
    return best


# --------------------------------------------------------------------------------------
# OS
# --------------------------------------------------------------------------------------

_ANDROID = re.compile(r"\bandroid\b", re.I)
_IOS = re.compile(r"\bios\b|\bi\s*os\s*\d", re.I)
_OTHER_OS = re.compile(
    r"\b(kai\s*os|windows\s*(?:phone|mobile|10\s*mobile)|blackberry\s*(?:os|10)|symbian|"
    r"tizen|harmony\s*os|firefox\s*os|ubuntu\s*touch|sailfish|webos|"
    r"proprietary|feature\s*phone\s*os)\b",
    re.I,
)


def parse_os(text: str, *, title: bool = False) -> str | None:
    """'Android 13' -> 'android', 'iOS 17' -> 'ios', 'KaiOS' -> 'other'.
    In a title, 'iPhone' also implies iOS (that is a fact, not a guess)."""
    text = text or ""
    a, i, o = _ANDROID.search(text), _IOS.search(text), _OTHER_OS.search(text)
    if title and re.search(r"\biphone\b", text, re.I):
        i = i or True
    hits = [name for name, hit in (("android", a), ("ios", i), ("other", o)) if hit]
    return hits[0] if len(hits) == 1 else None  # several OS names -> ambiguous


# --------------------------------------------------------------------------------------
# 5G
# --------------------------------------------------------------------------------------

_5G = re.compile(r"\b5G\b(?!\s*(?:hz|wi-?fi|wlan|band\s*wi))", re.I)
_NEG_5G = re.compile(r"\b(?:no|not|non|without|doesn'?t\s+support|does\s+not\s+support)"
                     r"\W+(?:\w+\W+){0,3}5G\b", re.I)
_NON_5G_NET = re.compile(r"\b(?:4G|LTE|3G|2G|GSM|CDMA|UMTS|HSPA\+?|WCDMA)\b", re.I)


def parse_5g(text: str) -> bool | None:
    """True if 5G is stated; False if a non-5G network is stated (or 5G is negated)."""
    text = text or ""
    if _NEG_5G.search(text):
        return False
    if _5G.search(text):
        return True
    if _NON_5G_NET.search(text):
        return False
    return None


# --------------------------------------------------------------------------------------
# Release year
# --------------------------------------------------------------------------------------

_YEAR = re.compile(r"\b(20[0-3]\d)\b")
_YEAR_TITLE = re.compile(
    r"\((?:[^()]*\b)?(20[0-3]\d)(?:\b[^()]*)?\)|\b(20[0-3]\d)\s+(?:model|edition|version|release)",
    re.I,
)
_YEAR_FEATURES = re.compile(r"(?:released?|launched?|model\s+year)\D{0,12}(20[0-3]\d)\b", re.I)


def _valid_year(y: int | None) -> int | None:
    return y if y is not None and YEAR_MIN <= y <= YEAR_MAX else None


def parse_year_value(value: str) -> int | None:
    """Details 'Date First Available' / 'Release date': 'March 3, 2023' -> 2023."""
    m = _YEAR.search(value or "")
    return _valid_year(int(m.group(1))) if m else None


def parse_year_title(text: str) -> int | None:
    """Only explicit year mentions: 'Galaxy A14 (2023)', '2022 model'."""
    m = _YEAR_TITLE.search(text or "")
    if not m:
        return None
    return _valid_year(int(m.group(1) or m.group(2)))


def parse_year_features(text: str) -> int | None:
    m = _YEAR_FEATURES.search(text or "")
    return _valid_year(int(m.group(1))) if m else None


# --------------------------------------------------------------------------------------
# details-key lookup
# --------------------------------------------------------------------------------------

DETAIL_KEYS: dict[str, re.Pattern] = {
    "ram": re.compile(r"\bram\b", re.I),
    "storage": re.compile(r"memory storage capacity|storage capacity|internal storage|"
                          r"flash memory size|hard disk size|^storage$|^rom$", re.I),
    "battery": re.compile(r"battery capacity|battery power|battery\s*$", re.I),
    "screen": re.compile(r"screen size|screen display size|display size", re.I),
    "camera": re.compile(r"photo sensor resolution|effective still resolution|"
                         r"rear camera|camera resolution|^camera$", re.I),
    "os": re.compile(r"^operating system$|^os$", re.I),
    "network": re.compile(r"cellular technology|wireless network technology|"
                          r"network technology|^network$", re.I),
    "year": re.compile(r"date first available|release date|year", re.I),
}


def detail_values(details: dict[str, Any], kind: str) -> list[str]:
    rx = DETAIL_KEYS[kind]
    out = []
    for k, v in details.items():
        if isinstance(k, str) and rx.search(k.strip()):
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            if isinstance(v, int | float) and not isinstance(v, bool):
                v = str(v)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out


def _first(values: list[str], fn: Callable[[str], Any]) -> Any:
    for v in values:
        r = fn(v)
        if r is not None:
            return r
    return None


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------


def parse_specs(details: dict[str, Any], title: str, features: list[str]
                ) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Return ``(specs, sources)``; ``specs`` has all 8 contract keys, ``sources[field]``
    is 'details' / 'title' / 'features' or None."""
    details = details or {}
    feats = " | ".join(features or [])
    specs: dict[str, Any] = dict.fromkeys(SPEC_FIELDS)
    sources: dict[str, str | None] = dict.fromkeys(SPEC_FIELDS)

    def put(field: str, value: Any, source: str) -> None:
        if specs[field] is None and value is not None:
            specs[field], sources[field] = value, source

    # --- memory (details keys first, then joint free-text parsing) ---
    put("ram_gb", _first(detail_values(details, "ram"), parse_ram_value), "details")
    put("storage_gb", _first(detail_values(details, "storage"), parse_storage_value), "details")
    for src, text in (("title", title), ("features", feats)):
        if specs["ram_gb"] is not None and specs["storage_gb"] is not None:
            break
        ram, storage = parse_memory(text)
        put("ram_gb", ram, src)
        put("storage_gb", storage, src)
    if (specs["ram_gb"] is not None and specs["storage_gb"] is not None
            and specs["ram_gb"] >= specs["storage_gb"]):
        specs["ram_gb"], sources["ram_gb"] = None, None

    # --- battery / screen / camera ---
    put("battery_mah", _first(detail_values(details, "battery"), parse_battery), "details")
    put("screen_in", _first(detail_values(details, "screen"), parse_screen_value), "details")
    put("main_camera_mp", _first(detail_values(details, "camera"), parse_camera), "details")
    for src, text in (("title", title), ("features", feats)):
        put("battery_mah", parse_battery(text), src)
        put("screen_in", parse_screen(text), src)
        put("main_camera_mp", parse_camera(text), src)

    # --- OS ---
    put("os", _first(detail_values(details, "os"), parse_os), "details")
    put("os", parse_os(title, title=True), "title")
    put("os", parse_os(feats), "features")

    # --- 5G: a positive statement anywhere wins; otherwise first explicit non-5G ---
    net_sources = [("details", " ".join(detail_values(details, "network"))),
                   ("title", title), ("features", feats)]
    for src, text in net_sources:
        if parse_5g(text) is True:
            put("has_5g", True, src)
            break
    else:
        for src, text in net_sources:
            put("has_5g", parse_5g(text), src)

    # --- release year ---
    put("release_year", _first(detail_values(details, "year"), parse_year_value), "details")
    put("release_year", parse_year_title(title), "title")
    put("release_year", parse_year_features(feats), "features")

    Specs.from_dict(specs)  # contract check: raises SchemaError if we produced garbage
    return specs, sources


# --------------------------------------------------------------------------------------
# Stage runner
# --------------------------------------------------------------------------------------


def run(s: Settings, force: bool = False) -> None:
    if not force and stage_is_cached(s, "parse_specs", [s.specs], [s.candidates]):
        print("[parse_specs] cached, skipping (use --force to rebuild)")
        return
    with AtomicJsonlWriter(s.specs) as w:
        for c in iter_jsonl(s.candidates):
            specs, sources = parse_specs(c["details"], c["title"], c["features"])
            w.write({"doc_id": c["doc_id"], "specs": specs, "sources": sources})
    mark_stage_done(s, "parse_specs")
    print(f"[parse_specs] parsed specs for {w.count} candidates -> {s.specs}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    run(Settings.from_config(), force=ap.parse_args().force)


if __name__ == "__main__":
    main()
