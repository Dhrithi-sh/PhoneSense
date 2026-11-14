"""Stage 2: keep real smartphones, reject accessories, with a transparent rule-based score.

Every raw item gets a score from positive and negative signals (see ``score_item``) and
is kept if ``score >= pipeline.filter_threshold``. Then items with fewer than
``pipeline.min_reviews`` ratings (dataset ``rating_number``) are dropped early; the exact
review count is enforced again later in build_phones. Every decision and its reasons go to
``data/processed/filter_log.jsonl``.

Run: ``python -m phonesense.data.filter_phones [--force] [--limit N]``
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from phonesense.data.common import (
    AtomicJsonlWriter,
    Settings,
    as_dict,
    as_str_list,
    clean_text,
    iter_raw_records,
    mark_stage_done,
    stage_is_cached,
    write_json,
)

# --------------------------------------------------------------------------------------
# Signals (weights are part of the method; the threshold lives in config.yaml)
# --------------------------------------------------------------------------------------

ROOT_CATEGORY = "cell phones & accessories"
# Category path elements that mean "this is a phone" (Amazon files trade-in phones and an
# "Android 101" shelf next to "Cell Phones").
PHONE_CATEGORY_RX = re.compile(
    r"^(?:(?:unlocked |carrier |prepaid )?cell phones|smartphones|android 101"
    r"|(?:unlocked |carrier branded )?wireless trade-in.*)$",
    re.I,
)
# Category path elements that mean "accessory / other device". Other categories (carrier
# or brand shelves such as "Nokia", "Verizon Wireless") are neutral.
ACCESSORY_CATEGORY_RX = re.compile(
    r"accessor|case|holster|sleeve|protector|charger|cable|adapter|mount|stand|grip|"
    r"head(?:set|phone)|earbud|speaker|batter|repair|maintenance|smartwatch|wearable|"
    r"tablet|stylus|skin|decal|d.cor|bag|armband|lens|memory card|sim card|booster|bundle|"
    r"virtual reality|photo|video|automobile|selfie",
    re.I,
)

PHONE_BRANDS = (
    "apple", "samsung", "google", "motorola", "moto", "oneplus", "nokia", "lg", "xiaomi",
    "redmi", "poco", "blu", "tcl", "sony", "huawei", "honor", "zte", "alcatel", "oppo",
    "realme", "vivo", "asus", "nothing", "ulefone", "doogee", "blackview", "oukitel", "cubot",
    "umidigi", "fairphone", "kyocera", "htc", "blackberry", "cat", "sonim", "lenovo",
    "infinix", "tecno", "itel", "meizu", "unihertz", "agm", "hotwav", "fossibot",
    "nuu", "orbic",
)
_BRAND_ALT = "|".join(re.escape(b) for b in sorted(PHONE_BRANDS, key=len, reverse=True))

# Brand + model line, e.g. "iPhone 13", "Galaxy S23", "Pixel 7", "Moto G Power".
MODEL_RX = re.compile(
    r"\b(iphone\s*(?:\d{1,2}|se|x[rs]?)\b|galaxy\s*(?:[sazmjf]\d{1,3}|note\s*\d+|z\s*(?:flip|fold))"
    r"|pixel\s*\d|moto(?:rola)?\s*(?:[gez]\d*\b|edge|razr|one)|oneplus\s*(?:\d|nord)"
    r"|redmi\s*(?:note\s*)?\d|poco\s*[xfm]\d|xiaomi\s*(?:\d|mi\s*\d)|nokia\s*[gcx]?\d"
    r"|lg\s*(?:g\d|v\d|stylo|k\d|velvet|wing)|blu\s*(?:g\d|c\d|view|vivo|studio|bold)"
    r"|tcl\s*\d|xperia|huawei\s*(?:p\d|mate|nova|y\d)|zte\s*(?:blade|axon)"
    r"|alcatel\s*\w+|asus\s*(?:rog|zenfone)|nothing\s*phone|oppo\s*\w+|realme\s*\w+"
    r"|ulefone\s*(?:armor|note|power)|doogee\s*[sxnv]\d|blackview\s*\w+|oukitel\s*\w+)",
    re.I,
)
SMARTPHONE_RX = re.compile(r"\bsmart\s?phones?\b|\bcell\s?phones?\b|\bmobile\s+phones?\b", re.I)
UNLOCKED_RX = re.compile(r"\bunlocked\b", re.I)
NETWORK_RX = re.compile(r"\b5G\b(?!\s*hz)|\b4G\s*LTE\b|\bLTE\b", re.I)
GB_RX = re.compile(r"\b\d+\s*(?:GB|TB)\b", re.I)

# Strong accessory evidence: "case for iPhone 15", "compatible with Galaxy S23".
FOR_BRAND_RX = re.compile(
    rf"\b(?:for|fits|fit)\s+(?:the\s+|all\s+|your\s+|new\s+)?"
    rf"(?:{_BRAND_ALT}|iphone|galaxy|pixel)\b",
    re.I,
)
# "compatible with Galaxy S23" (accessory) but not "not compatible with Verizon" (a phone
# describing its network support): a device brand must follow, and no negation precede.
COMPATIBLE_RX = re.compile(
    rf"(?<!not )(?<!non-)\b(?:compatible\s+(?:with|for)|designed\s+for|works\s+with)\s+"
    rf"(?:the\s+|all\s+|most\s+)?(?:{_BRAND_ALT}|iphone|galaxy|pixel|android\s+phones?)\b",
    re.I,
)
ACCESSORY_TERMS = (
    "case", "cases", "cover", "covers", "screen protector", "protector", "tempered glass",
    "charger", "charging pad", "charging station", "charging dock", "cable", "cord",
    "adapter", "holder", "mount", "skin", "skins",
    "lens protector", "camera protector", "stylus", "pen", "replacement battery",
    "battery pack", "power bank", "sim card", "sim tray", "sim adapter", "earbuds",
    "earphones", "headphones", "headset", "speaker", "wallet", "pouch", "holster", "strap",
    "lanyard", "ring light", "tripod", "selfie stick", "gimbal", "dock", "stand", "bumper",
    "film", "decal", "sticker", "grip", "popsocket", "kickstand", "repair kit",
    "replacement screen", "lcd assembly", "digitizer", "back glass", "housing",
    "flex cable", "charging port", "signal booster", "antenna", "keyboard", "car kit",
    "cleaning kit", "armband", "sleeve", "bag", "magsafe", "lens", "lenses",
)
# Phone names / features that contain accessory words ("Moto G Stylus", "with S Pen").
ACCESSORY_EXCEPTIONS_RX = re.compile(
    r"\bg\s*stylus\b|\bstylo\b|\bs\s?-?pen(?:\s+stylus)?\b|\bbuilt-?in\s+(?:stylus|pen)\b"
    r"|\bgorilla\s+glass\b|\bglass\s+back\b|\b(?:qwerty|t9)\s+(?:keyboard|keypad)\b"
    r"|\bkeyboard\s+phone\b",
    re.I,
)
# Other devices sold in the same category.
OTHER_DEVICE_RX = re.compile(
    r"\bhot\s?spot\b|\bmifi\b|\brouter\b|\bmodem\b|\barduino\b|\bdash\s*cam\b"
    r"|\bfitness\s+monitor\b|\bwalkie[-\s]talkie\b|\bcordless\s+phone\b|\blandline\b",
    re.I,
)
# A bare "phone" word ("Samsung Highlight t749 Phone (T-Mobile)"), weak evidence.
PHONE_WORD_RX = re.compile(r"(?<!for )(?<!to )\bphone\b(?!\s+(?:case|holder|mount|stand|"
                           r"grip|strap|charger|cable|lens|pouch|wallet|ring))", re.I)
ACCESSORY_RX = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ACCESSORY_TERMS) + r")\b",
                          re.I)
TABLET_RX = re.compile(r"\btablets?\b|\bipad\b|\bgalaxy\s*tab\b|\btab\s*[a-z]?\d|\be-?reader\b",
                       re.I)
WATCH_RX = re.compile(
    r"\bsmart\s?watch(?:es)?\b|\bwatch\b|\bfitness\s+tracker\b|\bsmart\s*band\b|\bwearable\b"
    r"|\bgps\s+tracker\b",
    re.I,
)
NOT_A_DEVICE_RX = re.compile(
    r"\bdummy\b|\bdisplay\s+model\b|\bfake\s+phone\b|\btoy\b|\bfor\s+parts\b|\bnon-?working\b"
    r"|\bmockup\b|\bprop\b",
    re.I,
)
LOT_RX = re.compile(r"\blot\s+of\s+\d+|\b\d+\s*(?:pcs|pack|-pack|units)\b|\bbulk\b|\bwholesale\b",
                    re.I)
BASIC_PHONE_RX = re.compile(
    r"\bflip\s+phone\b|\bfeature\s+phone\b|\bbasic\s+phone\b|\bsenior\s+(?:cell\s+)?phone\b"
    r"|\bbig\s+button\b|\bkids?\s+phone\b",
    re.I,
)
REFURB_RX = re.compile(r"\b(?:renewed|refurbished|pre-?owned|reconditioned)\b", re.I)

DETAIL_SIGNALS = (
    ("details_os", re.compile(r"^operating system$|^os$", re.I), 1.0),
    ("details_ram", re.compile(r"\bram\b", re.I), 1.0),
    ("details_storage", re.compile(r"memory storage capacity|storage capacity", re.I), 1.0),
    ("details_screen", re.compile(r"screen size|screen display size", re.I), 0.5),
    ("details_cellular", re.compile(r"cellular technology|wireless network technology", re.I),
     0.5),
    ("details_talk_time", re.compile(r"talk time|standby time", re.I), 0.5),
)
COMPATIBLE_DETAIL_RX = re.compile(r"compatible (?:phone )?models?|compatible devices", re.I)


def is_refurbished(title: str) -> bool:
    return bool(REFURB_RX.search(title or ""))


def score_item(item: dict[str, Any]) -> tuple[float, list[str]]:
    """Return ``(score, reasons)`` for one raw meta record. Reasons look like
    ``"+3 category:Cell Phones"`` so the log is self-explanatory."""
    title = clean_text(item.get("title"))
    cats = [c.lower() for c in as_str_list(item.get("categories"))]
    details = as_dict(item.get("details"))
    score, reasons = 0.0, []

    def add(w: float, why: str) -> None:
        nonlocal score
        score += w
        reasons.append(f"{w:+g} {why}")

    # Category path
    if any(PHONE_CATEGORY_RX.match(c) for c in cats):
        add(3, "category:cell_phones")
    acc_cats = [c for c in cats if c != ROOT_CATEGORY and not PHONE_CATEGORY_RX.match(c)
                and ACCESSORY_CATEGORY_RX.search(c)]
    if acc_cats:
        add(-3, f"category:{acc_cats[0][:40]}")

    # Details keys that phones have and accessories rarely do
    keys = [k.strip() for k in details if isinstance(k, str)]
    for name, rx, w in DETAIL_SIGNALS:
        if any(rx.search(k) for k in keys):
            add(w, name)
    if any(COMPATIBLE_DETAIL_RX.search(k) for k in keys):
        add(-2, "details_compatible_models")

    # Title patterns
    if MODEL_RX.search(title):
        add(2, "title_model")
    if SMARTPHONE_RX.search(title):
        add(2, "title_smartphone")
    elif PHONE_WORD_RX.search(title):
        add(1, "title_phone_word")
    if UNLOCKED_RX.search(title):
        add(1, "title_unlocked")
    if NETWORK_RX.search(title):
        add(0.5, "title_network")
    if GB_RX.search(title):
        add(1, "title_gb")

    # Negative title patterns
    if FOR_BRAND_RX.search(title):
        add(-5, "title_for_brand")
    if COMPATIBLE_RX.search(title):
        add(-5, "title_compatible_with")
    acc_title = ACCESSORY_EXCEPTIONS_RX.sub(" ", title)
    acc = {m.group(0).lower() for m in ACCESSORY_RX.finditer(acc_title)}
    if acc:
        add(-4 * min(len(acc), 2), "title_accessory:" + ",".join(sorted(acc))[:60])
    if TABLET_RX.search(title):
        add(-5, "title_tablet")
    if WATCH_RX.search(title):
        add(-5, "title_watch")
    if OTHER_DEVICE_RX.search(title):
        add(-5, "title_other_device")
    if NOT_A_DEVICE_RX.search(title):
        add(-6, "title_not_working_device")
    if LOT_RX.search(title):
        add(-3, "title_lot")
    if BASIC_PHONE_RX.search(title):
        add(-2, "title_basic_phone")
    if not title:
        add(-10, "no_title")
    return score, reasons


def _rating_number(item: dict) -> int | None:
    v = item.get("rating_number")
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return int(v)
    if isinstance(v, str) and v.replace(",", "").isdigit():
        return int(v.replace(",", ""))
    return None


def candidate_record(item: dict, score: float) -> dict[str, Any]:
    """The fields later stages need, cleaned. Interim format, not the contract."""
    title = clean_text(item.get("title"))
    details = as_dict(item.get("details"))
    return {
        "doc_id": item["parent_asin"],
        "title": title,
        "store": clean_text(item.get("store")) or None,
        "details": details,
        "features": as_str_list(item.get("features")),
        "description": " ".join(as_str_list(item.get("description"))),
        "categories": as_str_list(item.get("categories")),
        "main_category": clean_text(item.get("main_category")) or None,
        "price": item.get("price"),
        "average_rating": item.get("average_rating"),
        "rating_number": _rating_number(item),
        "score": score,
        "refurbished": is_refurbished(title),
    }


def run(s: Settings, force: bool = False) -> None:
    outputs = [s.candidates, s.filter_log, s.filter_summary]
    if not force and stage_is_cached(s, "filter_phones", outputs):
        print("[filter_phones] cached, skipping (use --force to rebuild)")
        return
    raw = kept = 0
    seen: set[str] = set()
    reasons_count: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    near_floor = s.threshold - s.handcheck_margin
    with AtomicJsonlWriter(s.candidates) as cand, AtomicJsonlWriter(s.filter_log) as log:
        for item in iter_raw_records(s.raw_meta, limit=s.limit, desc="filter meta",
                                     progress=s.progress):
            raw += 1
            asin = item.get("parent_asin")
            if not isinstance(asin, str) or not asin.strip() or any(c.isspace() for c in asin):
                decisions["rejected:no_parent_asin"] += 1
                continue
            if asin in seen:  # parent_asin should be unique in meta; keep the first
                decisions["rejected:duplicate_parent_asin"] += 1
                continue
            seen.add(asin)
            score, reasons = score_item(item)
            n_ratings = _rating_number(item)
            enough = n_ratings is None or n_ratings >= s.min_reviews
            is_phone = score >= s.threshold
            ok = is_phone and enough
            if is_phone and not enough:
                reasons = [*reasons, f"min_reviews:rating_number={n_ratings}<{s.min_reviews}"]
                decisions["rejected:min_reviews"] += 1
            elif not is_phone:
                decisions["rejected:score"] += 1
                for r in reasons:
                    if r.startswith("-"):
                        reasons_count[r.split(" ", 1)[1].split(":")[0]] += 1
            else:
                decisions["kept"] += 1
            entry = {"parent_asin": asin, "kept": ok, "score": score, "reasons": reasons,
                     "min_reviews_ok": enough}
            if ok or score >= near_floor:
                entry["title"] = clean_text(item.get("title"))[:200]
            log.write(entry)
            if ok:
                kept += 1
                cand.write(candidate_record(item, score))
    write_json(s.filter_summary, {
        "raw_items": raw, "kept": kept, "decisions": dict(decisions),
        "negative_signal_counts_among_score_rejects": dict(reasons_count.most_common()),
        "threshold": s.threshold, "min_reviews": s.min_reviews,
    })
    mark_stage_done(s, "filter_phones")
    print(f"[filter_phones] {raw} raw items -> {kept} phone candidates "
          f"({dict(decisions)})")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(Settings.from_config(limit=a.limit), force=a.force)


if __name__ == "__main__":
    main()
