"""Hand check of spec parsing and phone filtering (quality evidence for the report).

``make``  : samples kept phones -> ``handcheck_specs.csv`` (one row per phone x spec field)
            and near-threshold rejects -> ``handcheck_filter.csv``. Columns to fill are blank.
``score`` : reads the filled CSVs -> ``handcheck_summary.json``.

How to fill:
* handcheck_specs.csv, ``correct``: y / n for every row. For a ``null`` value write ``y`` if
  the listing really doesn't state it, ``n`` if the value was there and we missed it.
* handcheck_specs.csv, ``is_phone``: y / n, once per phone (any of its rows). This gives the
  filter precision on the kept sample.
* handcheck_filter.csv, ``is_phone``: y / n. A ``y`` is a false rejection.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from phonesense import io
from phonesense.data.common import iter_jsonl, write_json
from phonesense.schemas import SPEC_FIELDS

SPECS_COLUMNS = ["doc_id", "title", "raw_details", "field", "parsed_value", "source",
                 "is_phone", "correct", "notes"]
FILTER_COLUMNS = ["parent_asin", "title", "score", "reasons", "is_phone"]
YES = {"y", "yes", "1", "true", "t", "correct", "ok"}
NO = {"n", "no", "0", "false", "f", "wrong", "incorrect"}


def _write_csv(path: Path, columns: list[str], rows: list[dict], force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists (it may already be filled in by hand); "
                              "pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def make(*, phones_path: Path, phone_meta_path: Path, filter_log_path: Path,
         out_specs: Path, out_filter: Path, n_phones: int, n_rejects: int,
         threshold: float, margin: float, seed: int, force: bool = False) -> tuple[int, int]:
    rng = random.Random(seed)
    phones = sorted(io.read_phones(phones_path), key=lambda p: p.doc_id)
    meta = {m["doc_id"]: m for m in iter_jsonl(phone_meta_path)}
    chosen = rng.sample(phones, min(n_phones, len(phones)))
    rows = []
    for p in chosen:
        m = meta.get(p.doc_id, {})
        details = json.dumps(m.get("raw_details", {}), ensure_ascii=False,
                             separators=(",", ":"))
        sources = m.get("spec_sources", {})
        for f in SPEC_FIELDS:
            rows.append({"doc_id": p.doc_id, "title": p.title, "raw_details": details,
                         "field": f, "parsed_value": _fmt(getattr(p.specs, f)),
                         "source": sources.get(f) or "", "is_phone": "", "correct": "",
                         "notes": ""})
    _write_csv(out_specs, SPECS_COLUMNS, rows, force)

    near, near_any = [], []
    for e in iter_jsonl(filter_log_path):
        if not e["kept"] and threshold - margin <= e["score"] < threshold:
            (near if e.get("min_reviews_ok", True) else near_any).append(e)
    pool = near if len(near) >= n_rejects else near + near_any
    pool.sort(key=lambda e: e["parent_asin"])
    picked = rng.sample(pool, min(n_rejects, len(pool)))
    frows = [{"parent_asin": e["parent_asin"], "title": e.get("title", ""),
              "score": e["score"], "reasons": "; ".join(e["reasons"]), "is_phone": ""}
             for e in picked]
    _write_csv(out_filter, FILTER_COLUMNS, frows, force)
    print(f"[handcheck] {len(chosen)} phones ({len(rows)} rows) -> {out_specs}")
    print(f"[handcheck] {len(frows)} near-threshold rejects -> {out_filter}")
    return len(rows), len(frows)


def _label(v: str | None) -> bool | None:
    v = (v or "").strip().lower()
    if v in YES:
        return True
    if v in NO:
        return False
    return None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(*, specs_csv: Path, filter_csv: Path, out_path: Path) -> dict[str, Any]:
    with Path(specs_csv).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    is_phone: dict[str, bool] = {}
    for r in rows:
        f = r["field"]
        st = per_field[f]
        st["rows"] += 1
        parsed = r["parsed_value"].strip().lower() != "null"
        lab = _label(r["correct"])
        if parsed:
            st["parsed"] += 1
            if lab is not None:
                st["labelled"] += 1
                st["correct"] += lab
        else:
            st["null"] += 1
            if lab is False:
                st["null_missed"] += 1
            if lab is not None:
                st["null_labelled"] += 1
        ip = _label(r.get("is_phone"))
        if ip is not None:
            is_phone[r["doc_id"]] = ip

    def summarize(st: dict[str, int]) -> dict[str, Any]:
        return {"accuracy": _rate(st["correct"], st["labelled"]),
                "labelled_parsed_rows": st["labelled"], "parsed_rows": st["parsed"],
                "null_rate": _rate(st["null"], st["rows"]),
                "missed_rate_among_nulls": _rate(st["null_missed"], st["null_labelled"])}

    total: dict[str, int] = defaultdict(int)
    for st in per_field.values():
        for k, v in st.items():
            total[k] += v

    with Path(filter_csv).open(encoding="utf-8", newline="") as fh:
        frows = list(csv.DictReader(fh))
    flabels = [x for x in (_label(r.get("is_phone")) for r in frows) if x is not None]
    out = {
        "spec_accuracy_overall": summarize(total),
        "spec_accuracy_per_field": {f: summarize(per_field[f]) for f in SPEC_FIELDS
                                    if f in per_field},
        "phones_in_sample": len({r["doc_id"] for r in rows}),
        "filter_precision_kept_sample": _rate(sum(is_phone.values()), len(is_phone)),
        "filter_kept_labelled": len(is_phone),
        "false_rejection_rate_near_threshold": _rate(sum(flabels), len(flabels)),
        "near_threshold_rejects_labelled": len(flabels),
        "near_threshold_rejects_total": len(frows),
        "note": "accuracy counts only rows where a value was parsed and labelled; "
                "null_rate is over all rows of that field",
    }
    write_json(out_path, out)
    print(f"[handcheck] wrote {out_path}")
    return out
