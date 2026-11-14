"""Test format compliance against documentation/formats.md schema.

Verifies:
1. Every record contains all required fields with correct types.
2. Publication dates adhere strictly to ISO-8601 with +05:30 IST offset.
3. Strict privacy rule: zero author, byline, creator, or editor fields exist.
4. Geographic invariant: if city is not null, state must not be null.
5. All 300 records in data/news_sample_300.jsonl pass 100% contract validation.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import pytest

from dhvani.crawl.config import PRIMARY_SOURCES, PROJECT_ROOT, SAMPLE_ARTICLES_FILE
from dhvani.crawl.extractor import validate_article_schema


REQUIRED_FIELDS = {
    "doc_id": str,
    "url": str,
    "source": str,
    "section": str,
    "date": str,
    "headline": str,
    "body": str,
    "keywords": list,
    "agency_flag": bool,
    "content_hash": str,
    "links": list,
}

PROHIBITED_FIELDS = {"author", "authors", "byline", "creator", "editor", "writer"}
IST_OFFSET = "+05:30"


def check_record_compliance(record: dict) -> None:
    """Run comprehensive format and invariant checks on an article record."""
    # 1. Required fields presence and types
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in record, f"Missing required field: {field}"
        assert isinstance(record[field], expected_type), f"Field {field} is not {expected_type}"

    # 2. Nullable fields types
    assert "state" in record and (record["state"] is None or isinstance(record["state"], str))
    assert "city" in record and (record["city"] is None or isinstance(record["city"], str))
    assert "dup_of" in record and (record["dup_of"] is None or isinstance(record["dup_of"], str))

    # 3. Privacy constraint: zero author fields
    for forbidden in PROHIBITED_FIELDS:
        assert forbidden not in record, f"Privacy violation: prohibited field '{forbidden}' found"

    # 4. IST Date compliance
    date_str = record["date"]
    assert date_str.endswith(IST_OFFSET), f"Date '{date_str}' missing IST offset {IST_OFFSET}"
    try:
        dt = datetime.fromisoformat(date_str)
        assert dt.tzinfo is not None, "Date must be timezone-aware"
        expected_tz = timezone(timedelta(hours=5, minutes=30))
        assert dt.tzinfo.utcoffset(dt) == expected_tz.utcoffset(dt), "Timezone offset must be +05:30"
    except Exception as e:
        pytest.fail(f"Invalid ISO-8601 date string '{date_str}': {e}")

    # 5. Geographic consistency invariant: if city != null, state != null
    if record["city"] is not None:
        assert record["state"] is not None, f"Geographic invariant violated: city '{record['city']}' without state"

    # 6. Substantive content
    assert len(record["headline"].strip()) > 0, "Headline cannot be empty"
    assert len(record["body"].strip()) >= 40, "Body must contain substantive text (>= 40 chars)"

    # 7. MD5 content hash
    assert re.match(r"^[0-9a-f]{32}$", record["content_hash"]), "content_hash must be valid 32-char hex MD5"

    # 8. doc_id format
    assert re.match(r"^[a-z0-9_-]+_[a-z0-9_-]+$", record["doc_id"]), f"Invalid doc_id format: {record['doc_id']}"

    # 9. validate_article_schema validator
    assert validate_article_schema(record) is True, "validate_article_schema returned False"


def test_synthetic_record_compliance():
    """Verify synthetic valid and invalid records against compliance checks."""
    valid_record = {
        "doc_id": "jagran_12345678",
        "url": "https://www.jagran.com/uttar-pradesh/lucknow-weather-12345678.html",
        "source": "jagran",
        "section": "weather",
        "state": "uttar-pradesh",
        "city": "lucknow",
        "date": "2026-10-07T12:00:00+05:30",
        "headline": "लखनऊ में बारिश",
        "body": "लखनऊ और आसपास के इलाकों में भारी बारिश दर्ज की गई है। स्थानीय प्रशासन ने राहत कार्य शुरू कर दिए हैं।",
        "keywords": ["मौसम", "बारिश"],
        "agency_flag": False,
        "content_hash": "a" * 32,
        "dup_of": None,
        "links": [],
    }
    check_record_compliance(valid_record)

    # Prohibited field check
    invalid_record = dict(valid_record, author="Reporter Name")
    with pytest.raises(AssertionError, match="Privacy violation"):
        check_record_compliance(invalid_record)

    # Geographic inconsistency check
    invalid_geo = dict(valid_record, state=None, city="lucknow")
    with pytest.raises(AssertionError, match="Geographic invariant violated"):
        check_record_compliance(invalid_geo)

    # Non-IST date check
    invalid_date = dict(valid_record, date="2026-10-07T12:00:00Z")
    with pytest.raises(AssertionError, match="missing IST offset"):
        check_record_compliance(invalid_date)


def test_sample_300_file_compliance():
    """Validate that every single record in data/news_sample_300.jsonl passes all compliance checks."""
    sample_path = SAMPLE_ARTICLES_FILE
    if not sample_path.exists():
        pytest.skip(f"Sample file not found at {sample_path}")

    lines = sample_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 300, f"Expected 300 records in sample, found {len(lines)}"

    for i, line in enumerate(lines, start=1):
        record = json.loads(line)
        try:
            check_record_compliance(record)
        except AssertionError as e:
            pytest.fail(f"Record #{i} (doc_id={record.get('doc_id')}) failed compliance: {e}")


def test_full_corpus_dedup_file_compliance():
    """Validate that every single record in data/news_dedup.jsonl passes all compliance checks and graph invariants."""
    dedup_path = PROJECT_ROOT / "data" / "news_dedup.jsonl"
    if not dedup_path.exists():
        pytest.skip(f"Deduplicated corpus file not found at {dedup_path}")

    articles = []
    with open(dedup_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                check_record_compliance(rec)
            except AssertionError as e:
                pytest.fail(f"Record #{idx} ({rec.get('doc_id')}) failed compliance: {e}")
            articles.append(rec)

    assert len(articles) >= 5000, f"Expected >= 5000 articles, got {len(articles)}"

    # Invariant checks across full corpus
    doc_ids = {a["doc_id"] for a in articles}
    for idx, rec in enumerate(articles, 1):
        if rec["dup_of"] is not None:
            assert rec["dup_of"] in doc_ids, f"Record #{idx} ({rec['doc_id']}) dup_of pointer not in corpus"
            assert rec["dup_of"] != rec["doc_id"], f"Record #{idx} ({rec['doc_id']}) dup_of points to itself"
        for link_target in rec.get("links", []):
            assert link_target in doc_ids, f"Record #{idx} ({rec['doc_id']}) dangling link {link_target}"

