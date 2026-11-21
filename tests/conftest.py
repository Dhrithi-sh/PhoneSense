import copy
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def phone_dict() -> dict:
    with (FIXTURES / "phones.jsonl").open(encoding="utf-8") as fh:
        return copy.deepcopy(json.loads(fh.readline()))


@pytest.fixture
def make_phone():
    """Factory for minimal valid Phone records (specs unknown) from title/features/description."""
    from phonesense.schemas import Phone, PhoneFields, build_text

    def _make(doc_id: str, title: str, features=(), description: str = "") -> Phone:
        fields = PhoneFields(title=title, features=list(features), description=description)
        return Phone.from_dict({
            "doc_id": doc_id, "title": title, "brand": None, "price": None, "currency": "USD",
            "avg_rating": None, "rating_count": 0,
            "specs": {k: None for k in ("ram_gb", "storage_gb", "battery_mah", "screen_in",
                                        "main_camera_mp", "os", "has_5g", "release_year")},
            "text": build_text(fields), "fields": fields.to_dict(),
        })

    return _make
