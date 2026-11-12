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
