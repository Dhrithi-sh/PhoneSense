import pytest

from phonesense import io
from phonesense.schemas import Phone, SchemaError


def test_valid(phone_dict):
    p = Phone.from_dict(phone_dict)
    assert p.specs.ram_gb == 8.0 and p.specs.has_5g is True
    assert Phone.from_dict(p.to_dict()) == p


def test_int_accepted_for_float_fields(phone_dict):
    phone_dict["specs"]["ram_gb"] = 8
    assert Phone.from_dict(phone_dict).specs.ram_gb == 8.0


@pytest.mark.parametrize("key,value", [
    ("ram_gb", "8"), ("ram_gb", "8 GB"), ("ram_gb", True), ("battery_mah", 5000.5),
    ("battery_mah", "5000mAh"), ("has_5g", "yes"), ("has_5g", 1), ("os", "Android"),
    ("os", "windows"), ("release_year", "2023"), ("release_year", 1995),
])
def test_rejects_wrong_spec_types(phone_dict, key, value):
    phone_dict["specs"][key] = value
    with pytest.raises(SchemaError):
        Phone.from_dict(phone_dict)


@pytest.mark.parametrize("key,value", [
    ("ram_gb", 0), ("storage_gb", 0), ("battery_mah", 0), ("ram_gb", -1),
    ("ram_gb", 8192),  # MB not converted to GB
])
def test_unknown_must_be_null_not_sentinel(phone_dict, key, value):
    phone_dict["specs"][key] = value
    with pytest.raises(SchemaError):
        Phone.from_dict(phone_dict)


@pytest.mark.parametrize("key,value", [("brand", ""), ("price", 0), ("price", "$299"),
                                       ("avg_rating", 0), ("rating_count", 1.0),
                                       ("currency", "usd"), ("doc_id", "B0 X"),
                                       ("text", "")])
def test_rejects_bad_top_level(phone_dict, key, value):
    phone_dict[key] = value
    with pytest.raises(SchemaError):
        Phone.from_dict(phone_dict)


def test_missing_spec_key_rejected(phone_dict):
    del phone_dict["specs"]["os"]
    with pytest.raises(SchemaError, match="missing"):
        Phone.from_dict(phone_dict)


def test_unknown_key_rejected(phone_dict):
    phone_dict["specs"]["weight_g"] = 180
    with pytest.raises(SchemaError, match="unknown"):
        Phone.from_dict(phone_dict)


def test_all_null_specs_ok(phone_dict):
    phone_dict["specs"] = {k: None for k in phone_dict["specs"]}
    assert Phone.from_dict(phone_dict).specs.os is None


def test_write_rejects_duplicate_doc_ids(tmp_path, phone_dict):
    p = Phone.from_dict(phone_dict)
    with pytest.raises(SchemaError, match="duplicate"):
        io.write_phones(tmp_path / "p.jsonl", [p, p])


def test_round_trip(tmp_path, fixtures):
    phones = io.read_phones(fixtures / "phones.jsonl")
    assert io.read_phones(io.write_phones(tmp_path / "p.jsonl", phones)) == phones
