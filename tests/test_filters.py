import pytest

from phonesense.schemas import FILTER_FIELDS, Query, SchemaError, validate_filter


def leaf(field, op, value):
    return {"field": field, "op": op, "value": value}


def test_nested_and_or_not():
    f = {"and": [leaf("ram_gb", ">=", 8),
                 {"or": [leaf("has_5g", "==", True),
                         {"not": leaf("brand", "in", ["Nova", "Pebble"])}]},
                 {"not": {"and": [leaf("price", ">", 500), leaf("os", "==", "ios")]}}]}
    assert validate_filter(f) is f


def test_every_field_accepts_equality():
    sample = {"number": 1, "string": "android", "bool": True}
    for field, kind in FILTER_FIELDS.items():
        validate_filter(leaf(field, "==", sample[kind]))


@pytest.mark.parametrize("f", [
    leaf("weight_g", "<", 200),                     # unknown field
    leaf("ram_gb", "~=", 8),                        # unknown op
    leaf("ram_gb", "contains", 8),                  # unknown op
    {"xor": [leaf("ram_gb", ">=", 8)]},             # unknown combinator
    {"and": []},                                    # empty list
    {"and": leaf("ram_gb", ">=", 8)},               # not a list
    {"not": [leaf("ram_gb", ">=", 8)]},             # not takes one object
    {"and": [leaf("ram_gb", ">=", 8)], "or": [leaf("ram_gb", ">=", 8)]},
    leaf("ram_gb", ">=", "8"),                      # wrong value type
    leaf("has_5g", ">", True),                      # ordering on bool
    leaf("has_5g", "==", 1),                        # int is not bool
    leaf("brand", "<", "M"),                        # ordering on string
    leaf("os", "==", "windows"),                    # os enum
    leaf("ram_gb", "in", []),                       # empty in-list
    leaf("ram_gb", "in", 8),                        # in needs a list
    leaf("ram_gb", ">=", None),                     # null values not allowed
    {"field": "ram_gb", "op": ">="},                # missing value
    {"field": "ram_gb", "op": ">=", "value": 8, "extra": 1},
    "ram_gb >= 8",                                  # string syntax not allowed
])
def test_rejects(f):
    with pytest.raises(SchemaError):
        validate_filter(f)


def test_query_validates_filters():
    q = {"qid": "q07", "text": "x", "split": "dev", "filters": leaf("nope", "==", 1)}
    with pytest.raises(SchemaError):
        Query.from_dict(q)


@pytest.mark.parametrize("patch", [{"qid": "q7"}, {"qid": "Q07"}, {"split": "train"},
                                   {"text": ""}, {"intent_notes": 3}, {"extra": 1}])
def test_query_rejects(patch):
    q = {"qid": "q07", "text": "x", "split": "dev", **patch}
    with pytest.raises(SchemaError):
        Query.from_dict(q)
