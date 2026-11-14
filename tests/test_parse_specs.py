"""T02 spec parser: units, RAM vs storage, impossible values, source priority."""

import pytest

from phonesense.data.parse_specs import (
    parse_5g,
    parse_battery,
    parse_camera,
    parse_memory,
    parse_os,
    parse_ram_value,
    parse_screen,
    parse_screen_value,
    parse_specs,
    parse_storage_value,
    parse_year_title,
    parse_year_value,
)
from phonesense.schemas import SPEC_FIELDS, Specs


@pytest.mark.parametrize("text,ram,storage", [
    ("8GB RAM 128GB", 8.0, 128.0),
    ("128GB 8GB RAM", 8.0, 128.0),
    ("4GB/64GB", 4.0, 64.0),
    ("8GB+256GB", 8.0, 256.0),
    ("4/64GB", 4.0, 64.0),
    ("6GB RAM + 128GB ROM, expandable up to 1TB microSD", 6.0, 128.0),
    ("Galaxy A14 64GB (Expandable up to 1TB)", None, 64.0),
    ("128GB storage, 8 GB of RAM", 8.0, 128.0),
    ("Unlocked 64GB 4GB Black", 4.0, 64.0),
    ("1TB", None, 1024.0),
    ("512MB", None, 0.5),
    ("Samsung Galaxy S23 128GB Unlocked", None, 128.0),
    ("Nokia 105 dual SIM", None, None),
    ("48GB RAM 256GB", None, 256.0),          # RAM > 32 GB is impossible -> null
    ("8GB 64GB 256GB 512GB", None, None),     # too many sizes: ambiguous -> null
])
def test_memory(text, ram, storage):
    assert parse_memory(text) == (ram, storage)


@pytest.mark.parametrize("value,expected", [
    ("8 GB", 8.0), ("6144 MB", 6.0), ("64 GB", None), ("8", None),
])
def test_ram_details_value(value, expected):
    assert parse_ram_value(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("128 GB", 128.0), ("1 TB", 1024.0), ("512 MB", 0.5), ("8 TB", None),
])
def test_storage_details_value(value, expected):
    assert parse_storage_value(value) == expected


@pytest.mark.parametrize("text,expected", [
    ("6.1 Inches", 6.1), ('6.7"', 6.7), ("6.5-inch FHD+ display", 6.5),
    ("16.5 cm display", 6.5), ("12.9 inch tablet", None), ("2 inches", None),
])
def test_screen(text, expected):
    assert parse_screen(text) == expected


def test_screen_details_value_without_unit():
    assert parse_screen_value("6.1") == 6.1
    assert parse_screen_value("6.1 Inches") == 6.1


@pytest.mark.parametrize("text,expected", [
    ("5000mAh", 5000), ("5,000 mAh", 5000), ("4500 Milliamp Hours", 4500),
    ("300mAh", None), ("20000 mAh power bank", None), ("15.4 Wh", None),
])
def test_battery(text, expected):
    assert parse_battery(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("48MP + 12MP", 48.0),
    ("50MP main, 8MP ultrawide, 32MP front camera", 50.0),
    ("16MP selfie, 13MP rear", 13.0),
    ("200 megapixel camera", 200.0),
    ("no camera info", None),
])
def test_camera(text, expected):
    assert parse_camera(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Android 13", "android"), ("iOS 17", "ios"), ("KaiOS", "other"),
    ("Windows Phone 8", "other"), ("Octa-core", None),
])
def test_os(text, expected):
    assert parse_os(text) == expected


def test_os_iphone_title():
    assert parse_os("Apple iPhone 13, 128GB", title=True) == "ios"
    assert parse_os("Apple iPhone 13, 128GB") is None


@pytest.mark.parametrize("text,expected", [
    ("5G", True), ("Galaxy A14 5G Unlocked", True), ("4G LTE", False),
    ("dual band 5GHz WiFi", None), ("not compatible with 5G networks", False),
    ("Octa-core", None),
])
def test_5g(text, expected):
    assert parse_5g(text) is expected


def test_years():
    assert parse_year_value("March 16, 2023") == 2023
    assert parse_year_value("unknown") is None
    assert parse_year_title("Samsung Galaxy A14 5G (2023), 64GB") == 2023
    assert parse_year_title("Moto G 2023 model") == 2023
    assert parse_year_title("Galaxy S23 128GB") is None


def test_parse_specs_priority_and_sources():
    details = {"RAM": "6 GB", "Memory Storage Capacity": "128 GB",
               "Operating System": "Android 13", "Date First Available": "March 16, 2023"}
    title = "Galaxy A54 5G 8GB+256GB 6.4 inch"
    features = ["5000mAh battery", "50MP + 12MP camera"]
    specs, sources = parse_specs(details, title, features)
    assert specs == {"ram_gb": 6.0, "storage_gb": 128.0, "battery_mah": 5000,
                     "screen_in": 6.4, "main_camera_mp": 50.0, "os": "android",
                     "has_5g": True, "release_year": 2023}
    assert sources == {"ram_gb": "details", "storage_gb": "details", "battery_mah": "features",
                       "screen_in": "title", "main_camera_mp": "features", "os": "details",
                       "has_5g": "title", "release_year": "details"}
    Specs.from_dict(specs)


def test_parse_specs_unknown_is_null():
    specs, sources = parse_specs({}, "Mystery phone", [])
    assert set(specs) == set(SPEC_FIELDS)
    assert all(v is None for v in specs.values())
    assert all(v is None for v in sources.values())


def test_impossible_details_values_are_null():
    details = {"RAM": "64 GB", "Battery Capacity": "50000 mAh", "Screen Size": "10.1 Inches"}
    specs, _ = parse_specs(details, "Phone", [])
    assert specs["ram_gb"] is None and specs["battery_mah"] is None
    assert specs["screen_in"] is None
