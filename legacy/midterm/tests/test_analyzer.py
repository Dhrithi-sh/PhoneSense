import pytest

from text.analyzer import analyze


def test_analyze_none_mode():
    result = analyze("लड़कियाँ किताबों", "none")

    assert result == [
        ("लड़कियाँ", 0),
        ("किताबों", 1),
    ]


def test_analyze_light_mode():
    result = analyze("लड़कियाँ किताबों", "light")

    assert result == [
        ("लड़क", 0),
        ("किताब", 1),
    ]


def test_analyze_aggressive_mode():
    result = analyze("लड़कियाँ किताबों", "aggr")

    assert result == [
        ("लड़क", 0),
        ("किताब", 1),
    ]


def test_analyze_preserves_positions():
    result = analyze(
        "दिल्ली में आज बारिश हुई",
        "none",
    )

    assert result == [
        ("दिल्ली", 0),
        ("में", 1),
        ("आज", 2),
        ("बारिश", 3),
        ("हुई", 4),
    ]


def test_analyze_normalizes_before_tokenizing():
    result = analyze(
        "हिंदी   में बारिश",
        "none",
    )

    assert result == [
        ("हिन्दी", 0),
        ("में", 1),
        ("बारिश", 2),
    ]


def test_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        analyze("बारिश", "invalid")