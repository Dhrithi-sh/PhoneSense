"""Tests for RobotsParser and comparison against standard urllib.robotparser.

Verifies:
1. Shortcomings of standard urllib.robotparser on wildcard patterns (* and $).
2. Correct RFC 9309 compliance in dhvani.crawl.robots.RobotsParser.
3. User-Agent fallback to '*'.
4. Sitemap: extraction across all 5 primary target sites.
5. In-memory per-host caching.
"""

from pathlib import Path
import urllib.robotparser as ufp
import pytest

from dhvani.crawl.robots import RobotsParser

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "robots"


@pytest.fixture
def parser():
    return RobotsParser(user_agent="CollegeProject_NewsBot(+https://github.com/InformationRetrieval-Mid/Dhvani-lingual; contact: ra810@snu.edu.in)")


def read_fixture(filename: str) -> str:
    path = FIXTURES_DIR / filename
    assert path.exists(), f"Fixture file not found: {path}"
    return path.read_text(encoding="utf-8")


# ==============================================================================
# 1. Comparison Tests: urllib.robotparser Failure Cases on Target Sites
# ==============================================================================

def test_urllib_vs_custom_on_jagran_wildcards(parser):
    """Demonstrate that standard urllib.robotparser fails to block wildcard patterns

    on Dainik Jagran, whereas RobotsParser handles them correctly.
    """
    content = read_fixture("jagran_robots.txt")
    parser.set_cached_rules("www.jagran.com", content)

    # Standard urllib parser
    standard_rp = ufp.RobotFileParser()
    standard_rp.parse(content.splitlines())

    # Disallowed wildcard patterns in Jagran's robots.txt:
    # Disallow: /search/*
    # Disallow: /delhi/video/
    # Disallow: /state/*
    test_urls = [
        "https://www.jagran.com/search/weather-lucknow",
        "https://www.jagran.com/delhi/video/",
        "https://www.jagran.com/state/lucknow",
    ]

    for url in test_urls:
        # Standard library fails: allows paths that should be blocked
        urllib_allowed = standard_rp.can_fetch(parser.user_agent, url)
        # Custom parser correctly blocks them
        custom_allowed = parser.can_fetch(url)

        assert urllib_allowed is True, f"urllib unexpectedly blocked {url}"
        assert custom_allowed is False, f"RobotsParser failed to block disallowed URL: {url}"


def test_urllib_vs_custom_on_amarujala_query_wildcards(parser):
    """Demonstrate that standard urllib.robotparser fails on query-string wildcards

    on Amar Ujala (/*?utm_*), whereas RobotsParser correctly blocks them.
    """
    content = read_fixture("amarujala_robots.txt")
    parser.set_cached_rules("www.amarujala.com", content)

    standard_rp = ufp.RobotFileParser()
    standard_rp.parse(content.splitlines())

    # Disallow: /*?utm_*
    # Disallow: /tags/sex*
    blocked_urls = [
        "https://www.amarujala.com/lucknow/news?utm_source=facebook",
        "https://www.amarujala.com/tags/sex123",
    ]

    for url in blocked_urls:
        urllib_allowed = standard_rp.can_fetch(parser.user_agent, url)
        custom_allowed = parser.can_fetch(url)

        # Standard library mistakenly allows them
        assert urllib_allowed is True
        # Custom parser correctly blocks them
        assert custom_allowed is False


# ==============================================================================
# 2. Functional Tests for RobotsParser on Target Sites
# ==============================================================================

def test_jagran_allowed_articles(parser):
    """Ensure legitimate news article URLs are allowed on Dainik Jagran."""
    content = read_fixture("jagran_robots.txt")
    parser.set_cached_rules("www.jagran.com", content)

    allowed_urls = [
        "https://www.jagran.com/news/national-weather-update-23456789.html",
        "https://www.jagran.com/uttar-pradesh/lucknow-city-news-23450001.html",
        "https://www.jagran.com/",
    ]
    for url in allowed_urls:
        assert parser.can_fetch(url) is True, f"Expected allowed for {url}"


def test_amarujala_allowed_articles(parser):
    """Ensure legitimate news article URLs are allowed on Amar Ujala."""
    content = read_fixture("amarujala_robots.txt")
    parser.set_cached_rules("www.amarujala.com", content)

    allowed_urls = [
        "https://www.amarujala.com/lucknow/weather-rain-update",
        "https://www.amarujala.com/india-news/national-affairs",
        "https://www.amarujala.com/.well-known/amphtml/apikey.pub",
    ]
    for url in allowed_urls:
        assert parser.can_fetch(url) is True, f"Expected allowed for {url}"


def test_sitemap_extraction_all_primary_sites(parser):
    """Ensure sitemaps are correctly extracted from all 5 primary sites' robots.txt."""
    fixtures = {
        "www.jagran.com": "jagran_robots.txt",
        "navbharattimes.indiatimes.com": "nbt_robots.txt",
        "www.livehindustan.com": "livehindustan_robots.txt",
        "www.amarujala.com": "amarujala_robots.txt",
        "www.aajtak.in": "aajtak_robots.txt",
    }

    for host, filename in fixtures.items():
        content = read_fixture(filename)
        parser.set_cached_rules(host, content)
        sitemaps = parser.get_sitemaps(host)

        assert len(sitemaps) > 0, f"No sitemaps extracted from {host}"
        for sm in sitemaps:
            assert sm.startswith("http"), f"Invalid sitemap URL in {host}: {sm}"
            assert "sitemap" in sm.lower(), f"Sitemap URL missing keyword: {sm}"


def test_user_agent_fallback_to_star(parser):
    """Ensure rule matching falls back to User-agent: * when custom bot is not listed."""
    robots_text = """
    User-agent: Googlebot
    Disallow: /private/

    User-agent: *
    Disallow: /admin/
    Allow: /
    """
    rules = parser.parse_content(robots_text)
    assert rules.can_fetch("/admin/panel") is False
    assert rules.can_fetch("/private/data") is True
    assert rules.can_fetch("/public/page") is True


def test_longest_match_precedence(parser):
    """RFC 9309 longest matching rule: longer Allow overrides shorter Disallow."""
    robots_text = """
    User-agent: *
    Disallow: /news/
    Allow: /news/breaking/
    """
    rules = parser.parse_content(robots_text)
    assert rules.can_fetch("/news/archive") is False
    assert rules.can_fetch("/news/breaking/story1") is True


def test_in_memory_caching(parser):
    """Ensure per-host cache returns stored HostRules without re-parsing."""
    host = "testnews.com"
    content = "User-agent: *\nDisallow: /blocked/"
    parser.set_cached_rules(host, content)

    assert host in parser.cache
    assert parser.can_fetch("https://testnews.com/blocked/page") is False
    assert parser.can_fetch("https://testnews.com/allowed/page") is True

    parser.clear_cache()
    assert host not in parser.cache
