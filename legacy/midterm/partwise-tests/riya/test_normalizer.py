"""Tests for URL Normalization and Route Filtering.

Verifies:
1. URL scheme and host lowercasing.
2. Stripping of tracking parameters (utm_*, fbclid, ref, amp_js_v, amp).
3. Preservation of legitimate query parameters.
4. AMP-to-canonical conversion.
5. Trailing slash and port normalization.
6. Domain blacklist filtering (BBC Hindi, News18, NDTV).
7. Non-article route filtering (rashifal, astrology, photos, videos, live blogs).
"""

import pytest
from dhvani.crawl.normalizer import normalize_url
from dhvani.crawl.filters import should_skip_url


# ==============================================================================
# 1. URL Normalization Tests
# ==============================================================================

def test_scheme_and_host_lowercasing():
    """Verify that scheme and hostname are converted to lowercase while path casing is preserved."""
    raw = "HTTPS://WWW.JAGRAN.COM/Uttar-Pradesh/Lucknow-News.html"
    expected = "https://www.jagran.com/Uttar-Pradesh/Lucknow-News.html"
    assert normalize_url(raw) == expected


def test_strip_tracking_parameters():
    """Verify stripping of marketing and analytics tracking parameters."""
    raw = "https://www.amarujala.com/delhi/news-123?utm_source=twitter&utm_medium=social&utm_campaign=breaking&fbclid=IwAR123&ref=hp_top"
    expected = "https://www.amarujala.com/delhi/news-123"
    assert normalize_url(raw) == expected


def test_preserve_substantive_query_parameters():
    """Verify that non-tracking parameters are retained."""
    raw = "https://www.livehindustan.com/search?q=weather&page=2&utm_source=feed"
    result = normalize_url(raw)
    assert "utm_source" not in result
    assert "q=weather" in result
    assert "page=2" in result


def test_amp_to_canonical_mapping():
    """Verify conversion of AMP paths and query flags to canonical equivalents."""
    # Prefix /amp/
    raw_prefix = "https://navbharattimes.indiatimes.com/amp/state/uttar-pradesh/news.cms"
    expected_prefix = "https://navbharattimes.indiatimes.com/state/uttar-pradesh/news.cms"
    assert normalize_url(raw_prefix) == expected_prefix

    # Suffix /amp
    raw_suffix = "https://aajtak.in/national/story-123/amp"
    expected_suffix = "https://aajtak.in/national/story-123"
    assert normalize_url(raw_suffix) == expected_suffix

    # Query amp=1
    raw_query = "https://www.amarujala.com/lucknow/article?amp=1&utm_medium=rss"
    expected_query = "https://www.amarujala.com/lucknow/article"
    assert normalize_url(raw_query) == expected_query


def test_trailing_slash_and_empty_inputs():
    """Verify trailing slash removal and handling of empty inputs."""
    assert normalize_url("https://www.jagran.com/national/") == "https://www.jagran.com/national"
    assert normalize_url("https://www.jagran.com/") == "https://www.jagran.com/"
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


def test_default_port_stripping():
    """Verify that default ports (80 for http, 443 for https) are stripped while custom ports are preserved."""
    assert normalize_url("http://www.jagran.com:80/national") == "http://www.jagran.com/national"
    assert normalize_url("https://www.jagran.com:443/national") == "https://www.jagran.com/national"
    assert normalize_url("https://www.jagran.com:8443/national") == "https://www.jagran.com:8443/national"


# ==============================================================================
# 2. Route and Domain Filtering Tests
# ==============================================================================

def test_blacklisted_domains():
    """Verify that blacklisted domains are rejected."""
    assert should_skip_url("https://www.bbc.com/hindi/articles/c000000000") is True
    assert should_skip_url("https://hindi.news18.com/news/nation/story.html") is True
    assert should_skip_url("https://ndtv.in/india-news/story-123") is True


def test_non_article_route_exclusions():
    """Verify that astrology, multimedia, and live blog routes are skipped."""
    assert should_skip_url("https://www.jagran.com/astrology/rashifal-today.html") is True
    assert should_skip_url("https://www.amarujala.com/photo-gallery/entertainment/photos-123") is True
    assert should_skip_url("https://aajtak.in/visualstories/web-stories/photos-456") is True
    assert should_skip_url("https://www.livehindustan.com/videos/news-video-789") is True
    assert should_skip_url("https://navbharattimes.indiatimes.com/live-updates/budget-2026") is True


def test_legitimate_news_urls_allowed():
    """Verify that valid regional news URLs pass the filter."""
    valid_urls = [
        "https://www.jagran.com/uttar-pradesh/lucknow-weather-alert-23456789.html",
        "https://navbharattimes.indiatimes.com/state/uttar-pradesh/lucknow/traffic-police-advisory/articleshow/98765432.cms",
        "https://www.livehindustan.com/bihar/patna/story-ganga-water-level-rise-54321.html",
        "https://www.amarujala.com/uttar-pradesh/kanpur/kanpur-metro-phase-2-work-accelerates-654321.html",
        "https://aajtak.in/india/news/story/delhi-pollution-aqi-update-123456-2026-10-06",
    ]
    for url in valid_urls:
        assert should_skip_url(url) is False, f"Expected valid URL not to be skipped: {url}"
