"""Unit tests for SitemapParser (dhvani/crawl/sitemap.py)."""

from unittest.mock import MagicMock, patch
import pytest

from dhvani.crawl.sitemap import SitemapParser, SitemapEntry


STANDARD_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.jagran.com/news/national-1234.html?utm_source=twitter&amp;ref=share</loc>
    <lastmod>2026-10-06T10:15:30+05:30</lastmod>
    <changefreq>hourly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://www.jagran.com/rashifal/mesh-today.html</loc>
    <lastmod>2026-10-06T09:00:00+05:30</lastmod>
  </url>
  <url>
    <loc>https://www.bbc.com/hindi/news5678</loc>
    <lastmod>2026-10-06T08:00:00+05:30</lastmod>
  </url>
</urlset>
"""

GOOGLE_NEWS_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://www.amarujala.com/uttar-pradesh/lucknow/weather-alert-999.html</loc>
    <news:news>
      <news:publication>
        <news:name>Amar Ujala</news:name>
        <news:language>hi</news:language>
      </news:publication>
      <news:publication_date>2026-10-06T12:00:00+05:30</news:publication_date>
      <news:title>लखनऊ में भारी बारिश का अलर्ट</news:title>
      <news:keywords>मौसम, बारिश, लखनऊ</news:keywords>
    </news:news>
  </url>
  <url>
    <loc>https://www.amarujala.com/videos/entertainment-111.html</loc>
    <news:news>
      <news:publication>
        <news:name>Amar Ujala</news:name>
        <news:language>hi</news:language>
      </news:publication>
      <news:publication_date>2026-10-06T11:00:00+05:30</news:publication_date>
      <news:title>वीडियो रिपोर्ट</news:title>
    </news:news>
  </url>
</urlset>
"""

SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://www.jagran.com/sitemaps/news-sitemap-today.xml</loc>
    <lastmod>2026-10-06T14:00:00+05:30</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://www.jagran.com/sitemaps/sitemap-2025-10.xml</loc>
    <lastmod>2025-10-31T23:59:59+05:30</lastmod>
  </sitemap>
  <sitemap>
    <loc>https://www.jagran.com/archive/sitemap-2024.xml</loc>
    <lastmod>2024-12-31T23:59:59+05:30</lastmod>
  </sitemap>
</sitemapindex>
"""


def test_standard_sitemap_parsing_and_filtering():
    """Verify standard sitemap parsing with normalization and route filtering."""
    parser = SitemapParser(filter_urls=True, normalize=True)
    entries = parser.parse_sitemap(STANDARD_SITEMAP_XML)

    # 3 URLs in XML:
    # 1. Jagran national news (allowed, query params stripped)
    # 2. Rashifal (filtered out)
    # 3. BBC (blacklisted domain, filtered out)
    assert len(entries) == 1
    e = entries[0]
    assert e.url == "https://www.jagran.com/news/national-1234.html"
    assert "utm_source" not in e.url
    assert e.lastmod == "2026-10-06T10:15:30+05:30"
    assert e.priority == 0.8
    assert e.changefreq == "hourly"
    assert e.is_news is False


def test_google_news_sitemap_parsing():
    """Verify extraction of Google News extensions (Devanagari title, IST dates, keywords)."""
    parser = SitemapParser(filter_urls=True, normalize=True)
    entries = parser.parse_sitemap(GOOGLE_NEWS_SITEMAP_XML)

    # 2 URLs in XML:
    # 1. Weather news (allowed)
    # 2. Videos route (filtered out by route filter)
    assert len(entries) == 1
    news = entries[0]
    assert news.url == "https://www.amarujala.com/uttar-pradesh/lucknow/weather-alert-999.html"
    assert news.is_news is True
    assert news.title == "लखनऊ में भारी बारिश का अलर्ट"
    assert news.publication_date == "2026-10-06T12:00:00+05:30"
    assert news.publication_name == "Amar Ujala"
    assert news.language == "hi"
    assert news.keywords == "मौसम, बारिश, लखनऊ"


def test_sitemap_index_detection_and_parsing():
    """Verify sitemap index detection and child sitemap discovery."""
    parser = SitemapParser()

    assert parser.is_sitemap_index(SITEMAP_INDEX_XML) is True
    assert parser.is_sitemap_index(STANDARD_SITEMAP_XML) is False

    child_sitemaps = parser.parse_sitemap_index(SITEMAP_INDEX_XML)
    assert len(child_sitemaps) == 3
    assert child_sitemaps[0] == (
        "https://www.jagran.com/sitemaps/news-sitemap-today.xml",
        "2026-10-06T14:00:00+05:30",
    )


def test_archive_sitemap_discovery():
    """Verify archive seed discovery matching historical year patterns."""
    parser = SitemapParser()

    # Discover all archive sitemaps
    archives = parser.discover_archive_sitemaps(SITEMAP_INDEX_XML)
    assert len(archives) == 2
    assert "https://www.jagran.com/sitemaps/sitemap-2025-10.xml" in archives
    assert "https://www.jagran.com/archive/sitemap-2024.xml" in archives

    # Filter archives by specific year
    archives_2024 = parser.discover_archive_sitemaps(SITEMAP_INDEX_XML, year=2024)
    assert len(archives_2024) == 1
    assert archives_2024[0] == "https://www.jagran.com/archive/sitemap-2024.xml"


def test_conditional_http_fetching_mock():
    """Verify conditional HTTP requests (ETag / If-Modified-Since)."""
    parser = SitemapParser()

    # Test HTTP 304 Not Modified
    mock_resp_304 = MagicMock()
    mock_resp_304.status_code = 304

    with patch("requests.get", return_value=mock_resp_304):
        status, content, etag, lastmod = parser.fetch_sitemap(
            "https://www.jagran.com/news-sitemap.xml",
            etag='"abc123etag"',
            last_modified="Mon, 06 Oct 2026 12:00:00 GMT",
        )
        assert status == 304
        assert content is None
        assert etag == '"abc123etag"'

    # Test HTTP 200 OK
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = STANDARD_SITEMAP_XML
    mock_resp_200.headers = {
        "ETag": '"newetag456"',
        "Last-Modified": "Tue, 07 Oct 2026 02:00:00 GMT",
    }

    with patch("requests.get", return_value=mock_resp_200):
        status, content, new_etag, new_lastmod = parser.fetch_sitemap(
            "https://www.jagran.com/news-sitemap.xml"
        )
        assert status == 200
        assert content == STANDARD_SITEMAP_XML
        assert new_etag == '"newetag456"'
        assert new_lastmod == "Tue, 07 Oct 2026 02:00:00 GMT"


def test_malformed_xml_resilience():
    """Verify that malformed or truncated XML does not raise an unhandled exception."""
    parser = SitemapParser()
    malformed_xml = "<urlset><url><loc>https://www.jagran.com/news/article.html</loc><lastmod>2026-10-06"  # unclosed

    # Should safely fallback to regex extraction without throwing ET.ParseError
    entries = parser.parse_sitemap(malformed_xml)
    assert len(entries) == 1
    assert entries[0].url == "https://www.jagran.com/news/article.html"


def test_configured_seeds_discovery():
    """Verify that initial seeds are retrieved from PRIMARY_SOURCES."""
    seeds = SitemapParser.get_configured_seeds()
    assert len(seeds) > 0
    # Must include our primary sources
    assert any("jagran.com" in s for s in seeds)
    assert any("amarujala.com" in s for s in seeds)
    assert any("navbharattimes" in s for s in seeds)
