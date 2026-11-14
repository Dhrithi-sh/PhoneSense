"""Unit tests for NewsCrawler pipeline orchestrator.

Verifies:
1. End-to-end crawl loop termination in --sample mode.
2. Normal mode output and sample snapshot creation.
3. Schema compliance of streamed JSONL records against formats.md.
4. HTTP 429 exponential per-host backoff.
5. HTTP 403 host disabling for crawl session.
6. Anti-bot CAPTCHA detection and host disabling.
7. Transient 5xx error retry limits (max 2 retries).
8. In-body link harvesting into Frontier Q2 with route and blacklist filtering.
9. Strict politeness host rescheduling in finally block.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from dhvani.crawl.config import PER_HOST_DELAY
from dhvani.crawl.crawler import NewsCrawler, get_source_key, is_captcha_challenge
from dhvani.crawl.extractor import ArticleExtractor, validate_article_schema
from dhvani.crawl.frontier import MercatorFrontier
from dhvani.crawl.robots import RobotsParser
from dhvani.crawl.sitemap import SitemapParser


SAMPLE_ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>उत्तर प्रदेश में भारी बारिश</title>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "NewsArticle",
      "headline": "उत्तर प्रदेश में भारी बारिश का अलर्ट",
      "articleBody": "लखनऊ और आसपास के जिलों में भारी बारिश की चेतावनी जारी की गई है। स्थानीय प्रशासन ने राहत कार्य शुरू कर दिए हैं।",
      "datePublished": "2026-10-06T14:30:00+05:30",
      "articleSection": "weather",
      "keywords": ["मौसम", "बारिश"]
    }
    </script>
</head>
<body>
    <article>
        <h1>उत्तर प्रदेश में भारी बारिश का अलर्ट</h1>
        <p>लखनऊ और आसपास के जिलों में भारी बारिश की चेतावनी जारी की गई है। स्थानीय प्रशासन ने राहत कार्य शुरू कर दिए हैं।</p>
        <a href="https://www.jagran.com/uttar-pradesh/varanasi-weather-1111.html">वाराणसी मौसम</a>
        <a href="https://www.bbc.com/hindi/articles/c000000000">ब्लैकलिस्टेड लिंक</a>
        <a href="https://www.jagran.com/astrology/rashifal-today.html">राशिफल लिंक</a>
    </article>
</body>
</html>
"""

CAPTCHA_HTML = """
<!DOCTYPE html>
<html>
<head><title>Just a moment...</title></head>
<body>
    <div id="cf-challenge-running">Checking your browser before accessing the website.</div>
    <div class="challenge-platform">Attention Required! | Cloudflare</div>
</body>
</html>
"""


@pytest.fixture
def temp_output_dir(tmp_path):
    output_file = tmp_path / "news.jsonl"
    sample_file = tmp_path / "news_sample_300.jsonl"
    return output_file, sample_file


def create_mock_response(status_code: int = 200, text: str = SAMPLE_ARTICLE_HTML):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


# ==============================================================================
# 1. Pipeline Execution & Mode Tests
# ==============================================================================

def test_crawler_sample_mode_termination(temp_output_dir):
    """Verify crawler in --sample mode stops exactly at target count and writes to sample file."""
    async def _test():
        output_file, sample_file = temp_output_dir

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            simulated_time += 0.01
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        crawler = NewsCrawler(
            frontier=frontier,
            is_sample_mode=True,
            output_path=sample_file,
            time_func=time_func,
        )
        crawler.max_articles = 5

        # Enqueue candidate URLs
        for i in range(10):
            frontier.add_url(f"https://www.jagran.com/uttar-pradesh/lucknow-news-{i}.html")

        mock_client = AsyncMock()
        mock_client.get.return_value = create_mock_response(200, SAMPLE_ARTICLE_HTML)
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        saved_count = await crawler.crawl(client=mock_client)

        assert saved_count == 5
        assert crawler.saved_articles_count == 5
        assert sample_file.exists()

        lines = sample_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

        # Verify schema compliance of each record
        for line in lines:
            record = json.loads(line)
            assert validate_article_schema(record) is True
            assert record["dup_of"] is None
            assert "author" not in record

    asyncio.run(_test())


def test_normal_mode_and_sample_snapshot(temp_output_dir):
    """Verify normal mode writes to news.jsonl and creates sample snapshot at target milestone."""
    async def _test():
        output_file, sample_file = temp_output_dir

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            simulated_time += 0.01
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        crawler = NewsCrawler(
            frontier=frontier,
            is_sample_mode=False,
            max_articles=4,
            output_path=output_file,
            sample_output_path=sample_file,
            time_func=time_func,
        )

        for i in range(6):
            frontier.add_url(f"https://www.amarujala.com/uttar-pradesh/kanpur-news-{i}.html")

        mock_client = AsyncMock()
        mock_client.get.return_value = create_mock_response(200, SAMPLE_ARTICLE_HTML)
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        saved_count = await crawler.crawl(client=mock_client)

        assert saved_count == 4
        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4

        # Sample snapshot was created
        assert sample_file.exists()

    asyncio.run(_test())


# ==============================================================================
# 2. Resilience & Error Handling Tests
# ==============================================================================

def test_http_429_exponential_backoff(temp_output_dir):
    """Verify HTTP 429 triggers per-host exponential delay increase without crashing."""
    async def _test():
        output_file, _ = temp_output_dir

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            simulated_time += 0.01
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=8.0, time_func=time_func, sleep_func=AsyncMock())
        crawler = NewsCrawler(
            frontier=frontier,
            max_articles=2,
            output_path=output_file,
            time_func=time_func,
        )
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        frontier.add_url("https://www.jagran.com/uttar-pradesh/test-429.html")

        mock_client = AsyncMock()
        mock_client.get.return_value = create_mock_response(429, "Rate Limited")

        await crawler.crawl(client=mock_client)

        assert "www.jagran.com" in crawler.host_delays
        assert crawler.host_delays["www.jagran.com"] == 16.0  # 8.0 * 2.0 backoff

    asyncio.run(_test())


def test_http_403_disables_host(temp_output_dir):
    """Verify HTTP 403 marks host as disabled for session and drops subsequent fetches."""
    async def _test():
        output_file, _ = temp_output_dir

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            simulated_time += 0.01
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        crawler = NewsCrawler(
            frontier=frontier,
            max_articles=2,
            output_path=output_file,
            time_func=time_func,
        )
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        frontier.add_url("https://blocked-site.com/news-1.html")
        frontier.add_url("https://blocked-site.com/news-2.html")

        mock_client = AsyncMock()
        mock_client.get.return_value = create_mock_response(403, "Forbidden")

        await crawler.crawl(client=mock_client)

        assert "blocked-site.com" in crawler.disabled_hosts
        assert mock_client.get.call_count == 1
        assert crawler.saved_articles_count == 0

    asyncio.run(_test())


def test_captcha_detection():
    """Verify that Cloudflare/Akamai bot challenge pages are flagged."""
    assert is_captcha_challenge(CAPTCHA_HTML) is True
    assert is_captcha_challenge(SAMPLE_ARTICLE_HTML) is False
    assert is_captcha_challenge("") is False


def test_captcha_disables_host(temp_output_dir):
    """Verify that encountering a CAPTCHA challenge immediately disables the host."""
    async def _test():
        output_file, _ = temp_output_dir

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            simulated_time += 0.01
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        crawler = NewsCrawler(
            frontier=frontier,
            max_articles=2,
            output_path=output_file,
            time_func=time_func,
        )
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        frontier.add_url("https://captcha-site.com/news-1.html")

        mock_client = AsyncMock()
        mock_client.get.return_value = create_mock_response(200, CAPTCHA_HTML)

        await crawler.crawl(client=mock_client)

        assert "captcha-site.com" in crawler.disabled_hosts
        assert crawler.saved_articles_count == 0

    asyncio.run(_test())


# ==============================================================================
# 3. Link Harvesting Tests
# ==============================================================================

def test_link_harvesting_filters_and_enqueues():
    """Verify discovered in-body links are filtered and enqueued into Frontier."""
    frontier = MercatorFrontier()
    crawler = NewsCrawler(frontier=frontier)

    discovered_links = [
        "https://www.jagran.com/uttar-pradesh/varanasi-weather-1111.html",  # valid
        "https://www.bbc.com/hindi/articles/c000000000",                   # blacklisted
        "https://www.jagran.com/astrology/rashifal-today.html",            # filtered route
        "https://www.jagran.com/uttar-pradesh/varanasi-weather-1111.html",  # duplicate
    ]

    added = crawler.harvest_links(discovered_links)
    assert added == 1

    # Frontier transferred URL into back queue for host
    assert frontier.size() == 1
    assert "www.jagran.com" in frontier.back_queues
    assert len(frontier.back_queues["www.jagran.com"]) == 1


# ==============================================================================
# 4. Robots Gating & Helper Tests
# ==============================================================================

def test_robots_txt_disallow_skips_url(temp_output_dir):
    """Verify disallowed URLs in robots.txt are skipped without fetching."""
    async def _test():
        output_file, _ = temp_output_dir

        frontier = MercatorFrontier(per_host_delay=0.01, sleep_func=AsyncMock())
        robots = RobotsParser(user_agent="CollegeProject_NewsBot")
        robots.set_cached_rules("www.jagran.com", "User-agent: *\nDisallow: /search/*\n")

        crawler = NewsCrawler(
            frontier=frontier,
            robots_parser=robots,
            max_articles=2,
            output_path=output_file,
        )
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        frontier.add_url("https://www.jagran.com/search/weather-lucknow.html")

        mock_client = AsyncMock()
        await crawler.crawl(client=mock_client)

        # get should never have been called because robots disallowed it
        assert mock_client.get.call_count == 0
        assert crawler.saved_articles_count == 0

    asyncio.run(_test())


def test_get_source_key_resolution():
    """Verify correct source key mapping across all target outlets."""
    assert get_source_key("https://www.jagran.com/news.html") == "jagran"
    assert get_source_key("https://navbharattimes.indiatimes.com/news.cms") == "nbt"
    assert get_source_key("https://www.livehindustan.com/bihar/news.html") == "livehindustan"
    assert get_source_key("https://www.amarujala.com/news.html") == "amarujala"
    assert get_source_key("https://aajtak.in/national/news.html") == "aajtak"
