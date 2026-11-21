"""Unit tests for AdaptiveRecrawler and Event Prioritization (Task 7).

Verifies:
1. Cho & Garcia-Molina EWMA change-rate tracking (alpha = 0.3) with delta_t converted to hours.
2. Polling interval formula tau = 1800 / lambda clamped strictly within [1800s, 21600s].
3. Generation of HTTP conditional headers (If-None-Match, If-Modified-Since).
4. HTTP 304 handling recording delta_n = 0 and smoothly decaying velocity without penalties.
5. BurstScore calculation over rolling 60m acute and 6h baseline windows.
6. Surge detection (> 2.0) routing verified bursting URLs into Front Queue Q0.
7. Conservative category routing: unknown or unverified categories default to Q1.
8. Burst decay reverting priority tier back to Q1 after event subsidence and window pruning.
9. Non-preemptive FIFO behavior: Q0 URLs refill back queues without preempting existing URLs.
10. End-to-end integration in NewsCrawler with conditional sitemap polling and politeness.
"""

import asyncio
from collections import deque
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from dhvani.crawl.config import (
    EWMA_ALPHA,
    MAX_POLL_INTERVAL_SECONDS,
    MIN_POLL_INTERVAL_SECONDS,
    PER_HOST_DELAY,
)
from dhvani.crawl.crawler import NewsCrawler
from dhvani.crawl.frontier import MercatorFrontier
from dhvani.crawl.recrawl import AdaptiveRecrawler


# ==============================================================================
# 1. EWMA Rate Estimation & Interval Bounds Tests
# ==============================================================================

def test_ewma_velocity_smoothing():
    """Verify arrival rate lambda is in URLs/hour and smoothed via EWMA (alpha = 0.3)."""
    current_time = 10000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: current_time)

    # Initial check: source is initialized
    recrawler.last_checked["jagran"] = current_time
    recrawler.change_rates["jagran"] = 1.0  # Nominal baseline: 1.0 URL/hour

    # Advance 1800 seconds (0.5 hours) with 5 new URLs discovered
    # delta_t = 0.5 hr, delta_n = 5 => observed_rate = 5 / 0.5 = 10.0 URLs/hour
    # Expected EWMA: 0.3 * 10.0 + 0.7 * 1.0 = 3.0 + 0.7 = 3.7 URLs/hour
    current_time += 1800.0
    new_rate = recrawler.update_rate("jagran", delta_n=5, current_time=current_time)
    assert pytest.approx(new_rate, 0.001) == 3.7
    assert pytest.approx(recrawler.change_rates["jagran"], 0.001) == 3.7
    assert recrawler.last_checked["jagran"] == current_time

    # Advance another 3600 seconds (1.0 hour) with 1 new URL discovered
    # delta_t = 1.0 hr, delta_n = 1 => observed_rate = 1 / 1.0 = 1.0 URL/hour
    # Expected EWMA: 0.3 * 1.0 + 0.7 * 3.7 = 0.3 + 2.59 = 2.89 URLs/hour
    current_time += 3600.0
    new_rate = recrawler.update_rate("jagran", delta_n=1, current_time=current_time)
    assert pytest.approx(new_rate, 0.001) == 2.89


def test_polling_interval_bounds_and_formula():
    """Verify Cho & Garcia-Molina formula tau = 1800 / lambda bounded in [1800s, 21600s]."""
    recrawler = AdaptiveRecrawler()

    # Case 1: Nominal rate (1.0 URL/hour) => tau = 1800 / 1.0 = 1800.0s (30 minutes)
    recrawler.change_rates["jagran"] = 1.0
    assert recrawler.get_next_poll_interval("jagran") == 1800.0

    # Case 2: High velocity (5.0 URLs/hour) => 1800 / 5 = 360s => clamped to tau_min = 1800.0s
    recrawler.change_rates["jagran"] = 5.0
    assert recrawler.get_next_poll_interval("jagran") == 1800.0

    # Case 3: Moderate velocity (0.5 URLs/hour) => 1800 / 0.5 = 3600.0s (1 hour)
    recrawler.change_rates["amarujala"] = 0.5
    assert recrawler.get_next_poll_interval("amarujala") == 3600.0

    # Case 4: Low velocity (0.25 URLs/hour) => 1800 / 0.25 = 7200.0s (2 hours)
    recrawler.change_rates["nbt"] = 0.25
    assert recrawler.get_next_poll_interval("nbt") == 7200.0

    # Case 5: Idle rate (< 0.0833 URLs/hour, i.e. < 1 URL per 12h) => clamped to tau_max = 21600.0s
    recrawler.change_rates["livehindustan"] = 0.05
    assert recrawler.get_next_poll_interval("livehindustan") == 21600.0

    # Case 6: Zero rate (0.0 URLs/hour) => clamped to effective 0.0833 => clamped to 21600.0s
    recrawler.change_rates["aajtak"] = 0.0
    assert recrawler.get_next_poll_interval("aajtak") == 21600.0


def test_should_poll_decision():
    """Verify should_poll adheres to calculated intervals and check timestamps."""
    simulated_time = 1000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: simulated_time)

    # Unchecked source should poll immediately
    assert recrawler.should_poll("jagran", current_time=simulated_time) is True

    # Checked source with rate 1.0 (interval = 1800s)
    recrawler.last_checked["jagran"] = simulated_time
    recrawler.change_rates["jagran"] = 1.0

    # 1000s elapsed (< 1800s) => should not poll
    assert recrawler.should_poll("jagran", current_time=simulated_time + 1000.0) is False

    # 1800s elapsed (== 1800s) => should poll
    assert recrawler.should_poll("jagran", current_time=simulated_time + 1800.0) is True

    # 2000s elapsed (> 1800s) => should poll
    assert recrawler.should_poll("jagran", current_time=simulated_time + 2000.0) is True


# ==============================================================================
# 2. HTTP Conditional Caching & 304 Handling Tests
# ==============================================================================

def test_conditional_headers_generation():
    """Verify construction of If-None-Match and If-Modified-Since from cache."""
    recrawler = AdaptiveRecrawler()
    sitemap_url = "https://www.jagran.com/news-sitemap.xml"

    # Before caching, headers dict is empty
    assert recrawler.get_conditional_headers(sitemap_url) == {}

    # Record response with ETag and Last-Modified (case-insensitive check)
    headers = {
        "ETag": '"abc12345"',
        "Last-Modified": "Wed, 07 Oct 2026 06:00:00 GMT",
    }
    recrawler.record_poll_response(
        source_slug="jagran",
        sitemap_url=sitemap_url,
        status_code=200,
        headers=headers,
        new_urls_count=10,
        current_time=1000.0,
    )

    cond_headers = recrawler.get_conditional_headers(sitemap_url)
    assert cond_headers == {
        "If-None-Match": '"abc12345"',
        "If-Modified-Since": "Wed, 07 Oct 2026 06:00:00 GMT",
    }


def test_http_304_handling():
    """Verify HTTP 304 response records delta_n = 0 and smoothly scales polling interval."""
    simulated_time = 10000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: simulated_time)

    # Initial state: 1.0 URL/hour
    recrawler.last_checked["jagran"] = simulated_time
    recrawler.change_rates["jagran"] = 1.0
    assert recrawler.get_next_poll_interval("jagran") == 1800.0

    # Advance 1800s (0.5 hour) and receive 304 Not Modified
    # delta_t = 0.5h, delta_n = 0 => observed_rate = 0.0
    # new_rate = 0.3 * 0.0 + 0.7 * 1.0 = 0.7 URLs/hour
    simulated_time += 1800.0
    new_rate = recrawler.record_poll_response(
        source_slug="jagran",
        sitemap_url="https://www.jagran.com/news-sitemap.xml",
        status_code=304,
        headers={"ETag": '"abc12345"'},
        new_urls_count=999,  # Should be ignored on 304
        current_time=simulated_time,
    )

    assert pytest.approx(new_rate, 0.001) == 0.7
    # Interval smoothly stretches: 1800 / 0.7 = 2571.43s (~42.8 minutes)
    new_interval = recrawler.get_next_poll_interval("jagran")
    assert pytest.approx(new_interval, 0.1) == 2571.4

    # Verification: last_checked is updated to the 304 check timestamp
    assert recrawler.last_checked["jagran"] == simulated_time

    # Immediately after 304 check, should_poll is False
    assert recrawler.should_poll("jagran", current_time=simulated_time) is False

    # Within the newly stretched interval (e.g. at 2500s elapsed < 2571.4s), should_poll remains False
    assert recrawler.should_poll("jagran", current_time=simulated_time + 2500.0) is False

    # Only once elapsed time from the 304 check meets/exceeds 2571.4s does should_poll become True
    assert recrawler.should_poll("jagran", current_time=simulated_time + 2572.0) is True


# ==============================================================================
# 3. Burst Score & Event Prioritization Tests
# ==============================================================================

def test_burst_score_calculation():
    """Verify BurstScore = Count_1h / (MovingAvg_6h + 1.0)."""
    now = 25000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: now)

    # Empty category
    assert recrawler.get_burst_score("weather", current_time=now) == 0.0

    # Steady state: 1 article every hour for the past 6 hours (total 6 articles)
    # 5 articles in previous hours, 1 article in current hour
    # Count_6h = 6, MovingAvg_6h = 6 / 6.0 = 1.0
    # Count_1h = 1
    # BurstScore = 1.0 / (1.0 + 1.0) = 0.5
    for h in range(5, 0, -1):
        recrawler.record_article("weather", timestamp=now - (h * 3600.0) - 50.0)
    recrawler.record_article("weather", timestamp=now - 50.0)  # 1 article in acute hour

    score = recrawler.get_burst_score("weather", current_time=now)
    assert pytest.approx(score, 0.01) == 0.5
    assert recrawler.is_bursting("weather", current_time=now) is False

    # Major volume surge: 10 new articles published within the last 30 minutes!
    for _ in range(10):
        recrawler.record_article("weather", timestamp=now - 1200.0)

    # Now:
    # Count_6h = 6 (baseline) + 10 (surge) = 16 articles
    # MovingAvg_6h = 16 / 6.0 = 2.6667
    # Count_1h = 1 (previous) + 10 (surge) = 11 articles
    # BurstScore = 11.0 / (2.6667 + 1.0) = 11.0 / 3.6667 = 3.00 > 2.0
    burst_score = recrawler.get_burst_score("weather", current_time=now)
    assert pytest.approx(burst_score, 0.05) == 3.0
    assert recrawler.is_bursting("weather", current_time=now) is True


def test_burst_detection_and_q0_routing():
    """Verify that active bursts route URLs into Q0, while normal URLs route to Q1."""
    now = 50000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: now)

    # Baseline for 'weather': 6 articles over 6h, plus 12 breaking articles in last 30m
    for h in range(5, 0, -1):
        recrawler.record_article("weather", timestamp=now - (h * 3600.0))
    for _ in range(12):
        recrawler.record_article("weather", timestamp=now - 600.0)

    assert recrawler.is_bursting("weather", current_time=now) is True
    assert recrawler.is_bursting("politics", current_time=now) is False

    # Priority tier evaluation with explicit category
    assert recrawler.get_priority_tier("weather", current_time=now) == "Q0"
    assert recrawler.get_priority_tier("politics", current_time=now) == "Q1"

    # Priority tier evaluation with URL path inference
    burst_url = "https://www.jagran.com/weather/cloudburst-warning-shimla-999.html"
    normal_url = "https://www.jagran.com/politics/parliament-session-today-888.html"
    unknown_url = "https://www.jagran.com/some-random-page-777.html"

    assert recrawler.get_priority_tier(url=burst_url, current_time=now) == "Q0"
    assert recrawler.get_priority_tier(url=normal_url, current_time=now) == "Q1"
    assert recrawler.get_priority_tier(url=unknown_url, current_time=now) == "Q1"

    # Conservative category routing: None or general defaults strictly to Q1
    assert recrawler.get_priority_tier(category=None, current_time=now) == "Q1"
    assert recrawler.get_priority_tier(category="general", current_time=now) == "Q1"


def test_burst_decay_and_normalization():
    """Verify that when the event subsides, BurstScore decays below 2.0 and reverts to Q1."""
    now = 60000.0
    recrawler = AdaptiveRecrawler(time_func=lambda: now)

    # Surge at t = 60000
    for _ in range(10):
        recrawler.record_article("weather", timestamp=now - 300.0)

    assert recrawler.is_bursting("weather", current_time=now) is True
    assert recrawler.get_priority_tier("weather", current_time=now) == "Q0"

    # Advance time by 4000s (more than 1 hour): acute 1h count becomes 0!
    future_time = now + 4000.0
    # Still within 6h window for baseline count, but 0 articles in acute 1h window
    assert recrawler.get_burst_score("weather", current_time=future_time) == 0.0
    assert recrawler.is_bursting("weather", current_time=future_time) is False
    assert recrawler.get_priority_tier("weather", current_time=future_time) == "Q1"

    # Advance time past 6 hours (22000s): all articles are pruned from memory deque
    far_future = now + 25000.0
    recrawler.get_burst_score("weather", current_time=far_future)
    assert len(recrawler.category_articles["weather"]) == 0


# ==============================================================================
# 4. Frontier Interaction & Non-Preemptive FIFO Invariant Tests
# ==============================================================================

def test_non_preemptive_fifo_behavior():
    """Verify that Q0 URLs never preempt or reorder URLs already inside a host's back queue."""
    simulated_time = 100.0
    frontier = MercatorFrontier(per_host_delay=PER_HOST_DELAY, time_func=lambda: simulated_time)

    # 1. Enqueue routine URLs for host jagran.com into Q1
    url_routine_1 = "https://www.jagran.com/state/uttar-pradesh-routine-1.html"
    url_routine_2 = "https://www.jagran.com/state/uttar-pradesh-routine-2.html"
    frontier.add_url(url_routine_1, priority_tier="Q1")
    frontier.add_url(url_routine_2, priority_tier="Q1")

    # Verify both routine URLs entered the host's FIFO back queue in exact order
    host = "www.jagran.com"
    assert host in frontier.back_queues
    assert list(frontier.back_queues[host]) == [url_routine_1, url_routine_2]

    # 2. Now a burst event occurs, generating a Q0 URL for the SAME host
    url_burst = "https://www.jagran.com/weather/flash-flood-emergency.html"
    frontier.add_url(url_burst, priority_tier="Q0")

    # Invariant: Host back queue is strictly FIFO. The Q0 URL must be appended at the TAIL.
    # It must NOT preempt or jump ahead of url_routine_1 or url_routine_2!
    assert list(frontier.back_queues[host]) == [url_routine_1, url_routine_2, url_burst]


# ==============================================================================
# 5. End-to-End NewsCrawler Integration Tests
# ==============================================================================

def test_crawler_recrawl_integration(tmp_path):
    """Verify NewsCrawler checks recrawler schedules, sends conditional headers, and handles 304."""
    async def _test():
        output_file = tmp_path / "news.jsonl"
        sample_file = tmp_path / "news_sample_300.jsonl"

        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        recrawler = AdaptiveRecrawler(time_func=time_func)

        crawler = NewsCrawler(
            frontier=frontier,
            max_articles=2,
            output_path=output_file,
            sample_output_path=sample_file,
            active_sources=["jagran"],
            recrawler=recrawler,
            time_func=time_func,
        )

        # Mock initial bootstrap
        recrawler.last_checked["jagran"] = simulated_time
        recrawler.change_rates["jagran"] = 1.0  # nominal 1.0 => interval = 1800s
        recrawler.http_cache["https://www.jagran.com/news-sitemap.xml"] = {"etag": '"etag-v1"'}

        # Enqueue 1 initial URL to allow loop to run
        frontier.add_url("https://www.jagran.com/weather/rain-1.html")

        # Set up mock HTTP client
        mock_client = AsyncMock()

        article_html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "NewsArticle",
              "headline": "भारी बारिश की चेतावनी",
              "articleBody": "उत्तर प्रदेश के कई जिलों में भारी बारिश का अलर्ट जारी किया गया है। मौसम विभाग ने लोगों से सतर्क रहने की अपील की है।",
              "datePublished": "2026-10-07T10:00:00+05:30",
              "articleSection": "weather"
            }
            </script>
        </head>
        <body><article><p>उत्तर प्रदेश के कई जिलों में भारी बारिश का अलर्ट जारी किया गया है।</p></article></body>
        </html>
        """

        # Return article HTML for article fetch
        mock_article_resp = MagicMock()
        mock_article_resp.status_code = 200
        mock_article_resp.text = article_html

        # Return 304 Not Modified for sitemap poll
        mock_sitemap_304 = MagicMock()
        mock_sitemap_304.status_code = 304
        mock_sitemap_304.headers = {"ETag": '"etag-v1"'}

        def mock_get(url, **kwargs):
            if "sitemap" in url:
                # Verify conditional header was passed
                headers = kwargs.get("headers", {})
                assert headers.get("If-None-Match") == '"etag-v1"'
                return mock_sitemap_304
            return mock_article_resp

        mock_client.get = AsyncMock(side_effect=mock_get)
        crawler.bootstrap_seeds = AsyncMock(return_value=0)

        # Advance simulated time past 1800s so recrawler triggers sitemap check
        simulated_time += 1850.0

        saved = await crawler.crawl(client=mock_client)
        assert saved == 1

        # Verify recrawler processed 304: delta_n = 0 decayed the rate
        assert recrawler.change_rates["jagran"] < 1.0
        # Polling interval increased beyond 1800s
        assert recrawler.get_next_poll_interval("jagran") > 1800.0

    asyncio.run(_test())


def test_sitemap_velocity_updates_with_newly_observed_urls_not_total_response(tmp_path):
    """Verify sitemap polling updates source velocity using newly observed URLs count (delta_n),

    NOT the raw total count of URLs returned in the sitemap XML response.
    """
    async def _test():
        simulated_time = 1000.0
        def time_func():
            nonlocal simulated_time
            return simulated_time

        frontier = MercatorFrontier(per_host_delay=0.01, time_func=time_func, sleep_func=AsyncMock())
        recrawler = AdaptiveRecrawler(time_func=time_func)

        crawler = NewsCrawler(
            frontier=frontier,
            output_path=tmp_path / "news.jsonl",
            sample_output_path=tmp_path / "news_sample_300.jsonl",
            active_sources=["jagran"],
            recrawler=recrawler,
            time_func=time_func,
        )

        # Baseline: jagran checked at t=1000 with nominal rate 1.0 URL/hour
        recrawler.last_checked["jagran"] = simulated_time
        recrawler.change_rates["jagran"] = 1.0

        # Simulate that 95 URLs from this outlet are ALREADY seen in previous crawl cycles
        old_urls = [f"https://www.jagran.com/state/old-news-{i}.html" for i in range(1, 96)]
        for u in old_urls:
            frontier.seen_urls.add(u)

        # 5 genuinely NEW URLs published in this interval
        new_urls = [f"https://www.jagran.com/weather/new-rain-{i}.html" for i in range(1, 6)]

        # Sitemap XML contains ALL 100 URLs (95 old + 5 new)
        all_sitemap_urls = old_urls + new_urls
        sitemap_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        for u in all_sitemap_urls:
            sitemap_xml += f"  <url><loc>{u}</loc></url>\n"
        sitemap_xml += "</urlset>"

        mock_sitemap_resp = MagicMock()
        mock_sitemap_resp.status_code = 200
        mock_sitemap_resp.text = sitemap_xml
        mock_sitemap_resp.headers = {"ETag": '"v2"'}

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_sitemap_resp

        # Advance simulated time by 1800s (0.5 hours)
        simulated_time += 1800.0

        # Poll sitemaps for jagran
        enqueued_count = await crawler.poll_source_sitemaps(mock_client, "jagran")

        # 1. Exactly 5 new URLs should have been enqueued (the 95 seen URLs are skipped)
        assert enqueued_count == 5

        # 2. Check source velocity update:
        # delta_t = 1800s = 0.5 hours.
        # If raw response count (100) were used: observed_rate = 100 / 0.5 = 200.0 => lambda = 0.3*200 + 0.7*1 = 60.7 URLs/hour!
        # When newly observed count (5) is used: observed_rate = 5 / 0.5 = 10.0 => lambda = 0.3*10 + 0.7*1 = 3.7 URLs/hour!
        assert recrawler.change_rates["jagran"] == pytest.approx(3.7, 0.01)
        assert recrawler.change_rates["jagran"] != pytest.approx(60.7, 0.01)

        # 3. last_checked is updated to the current poll timestamp
        assert recrawler.last_checked["jagran"] == simulated_time

    asyncio.run(_test())
