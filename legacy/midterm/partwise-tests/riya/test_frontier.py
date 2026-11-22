"""Unit tests for MercatorFrontier (dhvani/crawl/frontier.py)."""

import asyncio
from typing import List
import pytest

from dhvani.crawl.frontier import MercatorFrontier


class MockClock:
    """Deterministic simulated clock for testing frontier timing without real sleep."""

    def __init__(self, initial_time: float = 1000.0):
        self.current_time = initial_time
        self.sleep_calls: List[float] = []

    def now(self) -> float:
        return self.current_time

    async def sleep(self, seconds: float):
        self.sleep_calls.append(seconds)
        self.current_time += seconds


def test_url_deduplication_and_normalization():
    """Verify that seen URLs and duplicate variants are rejected."""
    frontier = MercatorFrontier(per_host_delay=8.0)

    url1 = "https://www.jagran.com/news/article1.html"
    url1_with_utm = "https://www.jagran.com/news/article1.html?utm_source=twitter&ref=share"

    assert frontier.add_url(url1, priority_tier="Q1") is True
    assert frontier.num_seen() == 1

    # Adding exact same URL again should return False
    assert frontier.add_url(url1, priority_tier="Q1") is False
    assert frontier.num_seen() == 1

    # Adding same URL with tracking params should normalize to same URL and return False
    assert frontier.add_url(url1_with_utm, priority_tier="Q1") is False
    assert frontier.num_seen() == 1


def test_blacklisted_and_filtered_routes():
    """Verify that blacklisted domains and non-article routes are rejected."""
    frontier = MercatorFrontier(per_host_delay=8.0)

    # Blacklisted domains (BBC, News18, NDTV)
    assert frontier.add_url("https://www.bbc.com/hindi/news123") is False
    assert frontier.add_url("https://hindi.news18.com/state/uttar-pradesh") is False
    assert frontier.add_url("https://ndtv.in/india/breaking") is False

    # Route filters (astrology, photo-gallery, videos)
    assert frontier.add_url("https://www.jagran.com/rashifal/mesh-rashi-today") is False
    assert frontier.add_url("https://www.amarujala.com/photo-gallery/entertainment") is False
    assert frontier.add_url("https://www.amarujala.com/videos/news") is False

    assert frontier.size() == 0
    assert frontier.num_seen() == 0


def test_biased_front_queue_selection_distribution():
    """Verify that front queue selection respects relative probability weights."""
    frontier = MercatorFrontier(per_host_delay=8.0)

    # Populate Q0 (Bursts, weight 0.60) and Q1 (Fresh, weight 0.25)
    for i in range(100):
        frontier.front_queues["Q0"].append(f"https://www.jagran.com/burst_{i}")
        frontier.front_queues["Q1"].append(f"https://www.jagran.com/fresh_{i}")

    # Theoretical relative probability for Q0: 0.60 / (0.60 + 0.25) ~ 70.6%
    q0_selections = sum(1 for _ in range(1000) if frontier._select_front_queue() == "Q0")
    q0_ratio = q0_selections / 1000.0

    # With 1000 trials, should be well within [0.63, 0.78]
    assert 0.63 <= q0_ratio <= 0.78, f"Unexpected Q0 selection ratio: {q0_ratio}"


def test_per_host_politeness_delay_enforcement():
    """Verify that consecutive requests to the exact same host strictly wait 8 seconds."""
    async def _test():
        clock = MockClock(initial_time=100.0)
        delay = 8.0

        frontier = MercatorFrontier(
            per_host_delay=delay,
            time_func=clock.now,
            sleep_func=clock.sleep,
        )

        host = "www.jagran.com"
        url1 = f"https://{host}/article1"
        url2 = f"https://{host}/article2"

        frontier.add_url(url1, priority_tier="Q1")
        frontier.add_url(url2, priority_tier="Q1")

        # First fetch: host has never been fetched, should be available immediately
        item1 = await frontier.get_next_url()
        assert item1 == (url1, host)
        assert len(clock.sleep_calls) == 0  # No sleep needed for first request

        # Simulate fetch taking 0.5s and completing at t = 100.5
        clock.current_time += 0.5
        frontier.complete_request(host, completion_time=clock.now())

        # Second fetch: next request for www.jagran.com is not allowed until t = 100.5 + 8.0 = 108.5
        item2 = await frontier.get_next_url()
        assert item2 == (url2, host)

        # Must have slept for exactly the remaining 8.0 seconds
        assert len(clock.sleep_calls) == 1
        assert clock.sleep_calls[0] == pytest.approx(8.0)
        assert clock.now() == pytest.approx(108.5)

    asyncio.run(_test())


def test_multi_host_interleaving_concurrency():
    """Verify that requests across different hosts interleave without blocking each other."""
    async def _test():
        clock = MockClock(initial_time=200.0)
        frontier = MercatorFrontier(
            per_host_delay=8.0,
            time_func=clock.now,
            sleep_func=clock.sleep,
        )

        h1 = "www.jagran.com"
        h2 = "www.amarujala.com"
        h3 = "www.livehindustan.com"

        frontier.add_url(f"https://{h1}/article1")
        frontier.add_url(f"https://{h2}/article1")
        frontier.add_url(f"https://{h3}/article1")

        # Fetch h1
        res1 = await frontier.get_next_url()
        assert res1[1] == h1
        # Do not complete h1 yet (h1 is in-flight).

        # Fetch h2 immediately: must not wait on h1's 8-second delay
        res2 = await frontier.get_next_url()
        assert res2[1] == h2

        # Fetch h3 immediately
        res3 = await frontier.get_next_url()
        assert res3[1] == h3

        # All three distinct hosts fetched at t=200 without any sleep calls
        assert len(clock.sleep_calls) == 0

    asyncio.run(_test())


def test_frontier_drain_and_is_empty():
    """Verify frontier empty state detection and graceful completion."""
    async def _test():
        clock = MockClock(initial_time=300.0)
        frontier = MercatorFrontier(
            per_host_delay=8.0,
            time_func=clock.now,
            sleep_func=clock.sleep,
        )

        url = "https://www.aajtak.in/article1"
        host = "www.aajtak.in"

        frontier.add_url(url)
        assert frontier.is_empty() is False

        item = await frontier.get_next_url()
        assert item == (url, host)
        assert frontier.is_empty() is False  # host is in-flight

        frontier.complete_request(host, completion_time=clock.now())
        # No more URLs for aajtak.in
        assert frontier.is_empty() is True

        # Calling get_next_url on empty frontier returns None
        assert await frontier.get_next_url() is None

    asyncio.run(_test())


def test_batch_add_urls():
    """Verify batch add_urls helper."""
    frontier = MercatorFrontier(per_host_delay=8.0)
    urls = [
        "https://www.jagran.com/news1",
        "https://www.jagran.com/news2",
        "https://www.jagran.com/news1",  # duplicate
        "https://www.bbc.com/news3",     # blacklisted
    ]
    added = frontier.add_urls(urls, priority_tier="Q1")
    assert added == 2
    assert frontier.num_seen() == 2
