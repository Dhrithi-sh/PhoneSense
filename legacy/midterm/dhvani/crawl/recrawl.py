"""Adaptive Recrawler: Sitemap monitoring, change-rate tracking, and burst event prioritization."""

from collections import deque
import logging
import time
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from dhvani.crawl.config import (
    EWMA_ALPHA,
    MAX_POLL_INTERVAL_SECONDS,
    MIN_POLL_INTERVAL_SECONDS,
)
from dhvani.crawl.extractor import extract_section_from_url

logger = logging.getLogger("dhvani.recrawl")


class AdaptiveRecrawler:
    """Tracks change velocity across news sources and topical event bursts.

    Implements:
    1. EWMA change-rate tracking per news source:
       observed_rate = delta_n / (delta_t_seconds / 3600.0)
       lambda_s = 0.3 * observed_rate + 0.7 * lambda_prev
    2. Dynamic polling schedule (Cho & Garcia-Molina model with K = 1800s):
       tau_s = max(tau_min, min(tau_max, 1800.0 / max(lambda_s, 0.0833)))
       where tau_min = 1800s (30m) and tau_max = 21600s (6h).
    3. HTTP conditional polling cache:
       Maintains ETag and Last-Modified headers per sitemap URL.
       Sends If-None-Match and If-Modified-Since.
       On 304 Not Modified, records delta_n = 0 to naturally decay lambda_s.
    4. Burst-aware event prioritization:
       Tracks rolling publication timestamps per category across all sources.
       Acute window: 60 minutes (3600s).
       Baseline window: 6 hours (21600s).
       BurstScore = Count_1h / (MovingAvg_6h + 1.0)
       When BurstScore > 2.0, category is bursting -> routes discovered URLs to Q0.
    """

    def __init__(self, time_func: Callable[[], float] = time.time):
        self.time_func = time_func
        self.last_checked: Dict[str, float] = {}
        self.change_rates: Dict[str, float] = {}
        self.http_cache: Dict[str, Dict[str, str]] = {}
        self.category_articles: Dict[str, deque] = {}

    def update_rate(
        self,
        source_slug: str,
        delta_n: int,
        current_time: Optional[float] = None,
    ) -> float:
        """Update estimated arrival rate lambda_s (URLs/hour) using EWMA (alpha = 0.3).

        Converts delta_t from seconds to hours before computing observed_rate.
        """
        now = current_time if current_time is not None else self.time_func()
        last_time = self.last_checked.get(source_slug)
        self.last_checked[source_slug] = now

        if last_time is None or now <= last_time:
            # First poll or instantaneous check: initialize or retain rate
            if source_slug not in self.change_rates:
                self.change_rates[source_slug] = 1.0
            return self.change_rates[source_slug]

        delta_t_seconds = now - last_time
        delta_t_hours = delta_t_seconds / 3600.0
        observed_rate = delta_n / delta_t_hours

        prev_rate = self.change_rates.get(source_slug, 1.0)
        new_rate = (EWMA_ALPHA * observed_rate) + ((1.0 - EWMA_ALPHA) * prev_rate)
        self.change_rates[source_slug] = new_rate
        return new_rate

    def get_next_poll_interval(self, source_slug: str) -> float:
        """Calculate next polling interval in seconds using K = 1800.0.

        Formula:
            tau = 1800.0 / max(lambda, 0.0833)
            clamped to [MIN_POLL_INTERVAL_SECONDS, MAX_POLL_INTERVAL_SECONDS]
        """
        rate = self.change_rates.get(source_slug, 1.0)
        # Bounded Cho & Garcia-Molina formula with K = 1800s
        effective_rate = max(rate, 0.0833)
        interval = float(MIN_POLL_INTERVAL_SECONDS) / effective_rate
        return max(
            float(MIN_POLL_INTERVAL_SECONDS),
            min(float(MAX_POLL_INTERVAL_SECONDS), interval),
        )

    def should_poll(
        self,
        source_slug: str,
        current_time: Optional[float] = None,
    ) -> bool:
        """Determine if a news source is due for sitemap re-checking."""
        now = current_time if current_time is not None else self.time_func()
        if source_slug not in self.last_checked:
            return True
        interval = self.get_next_poll_interval(source_slug)
        return (now - self.last_checked[source_slug]) >= interval

    def get_conditional_headers(self, sitemap_url: str) -> Dict[str, str]:
        """Construct HTTP conditional headers (If-None-Match, If-Modified-Since) from cache."""
        headers: Dict[str, str] = {}
        cached = self.http_cache.get(sitemap_url, {})
        if "etag" in cached:
            headers["If-None-Match"] = cached["etag"]
        if "last_modified" in cached:
            headers["If-Modified-Since"] = cached["last_modified"]
        return headers

    def cache_response_headers(
        self,
        sitemap_url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Cache ETag and Last-Modified headers for conditional sitemap requests."""
        if not headers:
            return
        etag = None
        last_modified = None
        for k, v in (headers.items() if hasattr(headers, "items") else []):
            kl = str(k).lower()
            if kl == "etag":
                etag = str(v)
            elif kl == "last-modified":
                last_modified = str(v)

        cached = self.http_cache.setdefault(sitemap_url, {})
        if etag:
            cached["etag"] = etag
        if last_modified:
            cached["last_modified"] = last_modified

    def record_poll_response(
        self,
        source_slug: str,
        sitemap_url: str,
        status_code: int,
        headers: Optional[Dict[str, str]] = None,
        new_urls_count: int = 0,
        current_time: Optional[float] = None,
    ) -> float:
        """Process sitemap poll HTTP response, update cache headers, and recalculate rate.

        If status_code is 304 (Not Modified), new_urls_count is treated as 0 without error.
        """
        now = current_time if current_time is not None else self.time_func()

        if headers:
            self.cache_response_headers(sitemap_url, headers)

        delta_n = 0 if status_code == 304 else max(0, new_urls_count)
        return self.update_rate(source_slug, delta_n, current_time=now)

    def record_article(
        self,
        category: Optional[str],
        timestamp: Optional[float] = None,
    ) -> None:
        """Record an article publication in category rolling window."""
        if not category:
            return
        cat = category.strip().lower()
        if not cat or cat == "general":
            return

        now = timestamp if timestamp is not None else self.time_func()
        deq = self.category_articles.setdefault(cat, deque())
        deq.append(now)

        # Prune events older than 6 hours (21600 seconds)
        cutoff_6h = now - float(MAX_POLL_INTERVAL_SECONDS)
        while deq and deq[0] < cutoff_6h:
            deq.popleft()

    def get_burst_score(
        self,
        category: Optional[str],
        current_time: Optional[float] = None,
    ) -> float:
        """Compute the burst score for a topical category.

        Formula:
            MovingAvg_6h = Count_6h / 6.0
            BurstScore = Count_1h / (MovingAvg_6h + 1.0)
        """
        if not category:
            return 0.0
        cat = category.strip().lower()
        if cat not in self.category_articles:
            return 0.0

        now = current_time if current_time is not None else self.time_func()
        deq = self.category_articles[cat]

        # Prune events older than 6 hours
        cutoff_6h = now - float(MAX_POLL_INTERVAL_SECONDS)
        while deq and deq[0] < cutoff_6h:
            deq.popleft()

        if not deq:
            return 0.0

        count_6h = len(deq)
        cutoff_1h = now - 3600.0
        count_1h = sum(1 for t in deq if t >= cutoff_1h)

        moving_avg_6h = count_6h / 6.0
        return count_1h / (moving_avg_6h + 1.0)

    def is_bursting(
        self,
        category: Optional[str],
        current_time: Optional[float] = None,
    ) -> bool:
        """Check if category publication volume exceeds burst surge threshold (BurstScore > 2.0)."""
        if not category:
            return False
        cat = category.strip().lower()
        if not cat or cat == "general":
            return False
        return self.get_burst_score(cat, current_time=current_time) > 2.0

    def get_priority_tier(
        self,
        category: Optional[str] = None,
        url: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> str:
        """Determine front queue priority tier for a URL or category.

        Returns 'Q0' (60% weight) only if the category has an active burst (BurstScore > 2.0).
        Unknown or unverified categories default strictly to 'Q1' (25% weight).
        """
        resolved_category = category
        if (not resolved_category or resolved_category == "general") and url:
            inferred = extract_section_from_url(url)
            if inferred and inferred != "general":
                resolved_category = inferred

        if resolved_category and self.is_bursting(resolved_category, current_time=current_time):
            return "Q0"

        return "Q1"

    def get_rate(self, source_slug: str) -> float:
        """Get current estimated arrival rate lambda_s (URLs/hour)."""
        return self.change_rates.get(source_slug, 1.0)

    def get_polling_interval(self, source_slug: str) -> float:
        """Alias for get_next_poll_interval."""
        return self.get_next_poll_interval(source_slug)

    def is_category_burst(
        self,
        category: Optional[str],
        current_time: Optional[float] = None,
    ) -> bool:
        """Alias for is_bursting."""
        return self.is_bursting(category, current_time=current_time)

