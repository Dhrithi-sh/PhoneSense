"""URL and Content Route Filtering."""

from typing import List
from dhvani.crawl.config import BLACKLISTED_DOMAINS, URL_EXCLUDE_PATTERNS


def should_skip_url(url: str, custom_exclude_patterns: List[str] = None) -> bool:
    """Determine if a URL should be skipped (blacklist, media, horoscope/astrology, etc.)."""
    patterns = custom_exclude_patterns or URL_EXCLUDE_PATTERNS
    url_lower = url.lower()

    # Check blacklisted domains
    for domain in BLACKLISTED_DOMAINS:
        if domain in url_lower:
            return True

    # Check non-article pattern filters
    for pattern in patterns:
        if pattern in url_lower:
            return True

    return False
