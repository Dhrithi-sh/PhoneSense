"""Configuration settings for the Dhvani Regional News Crawler (P1)."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTICLES_FILE = DATA_DIR / "news.jsonl"
SAMPLE_ARTICLES_FILE = DATA_DIR / "news_sample_300.jsonl"
SITEMAP_DIR = DATA_DIR / "sitemaps"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SITEMAP_DIR.mkdir(parents=True, exist_ok=True)

# Ethical Crawling & Identification
USER_AGENT = "CollegeProject_NewsBot(+https://github.com/InformationRetrieval-Mid/Dhvani-lingual; contact: ra810@snu.edu.in)"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "hi,en-US;q=0.7,en;q=0.3",
}

# Politeness Constraints
PER_HOST_DELAY = 8.0  # seconds between consecutive requests to the exact same host
DEFAULT_TIMEOUT = 12.0  # seconds
MAX_RETRIES = 2
BACKOFF_FACTOR = 2.0

# Target News Outlets (Primary)
PRIMARY_SOURCES = {
    "jagran": {
        "name": "Dainik Jagran",
        "domain": "jagran.com",
        "home_url": "https://www.jagran.com",
        "robots_url": "https://www.jagran.com/robots.txt",
        "sitemaps": [
            "https://www.jagran.com/news-sitemap.xml",
            "https://www.jagran.com/sitemap.xml",
        ],
    },
    "nbt": {
        "name": "Navbharat Times",
        "domain": "navbharattimes.indiatimes.com",
        "home_url": "https://navbharattimes.indiatimes.com",
        "robots_url": "https://navbharattimes.indiatimes.com/robots.txt",
        "sitemaps": [
            "https://navbharattimes.indiatimes.com/staticsitemap/nbt/news/sitemap-48hours.xml",
            "https://navbharattimes.indiatimes.com/sitemapxml.cms",
        ],
    },
    "livehindustan": {
        "name": "Live Hindustan",
        "domain": "livehindustan.com",
        "home_url": "https://www.livehindustan.com",
        "robots_url": "https://www.livehindustan.com/robots.txt",
        "sitemaps": [
            "https://www.livehindustan.com/news-sitemap.xml",
            "https://www.livehindustan.com/sitemap.xml",
        ],
    },
    "amarujala": {
        "name": "Amar Ujala",
        "domain": "amarujala.com",
        "home_url": "https://www.amarujala.com",
        "robots_url": "https://www.amarujala.com/robots.txt",
        "sitemaps": [
            "https://www.amarujala.com/sitemap-news-v1.xml",
            "https://www.amarujala.com/sitemap.xml",
        ],
    },
    "aajtak": {
        "name": "Aaj Tak",
        "domain": "aajtak.in",
        "home_url": "https://www.aajtak.in",
        "robots_url": "https://www.aajtak.in/robots.txt",
        "sitemaps": [
            "https://www.aajtak.in/rssfeeds/news-sitemap.xml",
            "https://www.aajtak.in/rssfeeds/sitemap.xml",
        ],
    },
}

# Backup Sources
BACKUP_SOURCES = {
    "jansatta": {
        "name": "Jansatta",
        "domain": "jansatta.com",
        "home_url": "https://www.jansatta.com",
        "robots_url": "https://www.jansatta.com/robots.txt",
        "sitemaps": ["https://www.jansatta.com/news-sitemap.xml"],
    },
    "bhaskar": {
        "name": "Dainik Bhaskar",
        "domain": "bhaskar.com",
        "home_url": "https://www.bhaskar.com",
        "robots_url": "https://www.bhaskar.com/robots.txt",
        "sitemaps": ["https://www.bhaskar.com/sitemap.xml"],
    },
}

# Blacklisted Domains (Do Not Crawl)
BLACKLISTED_DOMAINS = [
    "bbc.com",
    "www.bbc.com",
    "news18.com",
    "hindi.news18.com",
    "ndtv.com",
    "ndtv.in",
]

# Route Filters: Drop non-article, media, and horoscope/astrology URLs
URL_EXCLUDE_PATTERNS = [
    "/rashifal",
    "/astrology",
    "/horoscope",
    "/dharm",
    "/photo-gallery",
    "/photos",
    "/videos",
    "/video",
    "/web-stories",
    "/visual-stories",
    "/live-updates",
    "/cricket-scorecard",
    "/games",
    "/entertainment/web-series",
]

# Mercator Frontier Priority Queue Settings
FRONT_QUEUE_PROBABILITIES = {
    "Q0": 0.60,  # Bursts & breaking news
    "Q1": 0.25,  # Fresh live sitemaps
    "Q2": 0.10,  # In-article hyperlinks
    "Q3": 0.05,  # Archive backfill
}

# Adaptive Recrawling Constants
MIN_POLL_INTERVAL_SECONDS = 1800   # 30 minutes
MAX_POLL_INTERVAL_SECONDS = 21600  # 6 hours
EWMA_ALPHA = 0.3

# Near-Duplicate Detection Settings
SHINGLE_SIZE = 4                   # 4-word shingles
JACCARD_THRESHOLD = 0.70           # Duplicate cutoff
TEMPORAL_WINDOW_HOURS = 24         # Wire story comparison window
MINHASH_NUM_PERM = 64
MINHASH_BANDS = 16
MINHASH_ROWS = 4

# Wire agency signatures
WIRE_AGENCY_KEYWORDS = [
    "पीटीआई",
    "भाषा",
    "वार्ता",
    "ANI",
    "PTI",
    "Univarta",
    "Bhasha",
]
