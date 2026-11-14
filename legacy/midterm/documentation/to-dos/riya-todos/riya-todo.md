# P1: Regional News Crawler & Corpus Pipeline - Checklist

## Task 1: Setup, Compliance & Filtering
### 1. Project Setup & Configuration
- [x] Create `.gitignore` to keep `data/*.jsonl` and local files off git
- [x] Initialize package layout (`dhvani/`, `dhvani/crawl/`, `dhvani/eval/`, `partwise-tests/riya/`)
- [x] Configure ethical User-Agent with project URL and contact email
- [x] Configure per-host politeness delay (8.0s) and request timeouts
- [x] Define whitelist of 5 primary news sources and 2 backup sources
- [x] Define domain blacklist (BBC Hindi, News18, NDTV)
- [x] Define route exclusion patterns (astrology, photo galleries, videos, live blogs)
- [x] Set Mercator front queue sampling distribution (Q0-Q3)
- [x] Set deduplication thresholds (4-word shingles, Jaccard 0.70, 24h window)

### 2. URL Normalization & Route Filters
- [x] Implement URL scheme and domain lowercasing
- [x] Implement tracking query parameter stripping (`utm_*`, `fbclid`, `ref`, etc.)
- [x] Implement AMP-to-canonical desktop URL mapping
- [x] Implement trailing slash and default port normalization
- [x] Implement blacklist domain filtering
- [x] Implement non-article route filtering

### 3. Robots.txt Compliance (`robots.py`)
- [x] Implement RFC 9309 rules parser
- [x] Implement path pattern matching with `*` and `$` wildcards
- [x] Implement longest-match precedence rule for `Allow` vs `Disallow`
- [x] Implement user-agent group matching (`CollegeProject_NewsBot` with fallback to `*`)
- [x] Implement in-memory per-host caching for parsed rules
- [x] Implement sitemap directive extraction from `robots.txt`
- [x] Write unit tests verifying wildcard support, caching, precedence, and user-agent selection

## Task 2: Mercator Frontier & Scheduler
### 4. Mercator Frontier (`frontier.py`)
- [x] Implement 4 priority front queues (Q0: Bursts, Q1: Fresh, Q2: Links, Q3: Archive)
- [x] Implement biased random selector for front queue sampling
- [x] Implement per-host FIFO back queues for politeness isolation
- [x] Implement min-heap tracking `(next_allowed_time, host)` to enforce 8.0s delay
- [x] Implement seen-URL deduplication set
- [x] Write unit tests verifying 8.0s delay enforcement and queue priorities

## Task 3: Feed & Seed Management
### 5. Sitemap & Seed Management (`sitemap.py`)
- [x] Implement XML sitemap parser for standard sitemaps
- [x] Implement parser for Google News sitemaps (`<news:news>`, `<news:publication_date>`)
- [x] Implement sitemap index parser for nested feeds
- [x] Add archive sitemap seed discovery

## Task 4: Content & Metadata Extraction
### 6. Article & Metadata Extraction (`extractor.py`)
- [x] Implement Schema.org JSON-LD extraction (`NewsArticle`, `BlogPosting`)
- [x] Implement HTML fallback extraction (`<h1>`, `<article>`, `<p>`)
- [x] Implement date parsing to strict ISO-8601 with IST offset (`+05:30`)
- [x] Implement state and city regex extraction from URL paths
- [x] Implement section slug extraction and normalization
- [x] Implement in-body hyperlink extraction for PageRank (`links` field)
- [x] Enforce zero author names stored in extracted records
- [x] Validate extracted output against `documentation/formats.md`

## Task 5: Pipeline Execution & Sample Crawl
### 7. Crawler Pipeline Execution (`crawler.py`)
- [x] Integrate sitemap fetcher, frontier, downloader, extractor, and dedup pipeline
- [x] Add polite backoff on HTTP 403, 429, or network errors
- [x] Implement streaming JSONL writer to `data/news.jsonl`
- [x] Crawl and generate 300-article sample for team handoff (`data/news_sample_300.jsonl` - H3)
- [x] Complete full crawl to reach 5,000-12,000 articles (Corpus Freeze - H18: 5,001 articles crawled)

## Task 6: Deduplication & Story Clustering
### 8. Deduplication & Story Lineage (`dedup.py`)
- [x] Implement MD5 body text hashing for exact duplicates (`content_hash`)
- [x] Implement 4-word sliding shingles generator over Hindi text
- [x] Implement Jaccard similarity calculation
- [x] Implement wire agency keyword detector (`agency_flag`)
- [x] Implement candidate pair selection within +/- 24-hour temporal window
- [x] Implement canonical cluster head assignment (`dup_of: null` for earliest, else original `doc_id`)
- [x] Benchmark MinHash + LSH candidate bucketing (situational fallback)

## Task 7: Adaptive Recrawling & Event Prioritization
### 9. Adaptive Recrawling & Event Prioritization (`recrawl.py`)
- [x] Implement sitemap check timestamp tracking per source
- [x] Implement active period vs idle period check interval scaling (30m to 6h)
- [x] Add HTTP conditional headers (`If-Modified-Since`, `If-None-Match`/ETag)
- [x] Implement rolling 60-minute section volume tracking
- [x] Implement burst surge detection to route breaking URLs into Front Queue Q0

## Task 8: Shared Evaluation Tooling
### 10. Shared Tooling & Downstream Support (`eval/pool.py`, handoffs)
- [x] Implement TREC run pooling function (`pool_runs`)
- [x] Complete CLI for run file pooling and judgment template generation (`formats.md` #5)
- [ ] Document data ingestion and pre-work in `documentation/handoffs/riya.md`

## Verification & Evaluation Deliverables
### 11. Test Suite (`partwise-tests/riya/`)
- [x] `test_robots.py`: Benchmark custom parser vs `urllib.robotparser` bugs
- [x] `test_frontier.py`: Validate 8.0s per-host delay and front queue weighting
- [x] `test_sitemap.py`: Validate standard, Google News, index, and archive sitemaps
- [x] `test_normalizer.py`: Validate query stripping and AMP conversion
- [x] `test_extractor.py`: Validate JSON-LD extraction, IST dates, and absence of authors
- [x] `test_crawler.py`: Validate crawl loops, 429 backoff, 403/CAPTCHA disabling, and link harvesting
- [x] `test_dedup.py`: Measure precision and recall on 100 labeled article pairs
- [x] `test_recrawl.py`: Validate EWMA freshness tracking, HTTP 304 conditional polling, and burst routing
- [x] `test_format_compliance.py`: Validate schema compliance with `formats.md`
- [x] `test_pool.py`: Validate need_id aggregation, cross-form deduplication, and --raw-qid override

### 12. Evaluation & Submission Deliverables
- [x] Generate deduplication threshold precision/recall table
- [x] Generate crawler corpus statistics (articles per source, section distribution)
- [ ] Write report section (crawling architecture, robots.txt findings, deduplication)
- [ ] Record video segment (crawler log, robots tests, duplicate story grouping)
