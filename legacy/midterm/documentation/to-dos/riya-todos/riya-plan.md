# Crawler & Corpus Pipeline 
**Component:** P1 - Regional News Crawler, Corpus & Shared Tooling  
**Deliverable:** `data/news.jsonl` (following `documentation/formats.md`)  
**Shared Tool:** `dhvani/eval/pool.py` (TREC run pooling)

---

## 1. Scope & Crawling Rules

### Target Sites
* **5 Primary Sites:**
  1. *Dainik Jagran* (`jagran.com`)
  2. *Navbharat Times* (`navbharattimes.indiatimes.com`)
  3. *Live Hindustan* (`livehindustan.com`)
  4. *Amar Ujala* (`amarujala.com`)
  5. *Aaj Tak* (`aajtak.in`)
* **Backups:** *Jansatta* (`jansatta.com`), *Dainik Bhaskar* (`bhaskar.com`).
* **Corpus Target:** ~5,000 articles from live sitemaps (up to ~12,000 using archive sitemaps).

### Ethical Crawling Guidelines
* **User-Agent:** Use an honest, descriptive name: `CollegeProject_NewsBot(+https://github.com/InformationRetrieval-Mid/Dhvani-lingual; contact: ra810@snu.edu.in)`.
* **Politeness:** Minimum **8.0 seconds delay** between consecutive requests to the same site.
* **Privacy:** **Store no author names** anywhere in the dataset.
* **Storage Policy:** Article text stays local on your machine (`data/news.jsonl` is gitignored). Only crawler code and URLs are committed to GitHub.
* **Error Handling:** Stop/back off on HTTP 403, 429, or CAPTCHA encounters.

---

## 2. Directory Structure
```text
dhvani/
├── crawl/
│   ├── __init__.py
│   ├── config.py             # Sites whitelist, timeouts, 8s rate limit, user-agent
│   ├── robots.py             # Custom RFC 9309 robots.txt parser
│   ├── frontier.py           # Mercator frontier (front priority queues + politeness min-heap)
│   ├── sitemap.py            # Sitemap fetcher and parser (news-sitemap.xml + archives)
│   ├── normalizer.py         # URL normalization & AMP-to-canonical mapping
│   ├── filters.py            # Route filters (skips rashifal/astrology, photos, videos)
│   ├── extractor.py          # JSON-LD article extractor with HTML <p> fallback
│   ├── dedup.py              # Shingles, Jaccard, MinHash + LSH, story clustering
│   ├── recrawl.py            # Adaptive freshness & change-rate tracking
│   └── crawler.py            # Main crawler loop & CLI runner
│
└── eval/
    ├── __init__.py
    └── pool.py               # Shared pooling script for TREC run files
```

---

## 3. Implementation Design by Task (Dependency Order)

### Task 1: Setup, Compliance & Filtering (`config.py`, `robots.py`, `normalizer.py`, `filters.py`)
#### 1. Configuration & Source Rules (`config.py`)
* **Whitelisted Sources:** 5 primary domains (*Dainik Jagran*, *Navbharat Times*, *Live Hindustan*, *Amar Ujala*, *Aaj Tak*) and 2 backup domains (*Jansatta*, *Dainik Bhaskar*).
* **Blacklisted Domains:** *BBC Hindi* (robots.txt scraping prohibition), *News18* and *NDTV* (Cloudflare bot blocks).
* **Rate Limits & Politeness:** Global 8.0s per-host politeness delay, 15.0s network timeouts, honest `User-Agent`.

#### 2. URL Normalization & Route Filtering (`normalizer.py`, `filters.py`)
* **Normalization Logic:**
  - Lowercase URL scheme and host.
  - Strip tracking query parameters: `utm_*`, `ref`, `fbclid`, `amp_js_v`.
  - Map AMP pages to canonical desktop URLs (strip `/amp/`, `?amp=1`).
  - Strip default ports and trailing slashes.
* **Route Exclusions:**
  - Astrology / Horoscope: `/rashifal/`, `/astrology/`, `/horoscope/` (avoids keyword pollution).
  - Media: `/photo-gallery/`, `/photos/`, `/videos/`.
  - Live blogs without full text: `/live-updates/`.

#### 3. Robots.txt Compliance (`robots.py`)
* **Specification & Scope:** Enforces RFC 9309 compliance for target news domains prior to dispatching fetch requests. Handles wildcard and end-of-path patterns used across Indian news sites, maintains per-host rule caching, and automates sitemap discovery.
* **Rules & Matching Logic:**
  - **Wildcard & Path Matching:** Converts wildcard (`*`) and path-end (`$`) rules into compiled regular expressions adhering to RFC 9309 §2.2.2.
  - **Precedence:** Enforces longest-match precedence between conflicting `Allow` and `Disallow` rules; equal-length matches resolve in favor of `Allow`.
  - **User-Agent Resolution:** Matches site-specific rules for `CollegeProject_NewsBot` first; falls back to default wildcard (`*`) group if no explicit record exists. Does not alter user-agent to bypass disallow directives.
  - **In-Memory Caching:** Caches parsed rules per host (`scheme://host`) in memory to eliminate redundant HTTP requests across crawl sessions.
  - **Sitemap Extraction:** Extracts `Sitemap:` directives from `robots.txt` and forwards discovered feeds directly into the sitemap discovery queue.

---

### Task 2: Mercator Frontier & Scheduler (`frontier.py`)
#### How It Works & Concurrency Model
Separates priority from politeness using a two-tier queue structure driven by a **single-threaded asynchronous event loop (`asyncio` + `heapq`)**:
* While one request is waiting on network I/O, the event loop can dispatch requests for another eligible host.
* Eliminates thread locks, race conditions, and concurrent file-write problems while overlapping I/O.
* Theoretical scheduling interval across 5 target hosts: $8 / 5 = 1.6\text{ seconds/request}$.

```text
               [ Incoming URLs: Sitemaps, Bursts, In-Page Links ]
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                 FRONT QUEUES (Priority)                │
       │   Q0: Bursts (60%)           Q1: Fresh Sitemaps (25%)  │
       │   Q2: In-Page Links (10%)    Q3: Archive Sitemaps (5%) │
       └────────────────────────────┬───────────────────────────┘
                                    │ Biased Random Selector
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                BACK QUEUES (Politeness)                │
       │   [Jagran]   [Navbharat]   [Hindustan]   [AmarUjala]   │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
               ┌────────────────────────────────────────┐
               │         POLITENESS MIN-HEAP            │
               │  Entries: (next_allowed_time, host)    │
               │  Enforces 8-second delay per host      │
               └────────────────────────────────────────┘
```

#### Asynchronous Queue Loop
1. Min-Heap inspects top entry `(ready_time, entry_id, host)`.
2. If `ready_time > now()`, asynchronously sleeps until `ready_time`.
3. Pop next URL from `host`'s FIFO back queue.
4. Asynchronously fetch URL, extract content, and append to `data/news.jsonl`.
5. Reschedule `host` on the heap with `ready_time = now() + 8.0s`.

#### Crawl Budget & Timing Sanity Check
$$\text{Throughput} = \frac{5 \text{ sites}}{8.0 \text{ s/request}} = 0.625 \text{ req/s} \approx 2,250 \text{ articles/hour}$$
* **WAF Rate-Limit Compliance:** Indian news portals (Cloudflare/Akamai) typically throttle IPs sending $> 10\text{–}15$ req/min. An 8.0s per-host delay ($7.5\text{ req/min}$) safely avoids rate-limit blocks.
* **Crawl Schedule:** 5,000 articles in $\approx 2.2 \text{ hours}$; 12,000 articles in $\approx 5.3 \text{ hours}$.
* 5 sites run round-robin, so no site blocks others while waiting for its 8-second delay.

---

### Task 3: Feed & Seed Management (`sitemap.py`)
#### Specification & Scope
Discovers candidate article URLs across all target news portals using standard XML feeds, Google News feeds, and sitemap indexes.
* **Feed Types Supported:**
  - Standard `<urlset>` with `<loc>` and `<lastmod>`.
  - Google News sitemaps (`<news:news>`, `<news:publication_date>`, `<news:title>`).
  - Sitemap indexes (`<sitemapindex>`, `<sitemap>`) recursively resolved to child feeds.
  - Archive sitemaps (monthly / yearly archive feeds for historical depth up to ~12k articles).
* **Fault-Tolerant Parsing:**
  - Namespace-agnostic element matching (handling variations in `xmlns:news`).
  - Regex fallback parser for truncated or imperfect XML streams caused by network termination.
* **Conditional Polling:**
  - HTTP `304 Not Modified` handling via `If-None-Match` (ETag) and `If-Modified-Since` headers to avoid re-fetching unchanged sitemaps.

---

### Task 4: Content & Metadata Extraction (`extractor.py`)
#### Extraction Hierarchy & Strategy
* **Tier 1 (Schema.org JSON-LD):** Primary extraction targeting `schema.org/NewsArticle` or `BlogPosting` tags for `headline`, full `articleBody`, `datePublished`, `keywords`, and `articleSection`.
* **Tier 2 (Open Graph / Meta Tags):** Used strictly to recover missing metadata such as title, publication date, keywords, and section. `og:description` is never used as a substitute for the full article body.
* **Tier 3 (HTML DOM - Block-Level Fallback):**
  - Extract substantive body text from `<article>` or main story containers.
  - Collect text from `<p>` tags and non-nested leaf `<div>` blocks (e.g. Navbharat Times `<div class="Normal">`), filtering out captions, advertisements, bylines, and related-content widgets.
  - Prevent double-counting by prioritizing `<p>` blocks or taking only deepest leaf blocks $\ge 25$ characters.
* **Fields Extracted:**
  - `headline`: Title string in Devanagari.
  - `body`: Clean body text without ads, captions, or navigation.
  - `date`: Publication timestamp converted to **ISO-8601 in IST (`+05:30`)**.
  - `section`: Normalized category slug.
  - `state` / `city`: Extracted from URL path patterns; set to `null` if not reliably identified.
  - `keywords`: Extracted from metadata.
  - `links`: Collect in-body `<a href="...">` links to other crawled articles (for PageRank).
  - `content_hash`: MD5 hash of normalized article body.
  - `agency_flag`: Boolean flag for syndicated wire stories.
  - `dup_of`: Initialized to `null` (resolved during corpus clustering).
* **Privacy Enforcement:** **No author names** stored anywhere in the schema (bylines, creators, and authors stripped).
* **Schema Validation:** Records validated against `documentation/formats.md` via `validate_article_schema()`.

---

### Task 5: Pipeline Execution & Sample Crawl (`crawler.py`)
#### Asynchronous Pipeline Integration
Combines sitemap feed ingestion, Mercator frontier queuing, robots.txt gating, downloader, extraction, and streaming JSONL storage into a unified crawl pipeline.
* **Pipeline Flow:**
  1. Initialize `RobotsParser`, `MercatorFrontier`, and persistent `httpx.AsyncClient`.
  2. Parse seed sitemaps via `sitemap.py` and enqueue discovered candidate URLs into Frontier `Q1` (Fresh live sitemaps).
  3. Single-threaded async event loop continuously pops eligible host and URL from `MercatorFrontier` (strictly enforcing $\ge 8.0\text{s}$ per-host politeness delay).
  4. Evaluate robots.txt compliance (`robots.can_fetch(url)`); skip disallowed paths immediately.
  5. Fetch article HTML with resilient error handling:
     - **HTTP 429 (Rate Limit):** Apply per-host exponential backoff ($8\text{s} \to 16\text{s} \to 32\text{s} \to 64\text{s}$) and reschedule host.
     - **HTTP 403 / CAPTCHA Detection:** Disable the affected host for the remainder of the session to avoid IP bans.
     - **HTTP 5xx / Network Drops:** Retry up to 2 times with exponential backoff before discarding URL.
  6. In a strict `finally` block, always call `frontier.complete_request(host, delay=eff_delay)` to guarantee politeness and avoid orphaned in-flight locks.
  7. Extract structured article via `extractor.py` and validate schema against `documentation/formats.md`.
  8. Stream validated records line-by-line to output file with immediate `flush()`.
  9. Harvest in-body candidate hyperlinks, normalize and filter them, and push novel URLs into Frontier `Q2` (In-article hyperlinks).

#### Sampling Strategy & Execution Modes
* **Normal Mode (`--max-articles <N>`):**
  - Streams validated articles to `data/news.jsonl`.
  - Retains the first 300 records in memory and auto-snapshots them to `data/news_sample_300.jsonl` upon reaching article #300.
  - Continues crawling until `--max-articles` (default 5,000) or frontier exhaustion.
* **Sample Mode (`--sample`):**
  - Targets exactly 300 valid articles across sources.
  - Writes directly to `data/news_sample_300.jsonl` and terminates immediately (~8 minutes total runtime), unblocking the H3 team handoff without running the full ~2.2-hour corpus crawl.

---

### Task 6: Near-Duplicate & Story Clustering (`dedup.py`)
#### Why It's Feasible & Grounded in Literature
Newspapers frequently republish identical or slightly reworded wire stories from PTI, ANI, and Univarta/Bhasha:
* **Exact Duplicates:** MD5 hash of normalized body text $\to$ `content_hash`.
* **4-Word Shingles ($n=4$):** Grounded in Broder (1997) syntactic clustering adapted for Hindi grammar. In Hindi, 1- and 2-word shingles collide excessively due to frequent postpositions (*का, की, के, में, से*) and auxiliary verbs (*है, था, रहे*). 4-word shingles capture full syntactic clauses while tolerating minor regional word swaps.
* **Temporal Window ($\pm 24\text{ hours}$):** Wire syndications occur on the same news cycle; articles published days apart are distinct follow-ups rather than syndicated reprints.
* **MinHash + LSH (Scalable Candidate Selection):** 64-permutation MinHash signatures partitioned into $b=16$ bands of $r=4$ rows for $O(1)$ candidate bucketing.
* **Story Clustering for `dup_of`:** If Jaccard $\ge 0.70$, the earliest published article is the root (`dup_of: null`), and later articles point to it (`dup_of: "<earliest_doc_id>"`).
* **Agency Flag:** Detect wire keywords (`"पीटीआई"`, `"भाषा"`, `"वार्ता"`, `"ANI"`, `"PTI"`) and set `agency_flag: true`.
* **Empirical Calibration Note:** The $0.70$ Jaccard threshold will be calibrated empirically using the Precision-Recall curve evaluated on 100 labeled article pairs once real articles are crawled.

---

### Task 7: Adaptive Recrawling & Event Prioritization (`recrawl.py`)
#### Adaptive Recrawling (Freshness Tracking)
News sites publish sitemaps (`sitemap.xml` / `news-sitemap.xml`) updated throughout the day. Grounded in the **Cho & Garcia-Molina change-rate model** and **RFC 6298** smoothing standards:
* **Track Source Velocity:** Count new URLs added ($\Delta N$) over elapsed time ($\Delta t$) each time a sitemap is checked:
  $$\lambda_s^{(t)} = \alpha \cdot \frac{\Delta N}{\Delta t} + (1 - \alpha) \cdot \lambda_s^{(t-1)} \quad (\text{EWMA with } \alpha = 0.3)$$
  - $\alpha = 0.3$ is the standard smoothing factor from RFC 6298 network estimation, balancing response to sudden shifts with stability against short-term noise.
* **Dynamic Polling Schedule:** Compute next sitemap check interval $\tau_s$:
  $$\tau_s = \max\left(\tau_{\min},\; \min\left(\tau_{\max},\; \frac{K}{\lambda_s + \epsilon}\right)\right)$$
  - $\tau_{\min} = 30\text{ minutes}$: matches the typical 15–30 minute batch regeneration cycle of Indian news sitemaps.
  - $\tau_{\max} = 6\text{ hours}$: matches the overnight news lull (11 PM to 5 AM) in Indian newsrooms.
* **Conditional Fetching:** Send `If-Modified-Since` or `If-None-Match` (ETag) headers. A `304 Not Modified` terminates immediately, saving bandwidth and processing.

#### Event / Burst-Aware Prioritization
Breaking events (e.g. weather alerts, elections, accidents) produce sharp spikes in article volume under specific sections:
* **Time Windows:** 
  - Rolling 60-minute window captures the acute 45–90 minute breaking news cycle in Indian regional dailies.
  - 6-hour moving baseline provides diurnal smoothing without confounding daytime peaks with overnight troughs.
* **Detecting the Burst:**
  $$\text{BurstScore}(\text{category}) = \frac{\text{Count}_{1\text{h}}}{\text{MovingAvg}_{6\text{h}} + 1}$$
* **Front Queue Routing & Starvation-Free Biasing:**
  - If $\text{BurstScore} > 2.0$, new URLs from that category route into **Front Queue $Q_0$ (Highest Priority)**.
  - Biased selector samples: $P(Q_0)=0.60$, $P(Q_1)=0.25$, $P(Q_2)=0.10$, $P(Q_3)=0.05$.
  - Following Mercator (Heydon & Najork, 1999), biased random sampling guarantees **starvation-free prioritization** (avoiding strict priority queues that would completely stall routine sitemaps during prolonged bursts).
* **Politeness Preserved:** Prioritization only decides which URL is placed next in that host's back queue. Network requests remain strictly gated by the min-heap at 8 seconds per host.
* **Empirical Calibration Note:** The $2.0$ surge ratio and $60\%$ probability weight are baseline values that will be experimentally evaluated on live crawl data (measuring detection latency vs. false alarm rate).

---

### Task 8: Shared Evaluation Tooling (`dhvani/eval/pool.py`)
* **TREC Run Pooling:** Read standard TREC run files: `qid Q0 doc_id rank score run_name`.
* **Pool Formation:** Extract top-$k$ ($k=10$ or $20$) documents per information need across all runs.
* **Judgment Template Generation:** Deduplicate documents and generate human judgment sheets: `need_id 0 doc_id rel`.
* **Handoff Documentation:** Document ingestion format, pool utility, and schema in `documentation/handoffs/riya.md`.

---

### Optional Enhancement: Neural / Semantic Utility Prioritization (`dhvani/crawl/utility.py`)
#### Grounding in Literature & Concept
Standard focused crawlers target a specific query, which is unsuited for a general search engine where queries are unknown in advance. 

Following the retrieval-agnostic utility framework in **[Neural Prioritisation for Web Crawling](https://eprints.gla.ac.uk/359292/)** (Macdonald et al., University of Glasgow), we estimate a **retrieval-agnostic Corpus Utility model** $\mathcal{U}(u) \in [0, 1]$.
It predicts how much valuable, novel, and substantive information an unvisited URL will add to the overall corpus, using strictly pre-fetch crawl-time features without touching search queries or test relevance labels.

#### Crawl-Time Features Available (Retrieval-Agnostic)
1. **Anchor / Headline Text:** Discovered from referring page `<a href="...">anchor</a>` or `<news:title>` in sitemaps.
2. **Entity Density:** Salient entities (Hindi city names, state names, numbers/dates) present in the anchor/title text.
3. **URL Depth & Section Weight:** High-information sections (`national`, `state`, `weather`) prioritized over shallow filler.
4. **Corpus Novelty (Semantic Distance):** Using a lightweight embedding model (e.g. `paraphrase-multilingual-MiniLM-L12-v2` or fastText):
   - Compute embedding $\vec{e}_u$ of the candidate title/anchor text.
   - Compare against the running centroid $\vec{c}_{\text{sec}}$ of articles already crawled in that category.
   - Novelty score: $\text{Nov}(u) = 1 - \cos(\vec{e}_u, \vec{c}_{\text{sec}})$.
5. **Structural In-Degree:** Number of internal pages linking to this candidate URL.

#### Utility Score Formulation
$$\mathcal{U}(u) = w_1 \cdot \text{InformationDensity}(u) + w_2 \cdot \text{Novelty}(u) + w_3 \cdot \text{Freshness}(u) + w_4 \cdot \text{SectionWeight}(u)$$

#### Integration into Mercator Front Queues
The utility score determines Front Queue routing:
* $Q_0$ (Highest Priority): $\mathcal{U}(u) \ge 0.75$
* $Q_1$ (High Priority): $0.50 \le \mathcal{U}(u) < 0.75$
* $Q_2$ (Standard Priority): $0.25 \le \mathcal{U}(u) < 0.50$
* $Q_3$ (Low / Archive): $\mathcal{U}(u) < 0.25$

#### Politeness Preserved
The utility score only controls which front queue receives the URL. Per-host back queues and the min-heap strictly enforce the 8.0s per-host delay.

#### Downstream Pre-work
Crawl-time utility scores $\mathcal{U}(u)$ can be exported alongside articles as a pre-computed static document quality feature $g(d)$ for downstream ranking.

---

## 4. Contract Compliance: `documentation/formats.md`
All crawled articles must be written to `data/news.jsonl` (one JSON line per article):
```json
{
  "doc_id": "jagran_23456789",
  "url": "https://www.jagran.com/uttar-pradesh/lucknow-weather-update-23456789.html",
  "source": "jagran",
  "section": "weather",
  "state": "uttar-pradesh",
  "city": "lucknow",
  "date": "2026-10-06T13:21:43+05:30",
  "headline": "कल का मौसम: उत्तर प्रदेश में भारी बारिश का अलर्ट",
  "body": "लखनऊ और आसपास के इलाकों में कल सुबह से ही बादलों की आवाजाही रहेगी...",
  "keywords": ["मौसम", "बारिश", "लखनऊ"],
  "agency_flag": false,
  "content_hash": "a9f3b821...",
  "dup_of": null,
  "links": ["jagran_23450001"]
}
```

* `date` must be ISO-8601 in IST timezone (`+05:30`).
* `dup_of` is the `doc_id` of the original article if near-duplicate, otherwise `null`.
* `links` is a list of other crawled `doc_id`s found in this article's body.
* `agency_flag` is boolean (`true` if wire story).
* No author names stored.

---

## 5. Test Suite (`partwise-tests/riya/`)
* **`test_robots.py`:** Unit test comparing custom `RobotsParser` vs `urllib.robotparser` on cached `robots.txt` files from Jagran, Jansatta, Live Hindustan, and Amar Ujala.
* **`test_frontier.py`:** Verify that the min-heap strictly enforces $\ge 8.0$ seconds delay per host and front queues respect priority weights.
* **`test_normalizer.py`:** Verify URL cleaning, AMP-to-canonical conversion, and query parameter stripping.
* **`test_extractor.py`:** Verify JSON-LD extraction, IST date parsing, and absence of author names on sample HTML fixtures.
* **`test_crawler.py`:** Verify crawl loops, HTTP 429 backoff, HTTP 403 host quarantine, and in-body link harvesting.
* **`test_dedup.py`:** Precision & recall on 100 labeled article pairs, plus speed comparison of exact Jaccard vs MinHash + LSH.
* **`test_recrawl.py`:** Verify EWMA change-rate tracking, HTTP 304 conditional scheduling, and surge detection queue routing.
* **`test_format_compliance.py`:** JSON schema validation ensuring every record in `data/news.jsonl` adheres to `formats.md`.
* **`test_pool.py`:** Verify shared TREC run pooling aggregating by `need_id`, multi-form candidate deduplication, `--raw-qid` override, and Format 5 compliance.

---

## 6. Milestones & Checklist
* [x] **Phase 1 (H1–H3):**
  - [x] Implement `robots.py` and unit tests.
  - [x] Build `frontier.py` (asyncio + heapq) and sitemap parser.
  - **Handoff (H3):** Generate and provide `data/news_sample_300.jsonl` (300 clean articles).
* [x] **Phase 2 (H3–H8):**
  - [x] Implement `extractor.py`, `normalizer.py`, and `filters.py`.
  - [x] Start continuous 5-site crawling.
* [x] **Phase 3 (H8–H12):**
  - [x] Implement `dedup.py` (shingles, Jaccard, `dup_of`, `links`).
  - [x] Monitor crawl logs and site balance.
* [x] **Phase 4 (H12–H22):**
  - [x] Add MinHash + LSH, adaptive recrawl (`recrawl.py`), and burst prioritization.
  - **Corpus Freeze (H18):** Lock master news corpus at 5k–12k articles (5,001 articles crawled).
* [x] **Phase 5 (H22–H28):**
  - [x] Implement `dhvani/eval/pool.py` (need-level aggregation, cross-form deduplication, `--raw-qid`).
  - [x] Generate evaluation tables and figures (`documentation/results/riya-corpus-results/` and `documentation/figures/`).
* [ ] **Phase 6 (H28–H36):**
  - Write report section (crawling, robots.txt findings, deduplication).
  - Record video segment (crawler live log, robots test demo, duplicate story clusters).