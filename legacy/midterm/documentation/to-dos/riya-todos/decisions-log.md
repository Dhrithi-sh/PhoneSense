# Architectural & Design Decisions (Riya - P1)

## Task 1: Setup, Compliance & Filtering
- **Target Sites & Blacklist:**
  - 5 primary sites (*Dainik Jagran, Navbharat Times, Live Hindustan, Amar Ujala, Aaj Tak*) and 2 backups (*Jansatta, Bhaskar*).
  - Blacklisted *BBC Hindi* (robots.txt forbids scraping/datasets) and *News18 / NDTV* (Cloudflare bot blocks).

- **Ethical Crawling & Privacy:**
  - User-Agent set to `CollegeProject_NewsBot(+https://github.com/InformationRetrieval-Mid/Dhvani-lingual; contact: ra810@snu.edu.in)`.
  - Zero author names stored anywhere in the dataset.
  - Full article text stays strictly on local machines (`data/*.jsonl` gitignored); only code and URLs go to GitHub.

- **URL Normalization & Route Filtering:**
  - Implemented scheme/domain lowercasing, stripping default ports and trailing slashes.
  - Tracking parameters (`utm_*`, `ref`, `fbclid`, `amp_js_v`) are stripped prior to frontier queueing to prevent duplicate crawl budget waste.
  - AMP URLs (`/amp/`, `?amp=1`) are mapped directly to canonical desktop versions.
  - Route filter excludes non-article routes (`/rashifal/`, `/astrology/`, `/photo-gallery/`, `/videos/`, `/live-updates/`) to avoid corpus keyword pollution.

- **Robots Parser Strategy & Test Outcome:**
  - Originally considered wrapping Python's standard `urllib.robotparser` directly.
  - **Empirical Test Result:** Testing against cached target site `robots.txt` files in [`test_robots.py`](../../../partwise-tests/riya/test_robots.py) demonstrated that `urllib.robotparser` performs literal prefix matching and fails to parse wildcard rules (`*` and `$` patterns) on *Dainik Jagran* (e.g. `/search/*`, `/state/*`) and *Amar Ujala* (e.g. `/*?utm_*`), incorrectly returning `can_fetch == True` for disallowed URLs.
  - **Decision:** Following our decision rule, we implemented custom RFC 9309 pattern-to-regex matching logic inside [`dhvani/crawl/robots.py`](../../../dhvani/crawl/robots.py) to properly handle wildcards, longest-prefix rule precedence, per-host caching, user-agent fallback (`CollegeProject_NewsBot` -> `*`), and `Sitemap:` extraction.
  - No attempt to bypass Disallow rules by altering user-agent.

---

## Task 2: Mercator Frontier & Scheduler
- **Frontier Concurrency Model (`asyncio` + `heapq`):**
  - Use a single-threaded asynchronous event loop with `asyncio` and `heapq` implemented in [`dhvani/crawl/frontier.py`](../../../dhvani/crawl/frontier.py).
  - Maintain Front queues for URL priority ($Q_0$–$Q_3$), one FIFO back queue per host for politeness, and min-heap `(next_allowed_time, entry_id, host)` for scheduling.
  - When a host becomes eligible, asynchronously dispatch its next URL; while waiting on network I/O, the event loop handles another eligible host.
  - After a request completes, reschedule that host on the heap with the required 8-second delay.
  - Theoretical scheduling across 5 hosts is $8/5 = 1.6\text{ s/request}$.
  - Single-threaded model avoids locks, thread synchronization, and concurrent file-write conflicts while overlapping network I/O.
  - **Empirical Test Result & Verification:** Validated in [`test_frontier.py`](../../../partwise-tests/riya/test_frontier.py):
    - Multi-host concurrency verified: distinct hosts dispatch concurrently with zero inter-host blocking.
    - Politeness verified: consecutive requests to the same host strictly wait $8.0\text{s}$ via deterministic simulated clock.
    - Deterministic tie-breaking: monotonic `entry_id` resolves simultaneous host eligibility in strict FIFO order.
    - Biased selection verified: 1,000-iteration simulation confirmed priority distribution ($Q_0 \approx 70.6\%$ relative ratio over $Q_1$) without starving lower queues.
    - Bounded back queues (`max_back_queue_size = 10`) eliminate head-of-line blocking during burst spikes.

---

## Task 3: Feed & Seed Management
- **Sitemap Ingestion & Conditional Discovery:**
  - Robust namespace-agnostic XML parser implemented in [`dhvani/crawl/sitemap.py`](../../../dhvani/crawl/sitemap.py) supporting standard (`<urlset>`), Google News (`<news:news>`), and sitemap index (`<sitemapindex>`) feeds.
  - Live survey confirmed working Google News sitemaps across all 5 primary sources (`jagran.com`, `navbharattimes.indiatimes.com`, `livehindustan.com`, `amarujala.com`, `aajtak.in`).
  - Added regex fallback for truncated/imperfect XML streams so network interruptions do not cause unhandled parse crashes.
  - Implemented HTTP 304 conditional request support (`If-None-Match`, `If-Modified-Since`) to eliminate redundant bandwidth consumption during polling.
  - **Empirical Test Result & Verification:** Validated in [`test_sitemap.py`](../../../partwise-tests/riya/test_sitemap.py) (7/7 tests passing including Google News fields, index discovery, conditional headers, and malformed stream resilience).

---

## Task 4: Content & Metadata Extraction
- **Article & Metadata Extraction Pipeline:**
  - 3-tier hierarchy in [`dhvani/crawl/extractor.py`](../../../dhvani/crawl/extractor.py):
    - **Tier 1 (Schema.org JSON-LD):** Primary source for `headline`, full `articleBody`, `datePublished`, `keywords`, and `articleSection`.
    - **Tier 2 (Open Graph / Meta Tags):** Used strictly to recover missing metadata (title, publication date, keywords, section). `og:description` is never used as a substitute for the full article body.
    - **Tier 3 (HTML DOM - Block-Level Fallback):**
      - **Problem:** Many Indian news CMSs (e.g. Navbharat Times `<div class="Normal">`, Amar Ujala story blocks) wrap body paragraphs in `<div>` elements rather than semantic `<p>` tags. Strict `<p>`-only extraction caused empty body false negatives.
      - **Risk (Double-Counting):** Naive iteration over `find_all(["p", "div"])` causes parent wrapper `<div>`s and child `<div>`/`<p>` tags to repeat identical text multiple times.
      - **Solution:** Decompose boilerplate elements (captions, image credits, related-content boxes, ads, social shares, bylines). Extract text from `<p>` tags and non-nested leaf `<div>` blocks (substantive length $\ge 25$ chars). When `<p>` tags provide substantive text, they take precedence; otherwise, leaf block `<div>` elements are aggregated without parent duplication.
  - **Strict IST Date Formatting:** All timestamps parsed and converted to ISO-8601 with `+05:30` offset.
  - **Zero Author Privacy Enforcement:** Clearly identified author/byline elements removed where necessary, avoiding blind wildcard deletion of classes/IDs containing "author". Record schema strictly excludes author, creator, editor, and byline fields.
  - **Conservative Geographic Tagging:** If state or city cannot be reliably identified from URL path structures, fields are explicitly set to `null` rather than guessing.
  - **Contract Validation:** Extracted articles must pass `validate_article_schema()` verifying `documentation/formats.md` compliance before being written to `data/news.jsonl`.
  - **Embedded CMS JSON Metadata Filtering:**
    - On Indian news portals (e.g. *Amar Ujala* story containers), CMSs occasionally embed raw JSON metadata blobs (e.g. `{"_id":"...", "slug":"..."}`) inside content wrapper elements.
    - Added a filter in `_extract_body_from_dom` detecting and skipping blocks starting with JSON object syntax (`text.startswith("{") and "}" in text[:100]`), keeping body prose clean.
  - **Hindi-Belt District Bureau Expansion & City Routing Conventions:**
    - Expanded `INDIAN_CITIES` from 47 tier-1 cities to cover over 120 prominent district centers across the 9 Hindi-belt states (e.g. *Basti, Ratlam, Kangra, Sikar, Pithoragarh, Alwar, Hapur, Dehradun*).
    - Added pattern recognizers for portal routing conventions: Jagran's `/<state>/<city>-city-...` slug prefix and `new-delhi` compound naming.
    - Result: City recognition on regional news increased from 33.3% (100/300) to 55.0% (165/300) on the live sample, while preserving strict `null` compliance for genuinely national, international, cricket, and state-wide policy stories.
  - **Deterministic City-to-State Derivation & Delhi Normalization:**
    - Guaranteed that no article ever has a non-null `city` with a `null` state: added `CITY_TO_STATE` mapping covering 100% of Hindi-belt cities to deterministically derive the parent state if missing from the URL.
    - Implemented Delhi city-state equivalence: articles tagged with `state: "delhi"` automatically default to `city: "delhi"` (or `"new-delhi"` if present in path), eliminating 100% of geographic field inconsistencies.
  - **Empirical Test Result & Verification:** Validated in [`test_extractor.py`](../../../partwise-tests/riya/test_extractor.py) (JSON-LD priority, meta fallback without og:description substitution, DOM block-level and leaf-div extraction without double-counting, IST date parsing, author removal, and schema compliance).

---

## Task 5: Pipeline Execution & Sample Crawl
- **Sample Handoff Sequencing & Dual Modes:**
  - Defer generating the 300-article sample (`news_sample_300.jsonl` for H3) until initial extractor, normalizer, and crawler pipeline are complete.
  - Implemented dual operational modes:
    - Dedicated `--sample` mode stopping immediately after 300 valid articles (~8 minutes runtime) to provide the H3 deliverable without waiting for the full 2.2-hour corpus.
    - Normal mode crawling toward `--max-articles` (default 5,000) while auto-saving the first 300 articles snapshot simultaneously to `data/news_sample_300.jsonl` upon reaching article #300.

- **Streaming JSONL Writer & Checkpointing:**
  - Single-threaded asynchronous pipeline streams validated JSON lines directly to disk with immediate `.flush()` after every record, preventing data loss on unexpected termination.
  - Tracks in-memory `seen_doc_ids` to eliminate intra-session duplicate writes.

- **Anti-Bot, WAF & Error Resilience Policies:**
  - **HTTP 429 (Rate Limit):** Doubles per-host delay exponentially ($8.0\text{s} \to 16.0\text{s} \to 32.0\text{s} \to 64.0\text{s}$) and reschedules the host rather than crashing the crawler.
  - **HTTP 403 & CAPTCHA Challenges:** Cloudflare/Akamai bot challenge signatures immediately disable the affected host for the current crawl session, preventing further requests that could trigger hard IP bans.
  - **HTTP 5xx & Network Timeouts:** Transient failures retried up to 2 times with exponential backoff before dropping the URL.
  - **Strict Politeness Rescheduling:** `frontier.complete_request(host, delay=eff_delay)` is executed in a strict `finally` block, ensuring no host is orphaned in-flight regardless of fetch or extraction failure.

- **Dynamic Per-Host Delay Scheduling (`frontier.py` Integration):**
  - Extended `frontier.complete_request(host, delay=eff_delay)` to accept an optional custom delay parameter, allowing `crawler.py` to reschedule rate-limited hosts with backed-off intervals while preserving the default 8.0s delay for all healthy hosts.

- **Discovered-Link Harvesting:**
  - Extracted in-body hyperlinks are normalized via `normalizer.py`, filtered via `filters.py`, deduplicated against `frontier.seen_urls`, and pushed into Frontier Front Queue $Q_2$ (In-article hyperlinks, 10% sampling weight), preventing link discoveries from bypassing polite crawling gates.

- **Native Async Test Harness (Portable Verification):**
  - Replaced `@pytest.mark.asyncio` annotations with standard library `asyncio.run(_test())` coroutine execution, eliminating external pytest plugin dependencies and guaranteeing 100% portable test execution.
  - Empirical verification: 9/9 crawler unit tests passing in 2.2s in [`test_crawler.py`](../../../partwise-tests/riya/test_crawler.py).

- **Head-Of-Line (HOL) Starvation Prevention & Multi-Source Queue Balancing:**
  - *Symptom:* During live crawl runs, articles were retrieved exclusively from a single source (*Dainik Jagran*), starving the other 4 sources (*NBT, Live Hindustan, Amar Ujala, Aaj Tak*) despite successful seed discovery.
  - *Root Cause 1 (Sequential Sitemap Ingestion & Front Queue HOL):* Seeds were originally enqueued per-source sequentially. Jagran URLs filled the head of $Q_1$. In `_refill_back_queues()`, when Jagran's back queue reached capacity (`max_back_queue_size = 10`), a premature `break` terminated refills for all other hosts in the queue.
  - *Fix 1 (Frontier Queue Rotation):* Replaced the refill loop `break` with `q.rotate(-1)` when a host's back queue is full, rotating the full host's URL to the tail so downstream candidate hosts are evaluated without blocking.
  - *Fix 2 (Round-Robin Seed Interleaving):* In `bootstrap_seeds()`, sitemap URLs across all active sources are interleaved round-robin into $Q_1$, ensuring equal representation across sources from crawl startup.
  - *Root Cause 2 (Heap Duplication & Delay Collapse):* `complete_request()` previously called `_refill_back_queues()` (which scheduled the host on the heap) and then pushed the host onto the heap a second time unconditionally. Duplicate entries with expired timestamps bypassed the 8.0s politeness delay.
  - *Fix 3 (Single-Entry Heap Invariant):* Rescheduled the host on the min-heap strictly once with `hosts_in_heap` tracking before refilling back queues, guaranteeing at most one heap entry per host at all times.
  - *Root Cause 3 (Bulk Ingestion Queue Spin Stall):* After discovering ~10,000 URLs across sitemaps, transferring URLs into back queues spun through unneeded rotations (`O(len(q))` per call) when back queues were already saturated, causing an apparent freeze at startup.
  - *Fix 4 (State-Driven Back Queue Saturation Detection):* Rather than arbitrary constant skips, `_refill_back_queues()` explicitly computes `hungry_hosts = {h for h, bq in self.back_queues.items() if len(bq) < self.max_back_queue_size}`. If `not hungry_hosts and len(self.back_queues) >= 5`, it terminates in $O(1)$ time with 0 rotations. During rotation, skips are dynamically bounded by known active hosts (`consecutive_skips >= max(len(self.back_queues), 10)`), completely eliminating heuristic magic numbers while preventing bootstrap starvation.
  - *Root Cause 4 (CAPTCHA False Positives on CSS Stylesheets):* The initial signature list included generic substrings like `"recaptcha"`. Normal article pages on *Live Hindustan* containing `.grecaptcha-badge` in their `<style>` blocks triggered false-positive bot challenge detections on HTTP 200 responses, erroneously disabling healthy domains for the entire session.
  - *Fix 5 (High-Precision Interception Signatures & Content Guards):* Removed ambiguous substrings from `CAPTCHA_SIGNATURES`, narrowed down to verified challenge page markers (Cloudflare `cf-challenge-running`, `<title>Just a moment...</title>`), and added structural content guards ensuring pages with valid `schema.org/NewsArticle` JSON-LD or `<article>` tags are never classified as interstitial blocks.

---

## Task 6: Deduplication & Story Clustering
- **Deduplication & Story Clustering (MinHash + LSH):**
  - 3-tier pipeline implemented in [`dhvani/crawl/dedup.py`](../../../dhvani/crawl/dedup.py): Exact MD5 `content_hash` -> 4-word shingles -> 64-permutation MinHash with LSH ($b=16, r=4$) for scalable candidate bucketing.
  - 4-word shingles grounded in Broder (1997) adapted for Hindi syntax (prevents collisions on high-frequency postpositions like *का, की, के, में, से*).
  - Deterministic MinHash universal hashing: $h_i(x) = (a_i \cdot x + b_i) \pmod p$ with Mersenne prime $p = 2^{31} - 1$ and deterministic CRC32 shingle pre-hashing, guaranteeing platform-independent execution.
  - LSH configuration ($b=16, r=4$) yields an S-curve midpoint $(1/16)^{1/4} = 0.50$, providing $> 98.8\%$ collision probability for pairs with $J \ge 0.70$.
  - $\pm 24$-hour temporal window for clustering syndicated wire stories (PTI, ANI, Bhasha).
  - Earliest published story marked as canonical root (`dup_of: null`), later reprints set `dup_of: "<earliest_doc_id>"`.
  - In-corpus hyperlink pruning: filters `links` array to retain only existing non-self `doc_id`s, preventing dangling edges in downstream PageRank calculation.
  - Command-line interface provided for batch clustering over arbitrary JSONL files: `python -m dhvani.crawl.dedup --input <in.jsonl> --output <out.jsonl>`.

- **Empirical Calibration & Verification (`test_dedup.py`):**
  - **Fixture Separation:** Benchmark of 100 labeled Hindi article pairs (40 positive wire duplicates and rewrites, 60 negative topic-distinct, unrelated, and temporal-mismatch pairs) stored in [`partwise-tests/riya/fixtures/dedup_pairs_100.json`](../../../partwise-tests/riya/fixtures/dedup_pairs_100.json) rather than hardcoded in the test file.
  - **Calibration Results Across Jaccard Thresholds:**
    | Threshold $J$ | TP | FP | TN | FN | Precision | Recall | F1 Score |
    |---|---|---|---|---|---|---|---|
    | $0.60$ | 40 | 0 | 60 | 0 | 1.0000 | 1.0000 | 1.0000 |
    | $0.65$ | 40 | 0 | 60 | 0 | 1.0000 | 1.0000 | 1.0000 |
    | **$0.70$ (Optimal)** | **38** | **0** | **60** | **2** | **1.0000** | **0.9500** | **0.9744** |
    | $0.75$ | 26 | 0 | 60 | 14 | 1.0000 | 0.6500 | 0.7879 |
    | $0.80$ | 17 | 0 | 60 | 23 | 1.0000 | 0.4250 | 0.5965 |
  - **Conclusion:** $J = 0.70$ provides zero false positives ($\text{Precision} = 1.00$) while maintaining high recall ($95.0\%$) on regional rewrites. At $J = 0.80$, recall collapses to $42.5\%$ due to regional vocabulary variations in introductory lead sentences.
- **Real-Corpus Diagnostic Findings (`data/news_sample_300.jsonl`):**
  - Evaluated on the live 300-article corpus across all 5 primary sources: 300 total articles, 300 singletons, 0 multi-article clusters at $J \ge 0.70$.
  - **Nearest-Match Pair Analysis (Empirical Upper Bound):**
    - *Char Dham Yatra Record:* Jagran (`jagran_40396772`) vs NBT (`nbt_134746829`) at $J = 0.4544$ (234 shared 4-word shingles); Amar Ujala (`amarujala_2b7172e9`) at $J = 0.3547$. Caught by `MinHashLSH` bucket collision.
    - *Shreyas Iyer Press Statement:* Live Hindustan (`livehindustan_2dd0222e`) vs NBT (`nbt_134747344`) at $J = 0.4443$ (295 shared 4-word shingles).
  - **Architectural Distinction (Syndicated Wire vs. Independent Reporting):**
    - The 300-article sample comprises ~60 articles per outlet primarily from district editions (*e.g. Kanpur, Meerut, Gorakhpur, Varanasi*), authored by local correspondents rather than wire desks.
    - Independent journalists covering the same press release or press conference paraphrase, reorder sections, and insert editorial commentary, capping exact 4-word shingle Jaccard at $0.35\text{--}0.45$.
    - Daily horoscopes (Aaj Tak) collide at $J \approx 0.04$ due to boilerplate closing text (*"शुभ अंक 5, 6, 7... पितृों का तर्पण करें"*). Dropping the threshold below $0.50$ would introduce catastrophic false positives on templated features.
    - Preserving $J = 0.70$ strictly complies with `formats.md` (`dup_of` = near-duplicate syndication, not general topic clustering).
  - **Invariant Verification:** 100% pass across all 300 records (every `dup_of` valid, zero self-links, canonical roots null, and zero dangling in-corpus hyperlinks).

- **Full Corpus Deduplication & Testing Plan (Milestone H18):**
  - **Context:** At 5,000–12,000 articles, national and business sitemaps will ingest large volumes of shared PTI, ANI, and Univarta syndicated wire feeds published concurrently across portals.
  - **Execution Command:**
    ```bash
    python -m dhvani.crawl.dedup --input data/news.jsonl --output data/news.jsonl
    ```
  - **Planned Evaluation Criteria:**
    1. *Wire Cluster Formation:* Verify canonical root resolution on national wire syndications ($J \ge 0.70$), expecting $2\%\text{--}5\%$ total corpus duplication.
    2. *MinHash LSH Scaling:* Benchmark candidate retrieval time across 5,000+ documents against theoretical $O(N^2)$ all-pairs comparison ($\sim 1.25 \times 10^7$ comparisons).
    3. *Contract Invariants:* Run automated assertion audit verifying that every non-null `dup_of` references an existing `doc_id`, roots have `dup_of == null`, and zero dangling hyperlinks exist in `links`.

---

## Task 7: Adaptive Recrawling & Event Prioritization
- **Adaptive Recrawling & Freshness Engine (`recrawl.py`):**
  - Implemented `AdaptiveRecrawler` tracking individual publication velocity per news source.
  - **Velocity Unit Conversion & EWMA Smoothing:**
    - To accurately match the publication rate unit ($\text{URLs/hour}$) with second-based epoch timestamps, $\Delta t$ is strictly converted:
      $$\Delta t_{\text{hours}} = \frac{\Delta t_{\text{seconds}}}{3600.0}$$
    - Observed arrival rate: $\text{observed\_rate} = \frac{\Delta N}{\Delta t_{\text{hours}}}$.
    - Exponentially Weighted Moving Average (EWMA) smoothing with $\alpha = 0.3$:
      $$\lambda_s^{(t)} = 0.3 \cdot \text{observed\_rate} + 0.7 \cdot \lambda_s^{(t-1)}$$
      grounded in the Cho & Garcia-Molina Poisson change-rate model and RFC 6298 smoothing standards.
  - **Dynamic Polling Interval Scaling ($K = 1800\text{s}$):**
    - Evaluated interval formula:
      $$\tau_s = \max\left(\tau_{\min},\; \min\left(\tau_{\max},\; \frac{K}{\max(\lambda_s, 0.0833)}\right)\right)$$
    - Using $K = \text{MIN\_POLL\_INTERVAL\_SECONDS} = 1800\text{s}$ strictly bounds nominal crawling ($\lambda_s = 1.0\text{ URL/hour}$) to $1800\text{s}$ ($30\text{ minutes}$), scaling up to $21600\text{s}$ ($6\text{ hours}$) during idle periods ($\lambda_s \le 0.0833\text{ URL/hour}$, i.e. 1 URL per 12 hours). High-velocity periods ($\lambda_s > 1.0$) clamp safely at $\tau_{\min} = 1800\text{s}$.
  - **HTTP 304 Conditional Bandwidth Optimization & Scheduling Anchor:**
    - Sitemap responses cache `ETag` and `Last-Modified` headers, injected conditionally as `If-None-Match` and `If-Modified-Since`.
    - On `304 Not Modified`, the crawler records $\Delta N = 0$ URLs without redownloading or re-parsing XML. The standard EWMA formula smoothly decays $\lambda_s$, naturally extending the re-poll interval without requiring arbitrary penalty multipliers.
    - **Anchor Timestamp Invariant:** Upon receiving 304, `self.last_checked[source_slug] = now` is updated to the timestamp of the 304 check. Subsequent evaluations of `should_poll(src, now)` evaluate $(\text{now} - \text{last\_checked}) \ge \tau_s$ using the newly stretched EWMA interval, ensuring the next poll is scheduled strictly starting from this 304 check.
  - **Velocity Measured by Newly Observed URLs ($\Delta N$) vs Raw Feed Size:**
    - When a sitemap returns 200 OK, candidate URLs already encountered in the crawler's seen set (`frontier.seen_urls`) are filtered out.
    - Arrival rate estimation $\Delta N / \Delta t$ uses only the count of **newly observed/published URLs** ($\Delta N = \text{len(new\_cands)}$), rather than simply the total number of entries in the sitemap XML response. This prevents historical XML archives from falsely inflating runtime publication velocity.
  - **Bootstrap Seeding Baseline Decoupling:**
    - During initial bootstrap, sitemaps establish header caches and seed the frontier, but set `change_rates[source] = 1.0` and `last_checked[source] = now`, establishing a clean nominal baseline before runtime velocity is tracked.

- **Burst-Aware Event Prioritization Engine:**
  - **Topical Surge Detection:** Decoupled from source velocity; publication volume is tracked per topical category across all news portals using rolling deques.
  - **Dual Rolling Windows:**
    - Acute window: $W_{\text{acute}} = 3600\text{s}$ ($1\text{ hour}$).
    - Baseline moving window: $W_{\text{baseline}} = 21600\text{s}$ ($6\text{ hours}$).
    - Moving hourly baseline: $\text{MovingAvg}_{6\text{h}} = \frac{\text{Count}_{6\text{h}}}{6.0}$.
    - Surge formula:
      $$\text{BurstScore} = \frac{\text{Count}_{1\text{h}}}{\text{MovingAvg}_{6\text{h}} + 1.0}$$
    - When $\text{BurstScore} > 2.0$, the category enters `BURST` state.
  - **Conservative Category Routing:**
    - Only candidate URLs with a verified category currently undergoing an active burst ($\text{BurstScore} > 2.0$) route into **Front Queue $Q_0$** (60% biased sampling weight).
    - Unverified, unknown, or non-bursting categories default strictly to routine sitemap queue $Q_1$ (25% weight).
  - **Automatic Event Decay:**
    - As breaking events subside and publication counts fall outside the acute 1-hour window, $\text{BurstScore}$ drops $\le 2.0$ and priority automatically normalizes back to $Q_1$.
    - Stale timestamps older than 6 hours are automatically pruned from memory.

- **Non-Preemptive FIFO Invariant in Mercator Frontier:**
  - Per-host back queues in `MercatorFrontier` are strictly FIFO.
  - Biased selection ($Q_0$) prioritizes which URL is selected to refill empty host back queue slots, but **never preempts or reorders URLs already inside a host's back queue**.
  - All outgoing requests remain strictly gated by the min-heap at $\ge 8.0\text{s}$ per host, ensuring politeness is never compromised during breaking news bursts.

- **Empirical Test Result & Verification:**
  - Validated in [`test_recrawl.py`](../../../partwise-tests/riya/test_recrawl.py) (11/11 tests passing):
    1. EWMA rate smoothing: Verified $\Delta t$ hour conversion and $\alpha = 0.3$ smoothing across varying intervals.
    2. Polling interval bounds: Verified $K=1800\text{s}$ formula and clamping to $[1800\text{s}, 21600\text{s}]$ across rates from $0.0$ to $5.0$.
    3. Conditional headers: Verified case-insensitive `ETag`/`Last-Modified` extraction and `If-None-Match`/`If-Modified-Since` generation.
    4. HTTP 304 handling: Verified $\Delta N = 0$ rate decay, `last_checked` update to 304 timestamp, and `should_poll` scheduling.
    5. Burst score calculation: Verified acute 1h vs baseline 6h ratio on steady-state ($0.5$) vs surge ($3.0$).
    6. Surge detection & routing: Verified $Q_0$ assignment for active burst categories and URL paths.
    7. Conservative routing: Verified fallback to $Q_1$ for unverified/unknown categories.
    8. Burst decay: Verified automatic reversion to $Q_1$ after 1 hour and deque pruning after 6 hours.
    9. Non-preemptive FIFO invariant: Verified $Q_0$ appends to the tail of host back queue without preempting existing items.
    10. NewsCrawler integration: Verified mock crawl loop triggers conditional sitemap polling and processes 304 responses.
    11. Newly observed URLs velocity tracking: Verified that newly observed URLs count ($\Delta N$) updates velocity rather than total sitemap response count.
  - Validated in [`test_format_compliance.py`](../../../partwise-tests/riya/test_format_compliance.py) (2/2 tests passing):
    - 100% pass across all 300 records in `data/news_sample_300.jsonl` verifying all 12 required fields, IST dates, zero author names, and geographic invariants.
  - Overall test suite: 72/72 passing tests across all modules.

---

## Task 8: Shared Evaluation Tooling
- **Information-Need Level Pooling Architecture (`eval/pool.py`):**
  - **Problem & Root Cause:** Run files generated by experimental runs output `qid Q0 doc_id rank score run_name` where `qid` contains language/form suffixes (e.g. `R01_hi`, `R01_hinglish`, `R01_messy`, `R01_en`). When `pool.py` pooled candidate pools on raw `qid`, it generated disjoint candidate pools per surface form (`R01_hi 0 doc_id 0`). Downstream evaluation (`qrels.txt`, `experiments.py`, `metrics.py`) operates at the information need level (`need_id`, e.g. `R01`), as mandated by `documentation/formats.md` Format 5 (`need_id 0 doc_id rel`). Raw `qid` pooling resulted in missing `qrels` lookups during evaluation and forced judges to evaluate the same document multiple redundant times across different linguistic surface forms of the same information need.
  - **Decision:** By default (`per_need=True`), `pool_runs()` aggregates all surface forms of an information need into a single unified candidate pool keyed by `need_id`.
  - **Need ID Extraction Logic (`extract_need_id`):**
    1. **Explicit Queries TSV Mapping:** When `--queries queries.tsv` is supplied, mapping table lookup takes highest precedence.
    2. **Fallback Prefix Parsing:** Splits on first underscore (`qid.split("_")[0]`), cleanly mapping `R01_hi`, `R01_hinglish`, `R01_messy`, `R01_en` to `R01`.
    3. **Passthrough for Unsuffixed QIDs:** Query IDs without underscores (such as `N01` or standard TREC numeric IDs) pass through unchanged.
  - **Cross-Form Candidate Deduplication:**
    - Documents retrieved across multiple surface forms of the same need (e.g., retrieved for both `R01_hi` and `R01_hinglish`) are deduplicated within the `need_id` pool.
    - Candidate reduction naturally scales with run overlap across linguistic variants without hardcoding empirical bounds.
  - **Format 5 Compliance & Backward Compatibility:**
    - Output strictly adheres to `documentation/formats.md` Format 5: `need_id 0 doc_id 0`.
    - Added `--raw-qid` CLI flag to preserve raw per-surface-form pooling for exceptional edge cases where distinct surface form assessment is explicitly requested.
  - **Empirical Test Result & Verification:**
    - Validated in [`test_pool.py`](../../../partwise-tests/riya/test_pool.py) (8/8 tests passing):
      1. Aggregation of `R01_hi`, `R01_hinglish`, `R01_messy`, `R01_en` into `R01`.
      2. Multi-form candidate deduplication within the `R01` pool.
      3. `--raw-qid` preservation of raw qids.
      4. Passthrough of unsuffixed queries like `N01`.
      5. `--queries queries.tsv` precedence over underscore splitting fallback.
      6. Strict Format 5 line formatting (`need_id 0 doc_id 0`).
      7. CLI parsing and file generation integration.

- **Corpus Results Documentation & Visualization Reorganization:**
  - Migrated crawl, deduplication, and sample stats documentation from temporary storage (`data/result-documentation/`) into dedicated track structure: `documentation/results/riya-corpus-results/`:
    - `crawl_stats.md`: Source balance and geographic distribution across 11 states / 120+ cities.
    - `dedup_stats.md`: 4-word shingle Jaccard threshold calibration curve and speed benchmarks.
    - `sample_corpus_stats.md`: H3 300-article seed sample verification.
  - Generated publication-ready figures in `documentation/figures/`:
    - `riya-dedup-precision-recall.png`: Deduplication precision-recall curve across Jaccard thresholds $J \in [0.60, 0.80]$.
    - `riya-corpus-distribution.png`: Per-source article breakdown and regional state/city distribution.
    - `riya-recrawl-bandwidth.png`: Freshness adaptation and HTTP 304 conditional request bandwidth savings.
  - Automated plot generation script maintained at `scripts/plot_corpus_metrics.py`.

---

## Optional Enhancement & Research Grounding
- **Semantic Prioritization Grounding:**
  - Added academic reference for retrieval-agnostic corpus utility: [Neural Prioritisation for Web Crawling](https://eprints.gla.ac.uk/359292/) (Macdonald et al., Glasgow).
