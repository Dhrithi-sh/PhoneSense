# Riya's Handoff: Regional News Crawler, Corpus, Deduplication & Shared Tooling

**Component:** Track 5 (P1) · Regional News Crawler, Corpus Pipeline, Story Lineage, Adaptive Freshness & Shared Evaluation Tooling  
**Author:** Riya (`ra810@snu.edu.in`) · Branch: `riya` / Merged into: `main`  
**Master Dataset:** `data/news.jsonl` (5,000 articles) · **Sample Milestone:** `data/news_sample_300.jsonl` (300 articles) · **Deduplicated Lineage:** `data/news_dedup.jsonl`  
**Shared Tool:** `dhvani/eval/pool.py` · **Evaluations & Judgments:** `judgments/qrels_riya.txt`, `judgments/pool.tsv` (1,042 candidate pairs), `documentation/needs/riya-needs.md`  

---

## 1. Quick Start & Execution Commands

### 1.1 Generating Corpus Deliverables
```powershell
# 1. Generate H3 Milestone Sample (300 validated articles, ~8 minutes)
python -m dhvani.crawl.crawler --sample

# 2. Run Full Master News Corpus (5,000 validated articles, ~2.2 hours)
python -m dhvani.crawl.crawler --max-articles 5000

# 3. Run Deduplication & Story Clustering over crawled articles
python -m dhvani.crawl.dedup --input data/news.jsonl --output data/news_dedup.jsonl --threshold 0.70

# 4. Generate Publication Figures for Corpus & Deduplication Metrics
python scripts/plot_corpus_metrics.py
```

### 1.2 Running the Full Test Suite
The entire P1 subsystem includes **81 automated unit and integration tests** across 10 test modules:
```powershell
# Run all 81 crawler and tooling tests via pytest (~3.3s)
python -m pytest partwise-tests/riya/ -v

# Or run via the standalone crawler test runner script
python scripts/run_crawler_tests.py
```

---

## 2. Component Deliverables & Repository Locations

| Deliverable | Location | Description |
|---|---|---|
| **Master Corpus** | `data/news.jsonl` *(local / gitignored)* | 5,000 schema-compliant Hindi news articles across 5 primary sources |
| **H3 Sample Corpus** | `data/news_sample_300.jsonl` | 300 clean, validated seed articles for downstream integration |
| **Deduplicated Corpus** | `data/news_dedup.jsonl` *(local / gitignored)* | Clustered articles with `dup_of` lineage links and in-corpus PageRank `links` |
| **Shared TREC Pooler** | `dhvani/eval/pool.py` | Need-level aggregation pooler producing Format 5 judgment sheets |
| **Information Needs** | `documentation/needs/riya-needs.md` | Riya's 8 information needs (`Y01`–`Y08`) with 32 queries (Hindi, Hinglish, messy, English) |
| **Relevance Judgments** | `judgments/qrels_riya.txt` | 240 human-in-the-loop relevance grades (0, 1, 2) covering `Y01`–`Y08` |
| **Master Pool Sheet** | `judgments/pool.tsv` | 1,042 deduplicated `(need_id, doc_id)` candidate pairs across all team needs |
| **Corpus Statistics** | `documentation/results/riya-corpus-results/crawl_stats.md` | Source balance, section distribution, and geographic coverage audit |
| **Dedup Evaluation** | `documentation/results/riya-corpus-results/dedup_stats.md` | Precision/recall calibration table, Jaccard vs MinHash speedup benchmark |
| **Visual Figures** | `documentation/figures/` | Publication plots: `riya-dedup-precision-recall.png`, `riya-corpus-distribution.png`, `riya-recrawl-bandwidth.png` |

---

## 3. Architecture & Design Decisions (ADR)

### 3.1 Setup, Compliance & Filtering (`config.py`, `robots.py`, `normalizer.py`, `filters.py`)
- **Ethical Crawling Contract:**
  - `User-Agent`: Honest identification: `CollegeProject_NewsBot(+https://github.com/InformationRetrieval-Mid/Dhvani-lingual; contact: ra810@snu.edu.in)`.
  - **Politeness Delay:** Minimum strict **8.0 seconds delay** per host enforced between consecutive requests to the same site.
  - **Privacy:** Zero author, editor, or byline names stored anywhere in the dataset.
  - **Local Storage Policy:** News article text stays on local machines (`data/*.jsonl` gitignored); only code, metadata schemas, and URLs go to GitHub.
- **Source Selection & Blacklist:**
  - *5 Primary Sites:* Dainik Jagran (`jagran.com`), Navbharat Times (`navbharattimes.indiatimes.com`), Live Hindustan (`livehindustan.com`), Amar Ujala (`amarujala.com`), Aaj Tak (`aajtak.in`).
  - *Backups:* Jansatta (`jansatta.com`), Dainik Bhaskar (`bhaskar.com`).
  - *Blacklist:* BBC Hindi (robots.txt strictly forbids scraping and dataset creation), News18 and NDTV (aggressive Cloudflare WAF bot challenges).
- **RFC 9309 Robots Parser Decision:**
  - *Empirical Finding:* Python's built-in `urllib.robotparser` uses literal prefix matching and completely crashes on wildcard (`*`) and end-of-path (`$`) directives on *Dainik Jagran* (`/search/*`) and *Amar Ujala* (`/*?utm_*`), returning incorrect `can_fetch == True` for disallowed URLs.
  - *Decision:* Implemented a custom RFC 9309 compliant regex parser (`dhvani/crawl/robots.py`) handling wildcards, longest-prefix rule precedence (longest match wins; ties resolve to `Allow`), in-memory per-host caching, and automated `Sitemap:` directive harvesting.
- **URL Normalization & Route Filtering:**
  - Lowercase scheme and domain; strip tracking query parameters (`utm_*`, `ref`, `fbclid`, `amp_js_v`).
  - Canonical AMP mapping: strips `/amp/` and `?amp=1` paths.
  - Excluded non-article routes: `/rashifal/`, `/astrology/`, `/horoscope/` (prevents keyword pollution), `/photo-gallery/`, `/videos/`, and `/live-updates/` (empty live blog bodies).

---

### 3.2 Mercator Frontier & Concurrency Model (`frontier.py`)
- **Asynchronous Event Loop (`asyncio` + `heapq`):**
  - Uses a single-threaded asynchronous concurrency model. A min-heap tracks `(next_allowed_time, entry_id, host)`.
  - While one request waits on network I/O, the event loop dispatches requests for other eligible hosts without inter-host blocking.
  - Consecutive requests to the same host strictly wait $\ge 8.0\text{s}$.
  - Theoretical dispatch throughput across 5 target hosts: $8.0 / 5 = 1.6\text{ seconds/request}$.
  - Single-threaded execution eliminates thread locks, race conditions, and concurrent file-write conflicts.
- **Two-Tier Queue Architecture:**
  - **4 Front Queues (Priority):** Biased random selector distributes URL sampling:
    - $Q_0$ (Surge / Breaking Bursts): **60% weight**
    - $Q_1$ (Fresh Sitemaps): **25% weight**
    - $Q_2$ (In-Page Hyperlinks): **10% weight**
    - $Q_3$ (Archive Sitemaps): **5% weight**
  - **Per-Host FIFO Back Queues (Politeness):** One queue per domain. Front queues refill empty back queues, but **never preempt or reorder items already inside a host's back queue**, ensuring strict FIFO delivery per host.
  - Bounded back queues (`max_back_queue_size = 10`) eliminate Head-of-Line (HOL) starvation during large burst additions.
- **Error Backoff & WAF Protection:**
  - *HTTP 429 (Rate Limit):* Doubles per-host delay exponentially ($8\text{s} \to 16\text{s} \to 32\text{s} \to 64\text{s}$) and reschedules the host rather than terminating the crawl.
  - *HTTP 403 & CAPTCHAs:* Cloudflare/Akamai challenge signatures immediately quarantine the host for the remainder of the session, safeguarding against IP bans.

---

### 3.3 Content Extraction & Geographic Normalization (`extractor.py`)
- **3-Tier Fallback Extraction Hierarchy:**
  - **Tier 1 (Schema.org JSON-LD):** Primary extraction for `headline`, full `articleBody`, `datePublished`, `keywords`, and `articleSection`.
  - **Tier 2 (Open Graph / Meta Tags):** Secondary fallback for missing metadata. `og:description` is **never** used as a substitute for the full body.
  - **Tier 3 (HTML DOM Block Extraction):**
    - *CMS Compatibility:* Indian news portals (Navbharat Times `<div class="Normal">`, Amar Ujala story containers) frequently wrap paragraphs in `<div>` tags rather than `<p>`.
    - *Double-Counting Guard:* Extracts text from non-nested leaf `<div>` blocks ($\ge 25$ chars) and `<p>` tags, stripping boilerplate (ads, related boxes, image captions, social shares) without repeating identical text from parent wrapper `<div>`s.
    - *Embedded CMS JSON Filtering:* Filters out raw CMS metadata blobs (e.g. `{"_id": "...", "slug": "..."}`) embedded inside content wrappers.
- **Strict IST Date Normalization:** All published timestamps parsed and converted to ISO-8601 with `+05:30` offset.
- **District Bureau Geographic Expansion & Delhi Normalization:**
  - Expanded `INDIAN_CITIES` from 47 tier-1 cities to cover **120+ prominent district centers** across the Hindi belt (*Basti, Ratlam, Kangra, Sikar, Pithoragarh, Alwar, Dehradun*).
  - Portal slug recognizers: Jagran's `/<state>/<city>-city-...` conventions.
  - **Deterministic State Derivation:** Every city is deterministically mapped to its parent state (`CITY_TO_STATE`), guaranteeing no article has a valid `city` with a `null` state. Articles tagged with `state: "delhi"` automatically default to `city: "delhi"`.
  - Unidentifiable national or policy articles are conservatively set to `null` rather than guessing.

---

### 3.4 Near-Deduplication & Story Clustering (`dedup.py`)
- **Shingling & Similarity Formulation:**
  - 4-word sliding shingles extracted over Devanagari word sequences.
  - Exact Jaccard similarity:
    $$J(A, B) = \frac{|S_A \cap S_B|}{|S_A \cup S_B|}$$
  - Candidate comparison restricted to a **$\pm 24$-hour temporal window** ($|t_A - t_B| \le 86,400\text{s}$) to reflect news cycle relevance.
- **MinHash + LSH Scalable Clustering:**
  - 128 permutation hash functions ($h_i(x) = (a_i \cdot x + b_i) \pmod p$).
  - Locality-Sensitive Hashing: partitioned into $b = 16$ bands of $r = 8$ rows ($b \cdot r = 128$).
  - Theoretical S-curve detection threshold:
    $$t \approx \left(\frac{1}{b}\right)^{1/r} = \left(\frac{1}{16}\right)^{1/8} \approx 0.707$$
  - Provides $O(N)$ candidate pair generation vs $O(N^2)$ all-pairs comparison ($1.68\times$ throughput speedup on the full corpus).
- **Threshold Calibration & Ground Truth Results:**
  - Benchmarked on 100 manually labeled Hindi news pairs across thresholds $J \in [0.60, 0.80]$:
    - $J = 0.65$: Precision 0.885, Recall 0.920, F1 0.902
    - **$J = 0.70$ (Optimal Operational Threshold):** **Precision = 0.947**, **Recall = 0.900**, **F1 = 0.923**
    - $J = 0.75$: Precision 0.971, Recall 0.680, F1 0.800
  - *Full Corpus Yield:* Discovered 16 near-duplicate clusters (33 total articles, 17 secondary reprints linked via `dup_of`).
- **Story Lineage Contract:**
  - Earliest published article is designated the canonical root (`dup_of: null`).
  - Later wire stories and reprinted versions link to the root (`dup_of: "jagran_..."`).
  - In-corpus hyperlink extraction populates `links` for PageRank graph building.

---

### 3.5 Adaptive Recrawling, Freshness & Surge Priority (`recrawl.py`)
- **Exponential Weighted Moving Average (EWMA) Arrival Velocity:**
  - Velocity tracked per news portal using newly published URLs ($\Delta N$):
    $$\lambda_s^{(t)} = (1 - \alpha) \cdot \lambda_s^{(t-1)} + \alpha \cdot \frac{\Delta N}{\Delta t} \quad (\alpha = 0.3)$$
  - *Newly Observed Invariant:* $\Delta N$ uses only newly discovered URLs, ignoring historical XML entries to prevent archived sitemaps from falsely inflating live velocity.
- **Adaptive Polling Interval Formulation:**
  $$\tau_s = \max\left(\tau_{\min},\; \min\left(\tau_{\max},\; \frac{K}{\max(\lambda_s, 0.0833)}\right)\right)$$
  - $K = 1800\text{s}$ (30 min).
  - Nominal interval ($\lambda_s = 1.0\text{ URL/h}$) scales at $\tau = 1800\text{s}$, scaling up to $\tau_{\max} = 21600\text{s}$ (6 hours) during idle periods. Clamped to $\tau_{\min} = 1800\text{s}$.
- **HTTP 304 Conditional Bandwidth Optimization:**
  - Injects `If-None-Match` (ETag) and `If-Modified-Since` headers into sitemap requests.
  - On `304 Not Modified`, $\Delta N = 0$ URLs are processed without downloading XML payloads.
  - Delivers **12.5% bandwidth savings** across continuous re-crawl operations.
- **Topical Burst & Surge Prioritization Engine:**
  - Decoupled from source velocity; tracks publication volume per section across all sources using rolling deques.
  - Acute 1-hour window ($W_{\text{acute}} = 3600\text{s}$) vs baseline 6-hour moving average:
    $$\text{BurstScore} = \frac{\text{Count}_{1\text{h}}}{\text{MovingAvg}_{6\text{h}} + 1.0}$$
  - When $\text{BurstScore} > 2.0$, section enters `BURST` state; candidate URLs route into **Front Queue $Q_0$** (60% biased sampling weight).
  - Event priority decays automatically as breaking news subsides.

---

### 3.6 Shared Evaluation Tooling & TREC Pooling (`dhvani/eval/pool.py`)
- **Need-Level Pooling Architecture (`per_need=True` by default):**
  - *Problem:* System run files output `qid Q0 doc_id rank score run_name` where `qid` contains linguistic form tags (`R01_hi`, `R01_hinglish`, `R01_messy`, `R01_en`). Raw `qid` pooling split candidate pools into 4 disjoint sets per story, producing empty `qrels` lookups during evaluation and quadrupling manual judging effort.
  - *Fix:* By default, `pool_runs()` aggregates all surface forms into a unified candidate pool keyed by `need_id` (`R01`, `V01`, `Y01`), strictly fulfilling `documentation/formats.md` Format 5 (`need_id 0 doc_id rel`).
  - *Resolution Hierarchy:*
    1. `--queries queries.tsv` lookup table (highest precedence).
    2. Fallback prefix parsing on first underscore (`qid.split("_")[0]`).
    3. Passthrough for queries without underscores (e.g. `N01`).
  - Cross-form duplicate retrievals are deduplicated within each need's pool.
  - `--raw-qid` flag preserved for exceptional cases requiring per-surface-form pooling.

---

## 4. Downstream Integration Guidance (Using Dependencies Correctly)

### 4.1 Consuming the Corpus (`data/news.jsonl` / `data/news_sample_300.jsonl`)
Downstream components must adhere to the following field expectations:
```json
{
  "doc_id": "jagran_40396681",
  "url": "https://www.jagran.com/uttar-pradesh/lucknow-weather-40396681.html",
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
  "links": ["jagran_40395001"]
}
```

#### Guidelines by Component:
1. **For Text Processing & Indexing (Dhrithi / P2):**
   - **Fields to Index:** Index terms from both `headline` and `body`. In positional indexes, track headline positions and body positions separately to support zone scoring.
   - **Handling Duplicates:** When building master indexes, check `dup_of`. Canonical cluster heads have `dup_of == null`. Articles where `dup_of != null` can be indexed normally, but downstream rankers can use `dup_of` to collapse duplicate listings in search results.
   - **Date Invariant:** `date` is guaranteed ISO-8601 in IST offset (`+05:30`). Parsers do not need complex fallback guessing.
2. **For Hinglish & Query Layer (Viraja / P3):**
   - **Vocabulary Extraction:** The full corpus contains 61,541 unique Devanagari terms across 5,000 articles. Use this vocabulary to generate k-gram indexes and train Soundex / Dhvani-code transliteration alignments.
   - **Named Entities:** Regional Hindi-belt cities in `city` and `state` provide protected place names that should not be aggressively stemmed.
3. **For Ranking, Authority & App (Rishit / P4):**
   - **PageRank Graph (`links`):** The `links` field contains validated in-corpus `doc_id` references found within each article's body. Use these directed edges directly to compute static PageRank scores for authority $g(d)$.
   - **First-to-Publish & Wire Credit (`agency_flag`, `dup_of`):** Articles with `agency_flag: true` represent syndicated wire news. When collapsing duplicate stories, the cluster head (`dup_of == null`) published earliest represents the primary breaking source.
   - **Date-Aware "Kal":** Anchor recency scores to `date` against the latest article timestamp in the index rather than the real-time system clock.
   - **Metadata Filtering:** Use `source`, `section`, `state`, and `city` for faceted filtering.

---

### 4.2 Critical Guidance Regarding Sitemaps
> [!IMPORTANT]
> **Do NOT attempt to fetch full sitemaps directly in live search or app requests.**
> Sitemaps are heavy XML feeds containing 500 to 5,000 URLs per file. Ingesting sitemaps belongs strictly to offline crawler frontier discovery.

#### Rules for Downstream Modules Interacting with Sitemaps / Crawler Data:
1. **Always Observe Politeness (8.0s per host):** Any background script that interacts with external target sites must respect the 8.0s politeness delay and route through `RobotsParser`. Never issue unthrottled requests.
2. **Conditional Headers on Polling:** If polling sitemaps for updates, always store and send `If-None-Match` and `If-Modified-Since` headers to handle HTTP 304 responses, preventing bandwidth waste and avoiding rate-limit bans.
3. **Namespace-Agnostic XML Parsing:** Target news portals mix XML namespaces (`xmlns`, `xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"`). Downstream scripts parsing sitemaps should use tag-suffix matching (`elem.tag.endswith("news")`) rather than hardcoded XML namespace URIs.
4. **Publication Date Extraction:** In Google News sitemaps, the publication timestamp lives in `<news:publication_date>`. In standard sitemaps, `<lastmod>` represents the latest modification time, not necessarily the original article creation date.

---

### 4.3 Using the Shared Evaluation Tooling (`dhvani/eval/pool.py`)
- **Run File Contract:** Downstream experiment runs must emit lines in standard TREC format:
  ```text
  qid Q0 doc_id rank score run_name
  ```
- **Pooling Command:**
  ```powershell
  python -m dhvani.eval.pool --runs data/eval/full/runs --k 10 --out judgments/pool.tsv
  ```
- **Aggregation Behavior:** By default, `pool.py` automatically merges `R01_hi`, `R01_hinglish`, `R01_messy`, `R01_en` into `R01`, deduplicating candidates across runs into a single judging sheet.
- **Integration with `qrels_<person>.txt`:** Downstream scoring scripts (`experiments.py`, `metrics.py`) look up judgments by `need_id` (e.g. `R01`, `V01`, `Y01`). Pooling by need ensures evaluation matches `qrels.txt` without missing-key evaluation failures.

---

## 5. Information Needs & Relevance Judgments

### 5.1 Riya's 8 Information Needs (`documentation/needs/riya-needs.md`)
Riya's 8 needs are designated **`Y01` – `Y08`** (matching `PREFIX["riya"] = "Y"` in `app/pages/judge.py`):

| Need | Description | Focus Area / Project Purpose |
|---|---|---|
| **Y01** | Heavy rain alert and temperature drop in UP / Lucknow | Regional weather alerts & district geographic filtering |
| **Y02** | Gold and silver prices hit record highs ahead of Diwali/Dhanteras | Business commodity news & cross-paper pricing reports |
| **Y03** | Indian Railways announces festival special trains for Diwali and Chhath | Transportation crowd management & multi-source coverage |
| **Y04** | Supreme Court hearing and guidelines on bulldozer action | Legal governance & national policy duplicate clustering |
| **Y05** | India vs Bangladesh T20 series team performance and match results | High-frequency sports reporting & player entity matching |
| **Y06** | Israel airstrikes in Lebanon and Middle East conflict escalation | International breaking news & acute surge burst testing |
| **Y07** | UP CM Yogi Adityanath law and order & development review | State governance, administrative accountability & UP filters |
| **Y08** | Central Government Dearness Allowance (DA) hike for employees | National economic policy & syndicated wire reprint matching |

Each need is formulated in 4 linguistic variations (32 queries total) in `documentation/needs/riya-needs.md`.

### 5.2 Relevance Judgments Summary (`judgments/qrels_riya.txt`)
- **Candidate Pool:** 240 candidate articles (30 per need) pooled into `judgments/pool.tsv` (pool total: 1,042 pairs).
- **Human-in-the-Loop Evaluation:** All 240 articles were individually judged on the 3-point relevance scale:
  - **Grade 2 (Fully Relevant):** 92 articles
  - **Grade 1 (Partly Relevant):** 122 articles
  - **Grade 0 (Not Relevant):** 26 articles
- **Status:** **100% complete** (`240 of 240 judged`). Fully ingested by `dhvani.eval.judge` and the Streamlit judging UI.

---

## 6. Test Suite & Verification Matrix

All 81 tests pass in ~3.3 seconds:

| Test Module | Coverage | Status |
|---|---|:---:|
| `test_robots.py` | RFC 9309 rules, wildcard matching, precedence, caching, fallback | **Passed** (8/8) |
| `test_frontier.py` | 8.0s politeness delay, min-heap ordering, biased front queue weights | **Passed** (7/7) |
| `test_sitemap.py` | Standard, Google News, index, archive feeds, conditional HTTP headers | **Passed** (7/7) |
| `test_normalizer.py` | URL cleaning, tracking query stripping, AMP mapping, route filters | **Passed** (9/9) |
| `test_extractor.py` | JSON-LD, meta fallback, leaf-div DOM extraction, IST dates, zero authors | **Passed** (8/8) |
| `test_crawler.py` | Full crawl loop, HTTP 429 backoff, HTTP 403 host quarantine, sample mode | **Passed** (9/9) |
| `test_dedup.py` | 4-word shingles, Jaccard similarity, MinHash LSH, 100 labeled pairs benchmark | **Passed** (11/11) |
| `test_recrawl.py` | EWMA velocity smoothing, adaptive interval bounds, HTTP 304, burst scoring | **Passed** (11/11) |
| `test_format_compliance.py` | Contract validation across 300-article sample and 5,000 master corpus | **Passed** (3/3) |
| `test_pool.py` | Need-level aggregation, cross-form deduplication, `--raw-qid`, format 5 | **Passed** (8/8) |
| **Total** | **All 10 Subsystems Verified** | **81 / 81 Passed** |
