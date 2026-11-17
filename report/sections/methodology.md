# 7. Methodology

> Report section R-C1 (owner C). Every formula here is meant to match the code it names.
> Items marked **[to confirm]** describe components that are not implemented yet; their owner
> checks the formula against the code when the task lands and removes the marker.

**Notation.** $N$ is the number of phones (documents), $d$ a phone, $q$ a query, $t$ a term,
$\mathrm{tf}_{t,d}$ the count of $t$ in $d$, $\mathrm{df}_t$ the number of phones containing
$t$. $u$ is a reviewer, $r_{u,d} \in \{1,\dots,5\}$ the star rating $u$ gave $d$, $R_d$ the
reviewers of $d$ and $P_u$ the phones reviewed by $u$, with $n_u = |P_u|$. Every ranker
returns at most 100 phones per query, sorted by score, ties broken by `doc_id`.

## 7.1 Text preprocessing (T03, `phonesense/index/analyzer.py`)

Documents and queries pass through the same analyzer, in this order:

1. Unicode NFKC normalisation and lower-casing.
2. Number clean-up (`5,000` becomes `5000`).
3. Unit joining, so that a value and its unit form one token with a canonical spelling:
   `128 GB` becomes `128gb`, `5,000 mAh` becomes `5000mah`, `6.1 inches` becomes `6.1in`.
4. Tokenisation that keeps model codes (`a54`, `s23`), `5g`, decimals and unit tokens whole.
5. Stop-word removal. Variant words that distinguish models (`pro`, `max`, `plus`, `mini`,
   `ultra`, `lite`, `5g`, `4g`) are never removed.
6. Porter stemming. Tokens that contain a digit are never stemmed.

Each kept term also records its position before stop-word removal, so phrase and proximity
search treat documents and queries alike.

## 7.2 Lexical retrieval

### TF-IDF, SMART lnc.ltc (T04, `phonesense/retrieval/tfidf.py`)

Document weights use log term frequency and cosine normalisation, with no idf (lnc):

$$w_{t,d} = \frac{1 + \log_{10}\mathrm{tf}_{t,d}}{\sqrt{\sum_{t'}\left(1 + \log_{10}\mathrm{tf}_{t',d}\right)^2}}$$

Query weights add idf before normalising (ltc):

$$w_{t,q} \propto \left(1 + \log_{10}\mathrm{tf}_{t,q}\right)\log_{10}\frac{N}{\mathrm{df}_t},
\qquad \mathrm{score}(q,d) = \sum_t w_{t,q}\,w_{t,d}$$

Scores are accumulated term-at-a-time over the postings lists, so only phones that share a
term with the query are scored. A three-document worked example is checked in
`tests/test_tfidf.py`.

### BM25 (T05) **[to confirm]**

$$\mathrm{BM25}(q,d) = \sum_{t \in q} \log\frac{N - \mathrm{df}_t + 0.5}{\mathrm{df}_t + 0.5}
\cdot \frac{\mathrm{tf}_{t,d}\,(k_1 + 1)}{\mathrm{tf}_{t,d} + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgdl}}\right)}$$

$k_1$ controls how quickly repeated terms stop adding score, and $b$ how strongly long
listings are penalised. Both are tuned on the dev queries only. The owner confirms the idf
variant used.

### Boolean specification filters (T06) **[to confirm]**

Queries may carry structured constraints such as `battery_mah >= 5000` and `price < 300`
(CONTRACT.md 4.4). Phones that fail a constraint are removed before ranking. The owner states
how a phone with an unknown (`null`) value for a constrained field is handled.

## 7.3 Dense retrieval and query expansion

### Dense retrieval (T12, `phonesense/retrieval/dense.py`)

Each phone's text is embedded once with the sentence encoder
`sentence-transformers/all-MiniLM-L6-v2`, and the unit-length vectors are cached on disk.
A query is embedded the same way and phones are ranked by exact cosine similarity
$\mathrm{score}(q,d) = \mathbf{e}_q \cdot \mathbf{e}_d$; no approximate index is needed for a
collection of a few thousand phones.

### Rocchio pseudo-relevance feedback (T14, `phonesense/retrieval/rocchio.py`)

The query vector moves toward the centroid of phones assumed relevant and away from those
assumed non-relevant:

$$\vec q\,' = \alpha\,\vec q + \beta\,\frac{1}{|D_r|}\sum_{d \in D_r}\vec d
- \gamma\,\frac{1}{|D_{nr}|}\sum_{d \in D_{nr}}\vec d$$

$D_r$ is the top $m = 10$ phones of a first TF-IDF pass. With the default $\gamma = 0$ no
non-relevant set is used; for $\gamma > 0$ the phones ranked $m+1$ to $2m$ form $D_{nr}$.
Defaults are $\alpha = 1$, $\beta = 0.75$. Negative components are dropped and only the 20
heaviest terms of $\vec q\,'$ are kept, to limit topic drift. Phones are then re-scored as
$\sum_t q'_t\,w_{t,d}$ with the lnc document weights.

### HyDE-style LLM expansion (T15) **[to confirm]**

An LLM writes a short hypothetical product description for the query (Gao et al., 2022), and
its terms are weighted Rocchio-style instead of being concatenated (Jedidi and Lin, 2025).
The owner records the model, prompt and temperature.

## 7.4 Rank fusion (T13) **[to confirm]**

Reciprocal Rank Fusion (Cormack et al., 2009) combines systems using ranks only:

$$\mathrm{RRF}(d) = \sum_{s \in S} \frac{1}{k + \mathrm{rank}_s(d)}, \qquad k = 60$$

A phone missing from a system's list contributes nothing for that system. Weighted linear
fusion min-max normalises each system's scores per query and takes
$\sum_s \lambda_s\,\hat x_s(d)$, with the weights tuned on dev queries.
