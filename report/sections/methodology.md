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
