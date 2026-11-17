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

## 7.5 Phone graph (T16, `phonesense/graph/build_graph.py`)

Phones are nodes. Two edge types are built and merged; when both give the same directed pair,
their weights are added.

* **Co-review.** Phones $a$ and $b$ are linked in both directions when at least
  $s_{\min} = 2$ reviewers rated both. With $s_{ab}$ shared reviewers the weight is
  $w_{ab} = \min(s_{ab}, 5)/5$. Reviewers who rated more than 30 phones are skipped: they
  add a quadratic number of pairs and little signal.
* **Semantic kNN.** Each phone points to its $k = 8$ most similar phones by cosine similarity
  of the dense embeddings (7.3), with that similarity as the weight. Non-positive
  similarities are dropped.

Co-purchase links ("bought together") are not used: the phone file does not carry them.
For the report we record, per edge type, the number of edges and the share of phones
touched, and for the merged graph the dangling phones (no out-edges), the isolated phones
(no edges) and the size of the largest connected component.

## 7.6 PageRank and personalized PageRank (T17, `phonesense/graph/pagerank.py`)

Let $\pi$ be the rank vector, $d$ the damping factor ($d = 1 - \epsilon$ in the lecture's
notation, default 0.85), $\mathbf t$ the teleport distribution and
$W_u = \sum_v w_{uv}$ the out-weight of $u$. One power-iteration step is

$$\pi^{(i+1)}_v = d\left(\sum_{u \to v} \pi^{(i)}_u\,\frac{w_{uv}}{W_u}
+ t_v \sum_{u:\,W_u = 0} \pi^{(i)}_u\right) + (1 - d)\,t_v$$

The second term handles dangling phones: rank that would leak out of a phone with no
out-edges is redistributed according to $\mathbf t$, which keeps $\sum_v \pi_v = 1$. We stop
when $\lVert \pi^{(i+1)} - \pi^{(i)} \rVert_1 < 10^{-8}$. Because the error shrinks by at
least a factor $d$ per step, this takes at most $\lceil \log 10^{-8} / \log 0.85 \rceil = 114$
iterations.

* **Global PageRank** uses the uniform teleport $t_v = 1/N$. It is a query-independent
  "how central is this phone" score. Because PageRank is heavy-tailed, it is turned into a
  prior by its percentile rank: $\mathrm{pr}(d) = |\{d' : \pi_{d'} \le \pi_d\}| / N$.
* **Personalized PageRank** teleports only to seed phones. For a query, the seeds are its top
  10 results with weight $1/\mathrm{rank}$, so rank flows to phones linked to the best hits
  (the HippoRAG idea, Gutiérrez et al., 2024). Phones unreachable from the seeds score 0.

The implementation is our own (two sparse weighted sums per step with `numpy.bincount`);
tests check it against `networkx.pagerank` to within $10^{-9}$, for both variants.

**Algorithm 1: PageRank by power iteration**
```
input:  edges (u, v, w), teleport t (sums to 1), damping d, tolerance tol
W[u]  <- sum of w over edges leaving u
pi    <- t
repeat
    flow[v]  <- sum over edges (u, v, w) of pi[u] * w / W[u]
    leak     <- sum of pi[u] over u with W[u] = 0
    new[v]   <- d * (flow[v] + leak * t[v]) + (1 - d) * t[v]
    err      <- sum over v of |new[v] - pi[v]|
    pi       <- new
until err < tol
return pi
```

## 7.7 Reputation scores (T19, `phonesense/trust/`)

### Baselines

With $m$ the global mean rating and $n_d = |R_d|$:

| Method | Phone score |
|---|---|
| Raw average | $\frac{1}{n_d}\sum_{u \in R_d} r_{u,d}$ |
| Bayesian average | $\dfrac{C\,m + \sum_{u \in R_d} r_{u,d}}{C + n_d}$, with $C = 10$ |
| Helpful-vote weighted | weighted mean of $r_{u,d}$ with weight $1 + \ln(1 + h_{u,d})$, $h$ = helpful votes |
| Median | median of $\{r_{u,d}\}$ |

### Reviewer-trust score (our design, inspired by HITS)

HITS scores hubs by the authorities they point to and authorities by the hubs pointing to
them. We apply the same mutual reinforcement to the reviewer-phone bipartite graph: a phone's
quality depends on how reliable its reviewers are, and a reviewer's reliability depends on
how well their ratings agree with the quality of the phones they rated.

$$q_d = \frac{C\,m + \sum_{u \in R_d} \rho_u\,r_{u,d}}{C + \sum_{u \in R_d} \rho_u}$$

$$a_u = 1 - \frac{1}{n_u}\sum_{d \in P_u}\frac{|r_{u,d} - q_d|}{4}, \qquad
\rho_u = \frac{n_u\,a_u + c\,\rho_0}{n_u + c}$$

$q_d$ is the phone's quality in stars, $a_u \in [0, 1]$ the reviewer's agreement (4 is the
largest possible star difference) and $\rho_u \in [0, 1]$ their reliability. Two priors keep
the scores stable:

* $C = 10$ pulls phones with few reviews toward the global mean, exactly as in the Bayesian
  average.
* $c = 3$ and $\rho_0 = 0.5$ pull reviewers with few reviews toward a neutral reliability. A
  brand-new account with one review therefore lands near 0.5, never at 0 or 1; without this
  prior a fake account with a single review that happens to match the consensus would look
  perfectly reliable.

Differences from HITS: we use weighted averages and agreement instead of sums, and the priors
replace HITS's per-round normalisation (both scores are already bounded). Starting from
$\rho_u = 1$, the first $q_d$ equals the Bayesian average, so the method can only depart from
that baseline where reviewers disagree. We iterate until
$\max_d |q_d^{(i+1)} - q_d^{(i)}| < 10^{-6}$, capped at 50 rounds. Convergence is observed
(6 rounds on synthetic data), not proven; the iteration count is reported with every run.

For ranking, the prior is $(q_d - 1)/4 \in [0, 1]$. The reviewer reliabilities $\rho_u$ are
written to `reviewer_trust.tsv`.

**Algorithm 2: reviewer trust**
```
input:  reviews (u, d, r), priors C, c, rho0, tolerance tol, max_rounds
m       <- mean of all r;  rho[u] <- 1 for every reviewer
repeat up to max_rounds
    q[d]    <- (C*m + sum of rho[u]*r over reviews of d) / (C + sum of rho[u] over reviews of d)
    a[u]    <- 1 - mean over reviews of u of |r - q[d]| / 4
    rho[u]  <- (n[u]*a[u] + c*rho0) / (n[u] + c)
until the largest change in q is below tol
return q, rho
```

## 7.8 Reranking with a prior (T18, `phonesense/graph/rerank.py`)

Any run (TF-IDF, BM25, dense, fused) can be reranked with a prior $p(d) \in [0, 1]$:
global PageRank (7.6), personalized PageRank (7.6) or reviewer trust (7.7). Within each query
the run's scores are min-max normalised to $\hat s(d) \in [0, 1]$ and mixed:

$$s'(d) = (1 - \lambda)\,\hat s(d) + \lambda\,p(d)$$

$\lambda = 0$ keeps the original ranking and $\lambda = 1$ ranks by the prior alone. Only
phones already in the run are reordered. A phone the prior does not cover gets the prior's
median, so it is neither pushed up nor down. For personalized PageRank the prior is computed
per query and scaled by its largest value among that query's candidates. $\lambda$ is tuned
on dev queries from $\{0.1, 0.2, 0.3, 0.5\}$ and the chosen value is applied to the test
queries once.

We mix rather than multiply ($s \cdot (1 + \lambda p)$) because retrieval scores can be
negative (cosine similarity) and a multiplicative prior can never lift the lowest-scored
phone.

## 7.9 Review-attack simulation (T20, `phonesense/attacks/simulator.py`)

Following the attacks on reputation systems from Lecture 9, an attacker creates new accounts
(sybils) and posts fake reviews on one target phone:

* **Bad-mouthing:** one-star reviews on a good phone, as a competitor would. Targets are
  sampled from the top quarter of phones by raw average.
* **Ballot stuffing:** five-star reviews on a weak phone, as its seller would. Targets are
  sampled from the bottom quarter.

Only phones with at least 20 reviews are targets. Each sybil posts exactly one attack review,
marked unverified with no helpful votes. The attack size is the number of fake reviews as a
percentage of the target's real review count, from $\{0, 1, 5, 10, 20, 50, 100\}\%$
(rounded up), with 15 targets per attack type and 5 random seeds.

**Camouflaged attacker.** In a second setting each sybil first posts $k = 5$ cover reviews on
random other phones, each equal to that phone's rounded average rating, so that it agrees
with the consensus and gains reliability before it attacks. This is the attacker the trust
score is expected to struggle against.

For each reputation method we record the target's score and its rank among all phones
before and after the attack. The rank shift is signed so that a positive value always means
the attacker gained. Fake reviews are flagged `is_synthetic` and use `SYN_` account ids, and
the data loader refuses them in the real data file, so they cannot leak into other results.

**Algorithm 3: attack experiment**
```
for kind in {bad_mouth, ballot_stuff}, seed in seeds:
    for target in pick_targets(kind, seed):
        for pct in sizes:
            n        <- ceil(pct/100 * reviews(target))
            attacked <- reviews + n sybil reviews (rating 1 or 5) [+ cover reviews if camouflaged]
            for method in {raw, Bayesian, helpful, median, trust}:
                record rank and score of target, before and after
```
