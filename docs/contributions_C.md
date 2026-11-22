# Detailed contributions: Rishit Kamboj (Member C)

CSD358 Information Retrieval · PhoneSense · GitHub: `rishitkam` · Repository: `Dhrithi-sh/PhoneSense`

This document describes my part of PhoneSense in detail: what I was responsible for, how each
component works, the decisions I made along the way, the results, and where everything lives
in the repository. A shorter version is in [CONTRIBUTIONS.md](../CONTRIBUTIONS.md).

---

## 1. Overview

PhoneSense is a phone search engine built on the Amazon Reviews 2023 data (7,166 phones,
575,816 reviews). The project has three layers: classic IR, neural IR, and trust-aware graph
ranking. I was responsible for the third layer:

- a **phone graph** built from the review data, and **PageRank / personalized PageRank** on it;
- a **reviewer-trust model**, inspired by HITS, plus four reputation baselines;
- a **fake-review attack simulator** and the experiments that measure how each reputation
  method holds up against it;
- **reranking** of search results with these signals, evaluated against the team's tuned
  BM25;
- the **attack tab** in the demo.

In the report I wrote the **Methodology** for the whole system, the **graph, trust and attack
experiments (8.5)**, the **Conclusions and Limitations**, and the **References**.

| | |
|---|---|
| Planned share | 46 work units (33.5% of the project), 12 tasks |
| Completed | 11 of 12 tasks merged; the final report pass starts once the PDF is assembled |
| Pull requests | 12 opened and merged (#7, #8, #9, #10, #15, #16, #17, #18, #27, #28, #29, #30) |
| Team integration | Reviewed #3 and #4; tested together and merged #20 to #26 |
| Commits on `main` | 60 |
| Code | about 1,000 lines in `phonesense/`, 350 in scripts, 700 in tests |
| Tests | 61 test functions in 8 files; full project suite 663 passing |
| Report | about 700 lines in 4 sections, with 6 tables and 2 figures in Experiments 8.5 |

---

## 2. What I was assigned, and what happened to each task

| Task | Planned WU | Status | PR |
|---|---:|---|---|
| T17 PageRank (power iteration, dangling nodes) and personalized PageRank | 6 | Done | #7 |
| T19 Reviewer trust (HITS-inspired) and reputation baselines | 6 | Done | #7 |
| T16 Phone graph and edge statistics | 4 | Done; co-review graph in place of "bought together" (see 3.1) | #7, #28 |
| T20 Attack simulator (bad-mouthing, ballot stuffing, seeds, non-mutating) | 5 | Done, plus camouflaged attackers | #7 |
| T18 Reranking with the priors and λ sweep | 3 | Done; λ tuned on dev, test run once | #7, #10 |
| T21 Attack experiments | 4 | Done; rank and star shift per method | #7, #9 |
| L-C Hand-labelling share (for the LLM-judge agreement check) | 1.5 | Done (one third of the test sample) | via #6 |
| T25 Attack tab for the demo | 3 | Done | #7 |
| R-C1 Report: Methodology | 6 | Done | #8 |
| R-C2 Report: Experiments, graph and attack part | 3 | Done, extended with two follow-up studies | #9, #10, #28, #29 |
| R-C3 Report: Conclusions, Limitations, References | 3 | Done | #9, #15 |
| R-C4 Final pass over the whole report | 1.5 | Waiting for the assembled PDF | – |

Work I did beyond these tasks is described in sections 3.7, 3.8 and 5.

---

## 3. Components in detail

### 3.1 The phone graph (T16)

**Goal.** Link phones that are related, so that PageRank can use the structure of the
collection.

**The problem I had to solve first.** The plan assumed Amazon's "bought together" links, but the
2023 phone metadata does not contain them. Rather than drop the graph, I designed one from
what the data does have, the reviews.

**Design.**
- **Co-review edges.** Two phones are linked when at least 2 of the same reviewers rated both.
  With s shared reviewers the edge weight is min(s, 5)/5, in both directions.
- **Power users.** Reviewers who rated more than 30 phones are skipped. Otherwise a few very
  active accounts would connect large parts of the catalogue for reasons that have nothing to
  do with the phones.
- **Semantic kNN edges** (optional). Each phone is linked to its k nearest neighbours by cosine
  similarity of the dense embeddings, weighted by that similarity.
- **Output.** `data/processed/graph_edges.tsv` in the team's shared format (documented in
  `CONTRACT.md` 4.8), plus statistics: coverage, isolated phones, components and degrees.

**Results on the full data.**
- Co-review: 15,323 linked pairs (30,646 directed edges). 2,757 phones (38.5%) have at least
  one edge, the largest component has 2,636 phones, and out-degree has median 3 and maximum 625.
- The graph is sparse because 88.4% of reviewers wrote only one review.
- Co-review + kNN (k = 8): 85,883 edges, no isolated phones, one component of all 7,166 phones.

**Files.** `phonesense/graph/build_graph.py`, `scripts/build_graph.py`, `tests/test_build_graph.py`
(9 tests: builders, statistics and a file round trip).

### 3.2 PageRank and personalized PageRank (T17)

**Implementation.** My own power iteration, not a library call. One step is:

  π⁽ⁱ⁺¹⁾(v) = d · ( Σ_{u→v} π⁽ⁱ⁾(u) · w_uv / W_u + t_v · Σ_{u: W_u = 0} π⁽ⁱ⁾(u) ) + (1 − d) · t_v

- W_u is the total out-weight of u, and t is the teleport vector (uniform for PageRank, or
  focused on chosen phones for personalized PageRank).
- **Dangling nodes.** Rank sitting on phones with no out-edges is redistributed by t, which
  keeps the scores summing to 1. Isolated phones would otherwise make a naive implementation
  leak rank.
- **Settings.** Damping d = 0.85; iteration stops when the L1 change falls below 1e-8. Each
  step is two sparse weighted sums using `numpy.bincount`.
- **Scaling.** For use as a ranking prior, scores are converted to percentiles in [0, 1].
  Raw PageRank values are tiny and skewed, so mixing them directly with retrieval scores
  would not work.

**Verification.** 14 tests cover hand-computed graphs, dangling nodes and personalization, and
compare against `networkx.pagerank` to within 1e-9 for both plain and personalized PageRank.

**Results.** It converges in 74 iterations (0.014 s) on the co-review graph. The highest-ranked
phones are long-selling, widely reviewed models: the Samsung Galaxy S3 Mini (5,437 ratings),
the BLU Advance 5.5 HD (3,960) and the second-generation Moto G (2,370). This confirms that
PageRank here acts as a popularity and centrality signal.

**Files.** `phonesense/graph/pagerank.py`, `tests/test_pagerank.py`.

### 3.3 Reputation baselines and reviewer trust (T19)

**Baselines** (`phonesense/trust/baselines.py`):
- raw average;
- Bayesian average (shrunk toward the global mean with weight C = 10);
- helpful-vote weighted average (weight 1 + ln(1 + helpful votes));
- median rating.

**Trust model** (`phonesense/trust/trust.py`). Phone quality and reviewer reliability depend on
each other, as hubs and authorities do in HITS:

  q_d = (C·m + Σ_u ρ_u · r_ud) / (C + Σ_u ρ_u)
  a_u = 1 − (1/n_u) Σ_d |r_ud − q_d| / 4
  ρ_u = (n_u · a_u + c · ρ₀) / (n_u + c)

Here q_d is a phone's quality in stars, a_u is how closely a reviewer agrees with the
consensus, and ρ_u is their reliability. Parameters: C = 10, c = 3, ρ₀ = 0.5; iteration stops
when no quality changes by more than 1e-6 (at most 50 rounds).

**Design points.**
- **Starting point.** All reliabilities start at 1, so the first estimate of q is exactly the
  Bayesian average. The model can only move away from that baseline if reviewer agreement
  gives it a reason to, which makes it easy to reason about and test.
- **Smoothing.** The reliability prior (c, ρ₀) matters because 88.4% of reviewers have one
  review. Without it, a single review matching the consensus would make an account look
  perfectly reliable, which is exactly what an attacker would exploit.
- **Differences from HITS.** It uses weighted averages and agreement rather than sums, and has
  priors.

**Results on the full data.**
- It converges in 9 iterations.
- The median reliability is 0.56, and 90% of reviewers fall between 0.44 and 0.62.
- Trust-weighted quality differs from the Bayesian average by 0.091 stars on average (at most
  0.43 stars).
- The information comes from the 56,969 repeat reviewers, who wrote 24.8% of all reviews.

**Files.** `phonesense/trust/` (`baselines.py`, `trust.py`, `methods.py`, `toy.py`),
`scripts/build_trust.py`, `tests/test_trust_baselines.py` (4 tests) and `tests/test_trust.py`
(8 tests, including a bad-mouthing case on toy data).

### 3.4 Fake-review attack simulator (T20)

**What it simulates.**
- **Bad-mouthing:** fake 1-star reviews on a well-rated phone, as a competitor would post.
- **Ballot stuffing:** fake 5-star reviews on a poorly rated phone.
- **Attackers:** new accounts, unverified and with no helpful votes. Fake reviews are flagged
  `is_synthetic` and use `SYN_` account ids, so the shared data reader keeps them out of the
  real pipeline.
- **Camouflage:** before attacking, each fake account posts k reviews that copy the consensus
  rating of other phones, to earn a good reliability. This is the attacker that agreement-based
  reputation systems are known to be most exposed to.

**Properties.** Seeded and reproducible; it never modifies the real data. Helper functions
compute the attack size (as a percentage of the target's real reviews), pick targets (well-rated
phones for bad-mouthing, poorly rated ones for ballot stuffing, at least 20 reviews each), and
measure the shift in rank and stars.

**Files.** `phonesense/attacks/simulator.py`, `phonesense/attacks/report.py`,
`tests/test_attacks.py` (11 tests) and `tests/test_attack_report.py` (4 tests).

### 3.5 Attack experiments (T21)

**Setup.**
- For each attack type, 15 targets were drawn with each of 3 seeds: 45 targets per type and 90
  in total.
- Each target was attacked with fake reviews equal to 0, 1, 5, 10, 20, 50 and 100% of its real
  review count.
- I recorded its score and its rank among all 7,166 phones under each of the five methods,
  then repeated everything with camouflaged attackers.

**Bad-mouthing** (Table 8.5a): mean places the target dropped, with stars lost in brackets.

| Attack size | Raw | Bayesian | Helpful-vote | Median | Trust |
|---:|---:|---:|---:|---:|---:|
| 1% | 238 (0.07) | 225 (0.05) | 151 (0.05) | 0 (0.00) | 160 (0.04) |
| 5% | 644 (0.18) | 662 (0.15) | 418 (0.13) | 143 (0.03) | 459 (0.11) |
| 10% | 1,237 (0.32) | 1,351 (0.27) | 800 (0.22) | 305 (0.08) | 999 (0.21) |
| 20% | 2,384 (0.56) | 2,824 (0.47) | 1,591 (0.40) | 696 (0.21) | 2,395 (0.38) |
| 50% | 4,515 (1.09) | 5,513 (0.95) | 3,511 (0.83) | 2,682 (0.94) | 5,682 (0.85) |

**Ballot stuffing** (Table 8.5b): mean places the target rose, with stars gained in brackets.

| Attack size | Raw | Bayesian | Helpful-vote | Median | Trust |
|---:|---:|---:|---:|---:|---:|
| 1% | 109 (0.05) | 101 (0.04) | 65 (0.03) | 264 (0.14) | 97 (0.03) |
| 5% | 282 (0.12) | 275 (0.09) | 175 (0.07) | 561 (0.37) | 266 (0.08) |
| 10% | 516 (0.20) | 512 (0.15) | 314 (0.13) | 891 (0.58) | 513 (0.14) |
| 20% | 988 (0.35) | 1,043 (0.28) | 609 (0.23) | 1,359 (0.83) | 1,125 (0.26) |
| 50% | 2,322 (0.67) | 2,719 (0.55) | 1,494 (0.47) | 3,770 (1.76) | 3,161 (0.53) |

**Camouflaged attackers** (Table 8.5c, 5 cover reviews per account): places gained, raw
average against trust.

| Attack size | Bad-mouthing raw | Bad-mouthing trust | Ballot stuffing raw | Ballot stuffing trust |
|---:|---:|---:|---:|---:|
| 5% | 644 | 830 | 282 | 386 |
| 10% | 1,237 | 1,851 | 517 | 762 |
| 20% | 2,386 | 4,039 | 988 | 1,668 |
| 50% | 4,512 | 6,353 | 2,321 | 4,150 |

**What I concluded.**
- **Stars.** Trust reduces the change in a phone's stars. Against plain fake accounts it lost
  fewer stars than the raw average for every bad-mouthed target up to 20%, and for 82 to 87%
  of ballot-stuffing targets.
- **Rank.** For small bad-mouthing attacks trust also protects the rank: it kept the target
  higher for 91% of targets at 5% and 76% at 10%. It is level at 20% and worse at 50%.
- **Why the rank advantage fades.** I traced it to score compression. Trust, like the Bayesian
  average, pulls phones toward the mean, so its middle 50% of phones span only 0.40 stars
  against 0.81 for the raw average. A smaller loss in stars then crosses more phones.
- **The simple baselines behave in opposite ways.** The median ignores small bad-mouthing
  completely but is the easiest to game by ballot stuffing. Helpful-vote weighting is the most
  consistent, partly because fake reviews never receive votes in the simulator, which I listed
  as a limitation.
- **Camouflage defeats agreement-based trust.** The target moves 1.3 to 1.8 times as far as
  under the raw average. This points to the improvement a real system would need: judging
  reliability by account history, not agreement alone.

**Files.** `scripts/run_attacks.py`, `report/figures/attack_rank_shift.png` and
`attack_rank_shift_camo5.png`.

### 3.6 Reranking search results with the priors (T18)

**Design.**
- **Prior reranker.** It normalises a run's scores to [0, 1] per query and mixes in a prior:
  s' = (1 − λ)·ŝ + λ·p. Phones without a prior get the prior's median.
- **Personalized-PageRank reranker.** For each query it seeds personalized PageRank from that
  query's own top 10 results (weighted 1/rank), so rank flows to phones linked to the best hits.
- **Output.** Runs are written in the shared TREC format, named `<base>+<prior><λ>_<split>`, so
  the team's ablation script picks them up without manual work.

**Protocol.** λ ∈ {0.1, 0.2, 0.3, 0.5} chosen by nDCG@10 on the 23 scored dev queries; test run
once with the chosen value; paired randomization tests with Holm correction.

**Results** (Table 8.5d, on the team's tuned BM25):

| System | λ | Dev nDCG@10 | Test nDCG@10 | Test change | p (Holm) |
|---|---:|---:|---:|---:|---:|
| BM25 tuned | – | 0.600 | 0.663 | – | – |
| + PageRank | 0.1 | 0.582 | 0.660 | −0.003 | 0.78 |
| + Personalized PageRank | 0.1 | 0.586 | 0.671 | +0.008 | 0.29 |
| + Reviewer trust | 0.2 | 0.590 | 0.674 | +0.011 | 0.12 |

None of the differences is significant. At the chosen weights the priors mostly reorder rather
than replace results: trust changes 4 to 5 of the top-10 positions per query but brings in
only about one new phone.

**Pooling.** I also noted a pooling effect. The reranked runs were not in the judging pool, so
phones they pull in are often unjudged and count as non-relevant, which slightly favours the
baseline.

**Files.** `phonesense/graph/rerank.py`, `scripts/run_rerank.py`, `tests/test_rerank.py` (9 tests),
`runs/bm25_tuned+{pagerank0.1,ppr0.1,trust0.2}_{dev,test}.run`.

### 3.7 Follow-up: does a fully connected graph help? (beyond the plan)

**Question.** My limitations section said that 61.5% of phones are isolated in the co-review
graph. I wanted to test whether that explains the null result instead of only stating it.

**What I did.** I reran both graph priors on the co-review + kNN graph, which links every phone,
with the same protocol (λ on dev, test once). I added `--out` and `--tag` options to my scripts
so the experiment could not touch the default graph, the demo, or the team's ablation table,
and checked that by regenerating the ablation.

**Results** (Table 8.5e):

| System | λ | Dev nDCG@10 | Test nDCG@10 | Test change | p (Holm) |
|---|---:|---:|---:|---:|---:|
| + PageRank, co-review + kNN | 0.1 | 0.589 | 0.662 | −0.001 | 0.96 |
| + Personalized PageRank, co-review + kNN | 0.5 | 0.603 | 0.636 | −0.026 | 0.005 |

**What it shows.**
- The personalized-PageRank setting was the only prior in the study that scored above BM25 on
  dev, but it was significantly worse on test (worse on 10 of 16 queries).
- kNN edges are built from the same listing text the retrievers already use, so they add no new
  evidence about relevance.
- It is also a concrete example of why we choose settings on dev and run test only once.

### 3.8 Follow-up: sensitivity to damping and neighbourhood size (beyond the plan)

**What I did.** I reran both graph priors at λ = 0.1 for damping d ∈ {0.5, 0.7, 0.85, 0.95} on
both graphs, and for k ∈ {4, 8, 16, 32} kNN neighbours: 22 settings in all. I treated this as a
sensitivity check, not as more tuning, because 23 dev queries cannot support another round of
tuning.

**Results** (Table 8.5f).
- None of the 22 settings differs significantly from tuned BM25 (Holm-corrected, every p = 1.00).
- On the co-review graph, damping changes nothing at all. The prior uses each phone's PageRank
  percentile, and the order of phones does not change between d = 0.5 and d = 0.95.
- On the kNN graphs, every setting stays within 0.002 of BM25 on test.
- So the graph priors' null result does not depend on one choice of parameters.

### 3.9 Attack tab in the demo (T25)

**What it does.**
- **Inputs:** the user picks a phone, bad-mouthing or ballot stuffing, the attack size (0 to
  100% of its real reviews) and the number of camouflage reviews per fake account.
- **Output:** the tab re-ranks all 7,166 phones under each of the five methods and shows the
  phone's rank before and after, with the trust model labelled as ours.
- **Fake-account reliability:** it shows how reliable the trust model considers the fake
  accounts (0.46) compared with a brand-new account (0.50) and a typical real reviewer (0.56).
- **A 30-second walkthrough:** trust helping at 5%, losing its rank advantage at 50%, and being
  beaten by camouflage.

**Build.**
- `render_attack_tab()` is mounted as the "Fake-review attack" tab in the team's app, and also
  runs on its own (`streamlit run phonesense/ui/attack_app.py`).
- It has a light and dark theme with scoped styling, so it does not interfere with the rest of
  the app.
- It is tested with Streamlit's app tester (`tests/test_attack_tab.py`).

**Files.** `phonesense/ui/attack_tab.py`, `phonesense/ui/attack_app.py`, `.streamlit/config.toml`,
`requirements-ui.txt`.

---

## 4. Report sections I wrote

### Methodology (302 lines, PR #8)

I wrote this for the entire system so that the report describes every method in one consistent
notation:
- notation and text preprocessing;
- TF-IDF lnc.ltc, BM25 and Boolean spec filters;
- dense retrieval, Rocchio pseudo-relevance feedback and HyDE;
- rank fusion (RRF and weighted linear);
- the phone graph, PageRank and personalized PageRank, with pseudocode;
- the reputation baselines and the trust model, with pseudocode;
- prior reranking;
- the attack simulator;
- evaluation metrics (P@k, R@k, F1, MAP, nDCG@k with 2^g − 1 gain) and the labelling process.

### Experiments 8.5 (226 lines, PRs #9, #10, #28, #29)

- **8.5.1:** the data properties that matter for graph and trust methods (rating skew, one-time
  reviewers, uneven coverage).
- **8.5.2:** the graph and PageRank.
- **8.5.3:** trust on the real data.
- **8.5.4:** reranking (Tables 8.5d, 8.5e, 8.5f).
- **8.5.5:** the attack study (Tables 8.5a to 8.5c, Figures 8.5a and 8.5b).

### Conclusions and Limitations (122 lines, PRs #9, #15, #28, #29)

- **Conclusions for the whole project**, filled in from the team's final ablation table and
  the LLM-vs-human agreement results:
  - the test collection and judge agreement (kappa 0.26, with the system order unchanged under
    human labels);
  - the best system (RRF of tuned BM25 and dense retrieval, test nDCG@10 0.677);
  - HyDE against Rocchio;
  - the graph and trust findings.
- **Limitations**, grouped into data (including the 77% of phones with no price), evaluation
  (query count, judge agreement, pool depth), graph and trust, attacks, and engineering.
- **Future work.**

### References (51 lines)

Twelve references in the assignment's format. I added the dataset paper and HippoRAG because
the Methodology and Experiments cite them, and checked the page numbers of the HyDE and
LLM-judge papers against the ACL Anthology and DBLP.

---

## 5. Work with the team

- **Integration.** Before merging the remaining open PRs (#20 to #26: HyDE, the demo UI, the
  LLM judge, the LLM backends, the kappa analysis, the buying guide and the final report
  numbers), I merged them together locally and ran the whole test suite (651, later 663
  passing). This is how the demo app reached `main`.
- **Demo check and fixes (#27).** I ran the full app on the real data and fixed what I found:
  - an example query for a phone that is not in our dataset (replaced with one that is);
  - a fusion label showing k = 60 while the engine uses the dev-tuned k = 5 (the label now
    reads the value from the config);
  - the default system being TF-IDF alone (now the best system on dev, tuned BM25 + dense);
  - a HyDE fix that had been pushed after its PR was merged, so it never reached `main`. I
    brought it in, resolved the conflicts, and checked that HyDE replays the logged queries
    without an API key.
- **Reviews (#3, #4).** I ran the dense retrieval and Rocchio tests and tried Rocchio on the real
  index. Its expansion terms were mostly generic listing words ("unlock", "android", "dual",
  "sim"), which likely explains why it scores below TF-IDF. I passed this on as a suggestion.
- **Checking shared results.** I loaded all 15 of the dense, Rocchio and HyDE run files and the
  embedding cache, confirmed every phone id exists and that the dev scores reproduce, and
  forwarded the runs so the ablation table could be completed.
- **Missing prices.** I traced why price filters fail so often: 77% of phones have no price in
  the source data. It is missing, not unparsed, and the share grows with listing age (92% for
  phones listed in 2014 or earlier, 32% for 2021-23). This explains why one dev query has no
  relevant phone, and it is now in the Limitations.
- **Report consistency.**
  - I aligned the report's nDCG definition with the evaluation code.
  - I corrected the graph description ("bought together" to co-review) in the introduction,
    literature review, objective and proposed model, and regenerated the pipeline figure
    (#17, #18).
  - I filled the final numbers into the conclusions.
- **Demo machine.** My laptop has the real data, the dense model, the LLM replay logs, the graph
  and an LLM API key set up and tested, so the team can present from it.

---

## 6. Engineering

**Tests** (61 test functions):

| File | Tests | What they cover |
|---|---:|---|
| `test_pagerank.py` | 14 | Hand-computed graphs, dangling nodes, personalization, `networkx` cross-check |
| `test_build_graph.py` | 9 | Co-review and kNN builders, statistics, file round trip |
| `test_trust.py` | 8 | Convergence, the Bayesian starting point, bad-mouthing on toy data |
| `test_trust_baselines.py` | 4 | All four baselines on hand-computed reviews |
| `test_attacks.py` | 11 | Injection, camouflage, seeding, non-mutation, the synthetic-review guard |
| `test_attack_report.py` | 4 | Rank and score shift per method |
| `test_rerank.py` | 9 | Prior mixing, missing priors, personalized-PageRank reranking |
| `test_attack_tab.py` | 2 | The tab renders and responds, using Streamlit's app tester |

**Reproducing my results** (from the repository root, with the processed data in place):

```bash
python scripts/build_graph.py
```
```bash
python scripts/build_trust.py
```
```bash
python scripts/run_attacks.py --seeds 3
```
```bash
python scripts/run_attacks.py --seeds 3 --camouflage 5
```
```bash
python scripts/run_rerank.py --run runs/bm25_tuned_dev.run --prior trust
```
```bash
streamlit run phonesense/ui/attack_app.py
```

**Configuration.** All parameters are in `config.yaml`, in the `trust:`, `graph:`, `attacks:`
and `rerank:` blocks; nothing is hard-coded in the scripts.

**Working with the team's code.**
- My files follow the shared contract: the TREC run format, and the graph and trust file
  formats in `CONTRACT.md` 4.8.
- My run names plug into the ablation automatically.
- My follow-up experiments write to separate files and names, so they never change a
  teammate's results.

**History.** 60 small commits, each doing one thing; the linter (ruff) is clean.

---

## 7. Timeline

| Date | Work |
|---|---|
| 17 Nov | PageRank and personalized PageRank; baselines and trust model; co-review and kNN graph; attack simulator with camouflage; prior and personalized-PageRank reranking; the attack tab and theme; the full Methodology section (40 commits) |
| 21 Nov | Ran the graph, trust and attack experiments on the real data release; wrote Experiments 8.5, the Conclusions and Limitations, and the References |
| 22 Nov | Reranking on tuned BM25; final numbers and new limitations; PR reviews; integrated and merged the team's open PRs; the demo check and fixes; the kNN and sensitivity studies; the contributions entry |

## 8. Pull requests

| PR | What |
|---|---|
| #7 | Graph, PageRank, reviewer trust, attack simulator, reranking and attack tab (all my code, 31 commits) |
| #8 | Methodology section |
| #9 | Graph and attack experiments, Conclusions and Limitations, References |
| #10 | Reranking results on tuned BM25 |
| #15 | Final numbers in the conclusions, new limitations, reference page numbers |
| #16, #30 | My CONTRIBUTIONS entry |
| #17, #18 | Graph wording and pipeline figure in the introduction, literature review, objective and model sections |
| #27 | Demo fixes, and bringing the unmerged HyDE and filter fixes onto `main` |
| #28 | Co-review + kNN graph experiment |
| #29 | Damping and kNN-size sensitivity check |

Reviewed and approved: #3 (dense retrieval), #4 (Rocchio). Integrated and merged after a
combined test run: #20 to #26.

## 9. Notes on the results and on changes from the plan

Most of my findings describe limits rather than improvements:
- the priors do not significantly change retrieval quality;
- trust helps against small attacks but not large or camouflaged ones;
- connecting every phone with kNN edges does not help.

I reported each with a significance test and an explanation of the cause, rather than tuning
until something looked better on a small dev set.

Two things changed from the original plan, both documented in the report:
- the graph uses co-review links, because the data has no "bought together" links;
- the attacks were measured on each phone's reputation rank rather than end to end on search
  results.

## 10. File inventory

| Area | Files |
|---|---|
| Graph and PageRank | `phonesense/graph/pagerank.py`, `phonesense/graph/build_graph.py`, `scripts/build_graph.py` |
| Reranking | `phonesense/graph/rerank.py`, `scripts/run_rerank.py` |
| Trust | `phonesense/trust/baselines.py`, `trust.py`, `methods.py`, `toy.py`, `scripts/build_trust.py` |
| Attacks | `phonesense/attacks/simulator.py`, `phonesense/attacks/report.py`, `scripts/run_attacks.py` |
| Demo | `phonesense/ui/attack_tab.py`, `phonesense/ui/attack_app.py`, `.streamlit/config.toml`, `requirements-ui.txt` |
| Runs | `runs/bm25_tuned+pagerank0.1_*`, `+ppr0.1_*`, `+trust0.2_*`, `+knnpr0.1_*`, `+knnppr0.5_*` |
| Figures | `report/figures/attack_rank_shift.png`, `report/figures/attack_rank_shift_camo5.png` |
| Report | `report/sections/methodology.md`, `08_experiments_graph_attack_C.md`, `09_conclusions_limitations_C.md`, `10_references_C.md` |
| Tests | `tests/test_pagerank.py`, `test_build_graph.py`, `test_trust.py`, `test_trust_baselines.py`, `test_attacks.py`, `test_attack_report.py`, `test_rerank.py`, `test_attack_tab.py` |
| Contract and config | `CONTRACT.md` 4.8; `config.yaml` blocks `trust`, `graph`, `attacks`, `rerank` |
