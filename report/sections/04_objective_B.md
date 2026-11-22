# Objective

> **Status: final text (owner: B).** The objectives were fixed before the experiments and no
> numeric target was promised in advance; the closing table gives what we measured.

## Overall objective

Build and evaluate a smartphone search engine, **PhoneSense**, that turns a free-text need
("phone under $300 with a battery that lasts two days") into a ranked list of phones from the
Amazon Reviews 2023 Cell Phones data, and show which retrieval ideas from the course actually
help on this task, including how well the ranking holds up against fake reviews.

## Specific objectives

| # | Objective | Where it is answered |
|---|---|---|
| O1 | Create a reproducible test collection: 45 queries of four types (spec, aspect, model, vague), a fixed dev/test split, and graded relevance judgements (0-3) from an LLM judge whose agreement with people is measured (Cohen's kappa). | Experiments: test collection |
| O2 | Implement and compare lexical retrieval (TF-IDF lnc.ltc, BM25), dense retrieval, Rocchio feedback and HyDE-style expansion with the same metrics (P@5, P@10, R@10, R@100, F1, MAP, nDCG@10). | Experiments: retrieval |
| O3 | Combine the rankers with rank fusion (RRF, weighted linear) and measure what each component adds (add-one and leave-one-out ablation). | Experiments: fusion and ablation |
| O4 | Add graph and reputation priors (PageRank over the co-review graph, reviewer-trust score) and measure their effect on ranking quality. | Experiments: graph |
| O5 | Measure how much simulated fake reviews (bad-mouthing, ballot stuffing) move a phone under raw average, Bayesian average, helpful-vote average and reviewer trust. | Experiments: attacks |
| O6 | Deliver a demo with per-component toggles and a per-result score breakdown, so every ranking can be explained. | Proposed model: demo |

## Research questions

* **RQ1.** Does dense retrieval beat BM25 on this collection, and does the answer differ by query
  type (spec, aspect, model, vague)?
* **RQ2.** Do pseudo-relevance feedback and HyDE-style expansion improve nDCG@10 over their
  TF-IDF base, or do they drift?
* **RQ3.** Does fusing rankers beat the best single ranker, and which component can be removed
  with the smallest loss?
* **RQ4.** Does a graph or trust prior improve ranking quality, or only change it?
* **RQ5.** How much do fake reviews move a target phone under each reputation method, and does
  reviewer trust reduce the shift compared with a raw average?
* **RQ6.** How closely does the LLM judge agree with human labels (kappa on the shared sample)?

## Evaluation protocol

Every setting (BM25 `k1` and `b`, Rocchio and HyDE weights, the RRF constant, fusion weights,
the prior weight) is chosen on the 27 **dev** queries only. The 18 **test** queries are used
once per final configuration. Differences between systems are tested with a paired test over
per-query scores; with this few queries many differences will not be significant, and the report
says so.

## Out of scope

Training a ranker, fine-tuning an encoder, live prices, real attackers (the attacks are
simulated), and non-phone products. Phones are picked from the dataset by a heuristic filter,
so some accessories may remain and some phones may be missed.

## Answers in brief

Numbers are nDCG@10 on the 16 of 18 test queries that have a relevant phone, settings tuned on dev
(details and significance tests in the Experiments sections).

| Question | Answer |
|---|---|
| RQ1 dense vs BM25 | No. Dense 0.417 against 0.663 for tuned BM25 (and 0.544 for TF-IDF). |
| RQ2 feedback and HyDE | Rocchio feedback did not help (0.526 against 0.544 for its TF-IDF base; the tuned weight was the smallest tried). HyDE used on its own, without feedback, gave 0.564, a gain over TF-IDF that is not significant. |
| RQ3 fusion | RRF of BM25 and dense reaches 0.677 and weighted fusion 0.681 against 0.663 for tuned BM25 (+0.014 and +0.018; +0.042 and +0.021 on dev), not significant; the published RRF constant k = 60 was much worse than the tuned k = 5. |
| RQ4 graph and trust priors | They change nDCG@10 by -0.003 (PageRank), +0.008 (personalized PageRank) and +0.011 (reviewer trust); none is significant. |
| RQ5 fake reviews | Reviewer trust shrinks the star shift against plain fake accounts, but against accounts that first copy the consensus it is worse than the raw average (Experiments, graph and attacks). |
| RQ6 judge vs humans | Cohen's kappa 0.26 over four grades (0.49 weighted, 0.39 for relevant versus not) on 89 pairs: fair, not strong. The order of the systems does not change when the human grades are used. |
