# 4. Objective

> **Status: draft (owner: B).** Targets that depend on results are left open on purpose; we
> report what we measure instead of promising a number in advance.

## 4.1 Overall objective

Build and evaluate a smartphone search engine, **PhoneSense**, that turns a free-text need
("phone under $300 with a battery that lasts two days") into a ranked list of phones from the
Amazon Reviews 2023 Cell Phones data, and show which retrieval ideas from the course actually
help on this task, including how well the ranking holds up against fake reviews.

## 4.2 Specific objectives

| # | Objective | Where it is answered |
|---|---|---|
| O1 | Create a reproducible test collection: 45 queries of four types (spec, aspect, model, vague), a fixed dev/test split, and graded relevance judgements (0-3) from an LLM judge whose agreement with people is measured (Cohen's kappa). | Experiments: test collection |
| O2 | Implement and compare lexical retrieval (TF-IDF lnc.ltc, BM25), dense retrieval, Rocchio feedback and HyDE-style expansion with the same metrics (P@5, P@10, R@10, R@100, F1, MAP, nDCG@10). | Experiments: retrieval |
| O3 | Combine the rankers with rank fusion (RRF, weighted linear) and measure what each component adds (add-one and leave-one-out ablation). | Experiments: fusion and ablation |
| O4 | Add graph and reputation priors (PageRank over the co-review graph, reviewer-trust score) and measure their effect on ranking quality. | Experiments: graph |
| O5 | Measure how much simulated fake reviews (bad-mouthing, ballot stuffing) move a phone under raw average, Bayesian average, helpful-vote average and reviewer trust. | Experiments: attacks |
| O6 | Deliver a demo with per-component toggles and a per-result score breakdown, so every ranking can be explained. | Proposed model: demo |

## 4.3 Research questions

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

## 4.4 Evaluation protocol

Every setting (BM25 `k1` and `b`, Rocchio and HyDE weights, the RRF constant, fusion weights,
the prior weight) is chosen on the 27 **dev** queries only. The 18 **test** queries are used
once per final configuration. Differences between systems are tested with a paired test over
per-query scores; with this few queries many differences will not be significant, and the report
says so.

## 4.5 Out of scope

Training a ranker, fine-tuning an encoder, live prices, real attackers (the attacks are
simulated), and non-phone products. Phones are picked from the dataset by a heuristic filter,
so some accessories may remain and some phones may be missed.
