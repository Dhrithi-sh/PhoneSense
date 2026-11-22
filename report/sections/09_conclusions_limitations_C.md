# Conclusions and Limitations

> Report section R-C3 (owner C). Numbers come from the ablation tables (`results/ablation_*.md`,
> owner A), the kappa result (`results/kappa.json`, owner B) and sections 8.5.1-8.5.5.

## Conclusions

We built PhoneSense, a phone search system over 7,166 phones and 575,816 Amazon reviews, and
measured each of its components separately. Going back to the contributions listed in the
Introduction:

1. **Test collection.** We wrote 45 phone queries of four types (spec, aspect, model, vague),
   split 27 dev / 18 test, and labelled pooled results on a 0-3 scale with an LLM judge. On 89
   test pairs labelled by all of us, its agreement with the human grades is Cohen's kappa 0.26
   over the four grades (0.49 quadratic-weighted, 0.39 for relevant versus not relevant): fair,
   not strong. Scoring the systems with the human grades instead leaves their order unchanged
   (Kendall's tau = 1.00).
2. **Retrieval comparison.** The system chosen on dev, reciprocal rank fusion of tuned BM25 and
   dense retrieval, reaches nDCG@10 0.677 on the test queries, against 0.663 for tuned BM25. The
   highest test score, 0.693, comes from BM25 with Boolean spec filters, which had scored below
   tuned BM25 on dev. Fusing BM25 with dense retrieval helped on both splits (RRF +0.042 on dev
   and +0.014 on test, weighted fusion +0.021 and +0.018), while field-weighted BM25F was much
   worse (-0.194 on test, the only significant difference in the table). With 16 test queries,
   no other difference from tuned BM25 is significant after Holm correction.
3. **HyDE with Rocchio weighting.** Tuning on dev set the Rocchio feedback weight to 0 and the
   HyDE weight to 0.35, so the final system expands the query with the generated document only.
   It beats tuned Rocchio on both splits (nDCG@10 0.586 against 0.507 on dev, 0.564 against
   0.526 on test) and plain TF-IDF (0.564 against 0.544 on test), but the gain over TF-IDF is
   not significant (p = 0.47).
4. **Graph and trust signals.**
   * The co-review graph links only 38.5% of the phones, because 88.4% of reviewers wrote a
     single review. PageRank on it acts as a popularity signal: its top phones are the most
     widely reviewed models. As reranking priors on tuned BM25, global PageRank, personalized
     PageRank and reviewer trust change test nDCG@10 by −0.003, +0.008 and +0.011; none of
     these differences is significant, and on dev all three were slightly worse than BM25.
     Adding kNN edges connects every phone but does not help either: global PageRank is
     unchanged on test, and personalized PageRank, the only prior ahead of BM25 on dev, is
     significantly worse on test (−0.026, p = 0.005).
   * The reviewer-trust score converges in 9 iterations and moves phone quality by only 0.09
     stars on average away from the Bayesian average, because only the quarter of reviews
     written by repeat reviewers carry information about reliability.
   * Against plain fake accounts, reviewer trust reduces how many stars an attacked phone
     loses or gains (for every bad-mouthed target up to 20% attack size) and, for small
     bad-mouthing attacks, how far it falls in the ranking. For large attacks it protects the
     rank less well than the raw average, because it compresses the score scale. Against fake
     accounts that first copy the consensus, it is worse than the raw average for most targets,
     and for every bad-mouthed target once the attack reaches 20%.
5. **Explainable demo.** Every component can be switched on and off in the demo, each result
   shows its score breakdown, and the attack tab lets a viewer inject fake reviews and watch the
   ranking change live.

Taken together, the classic signals carry most of the retrieval quality: on the test queries
no system beats tuned BM25 by more than 0.03 nDCG@10, and none does so significantly. The reputation work gives a clearer negative result than a positive one: a
HITS-style trust score helps a little against naive fake reviews, and an attacker who first
builds a normal-looking history removes even that.

## Limitations

**Data.**
* The collection is a 2023 Amazon snapshot with prices in USD. It does not describe the
  current phone market, and conclusions about specific phones should not be drawn from it.
* Phones were separated from accessories by a keyword-and-rule filter. It was checked on a
  hand-labelled sample, but some accessories and some missing phones remain.
* Specifications were parsed from free-text listings; fields that could not be parsed are
  unknown, so Boolean filters cannot match those phones.
* **Most phones have no price.** The source metadata gives no price for 5,531 of the 7,166
  phones (77%). This is missing in the data, not a parsing failure: Amazon only records a price
  for listings on sale when the snapshot was taken, so 92% of phones first listed in 2014 or
  earlier have none, against 32% of those listed in 2021-2023. A price filter can therefore
  only match about a quarter of the collection, and dev query q04 has no relevant phone for
  this reason. Prices from the 2018 release of the same dataset could fill some of the gaps by
  product id, at the cost of mixing years.

**Evaluation.**
* 45 queries (18 for testing) is a small set. Many differences between systems are not
  statistically significant, and the significance markers in the ablation table should be read
  with that in mind.
* Relevance labels come from an LLM judge checked against 89 human-labelled pairs. Agreement is
  only fair (kappa 0.26; where they differ, the humans were usually more lenient), so absolute
  scores inherit the judge's errors: with the human grades, test nDCG@10 is 0.04-0.07 lower.
  The order of the systems, which is what the conclusions rest on, did not change.
* Only pooled results were judged, to depth 5 (depth 10 for queries with no relevant phone).
  A phone that no pooled system retrieved counts as not relevant, which favours the systems in the pool and makes scores for other systems lower
  bounds (reported as judged@10 in the ablation table).

**Graph and trust.**
* The graph has no purchase links: the phone data has no "bought together" field. Co-review
  links leave 61.5% of phones isolated. Semantic kNN edges connect every phone, but they are
  built from the same listing text the retrievers already use, and they did not help (8.5.4).
  The kNN neighbourhood size (8) was not tuned.
* The trust score can only learn about reviewers who wrote more than one review. With 88.4% of
  reviewers writing once, most reliabilities stay near the 0.5 prior, which limits how much the
  score can differ from a Bayesian average.
* Convergence of the trust iteration is observed (9 rounds here), not proven.

**Attacks.**
* The attacks are simulated, not observed. Real review fraud uses aged accounts, varied text,
  coordinated timing and purchases, none of which the simulator models.
* Fake accounts in the simulation are new, unverified and have no helpful votes. That makes them
  easier to discount than real ones, especially for the helpful-vote baseline.
* The trust score has no defence against camouflaged fake accounts: five consensus-copying
  reviews per account are enough to make it weigh the attack fully (8.5.5).
* Results are averaged over 45 targets per attack type and 3 seeds; we report how often trust
  beats the raw average per target, but no confidence intervals.
* The attack study measures each phone's reputation rank. It does not re-run the full search
  pipeline on attacked data, so it does not show how far an attacked phone moves in search
  results for a given query.

**Engineering.**
* LLM-based components (HyDE, the judge, the buying guide) depend on one external model and its
  version. All calls are logged and can be replayed, but a different model would give different
  labels and expansions.
* The reranking weight $\lambda$ is a single value per prior, chosen on 23 dev queries (the
  4 with no relevant phone drop out), and the reranked runs were not part of the judging pool.

## Future work

The most direct improvements are to build graph edges from evidence the text does not contain,
such as co-purchase or browsing data (kNN edges over the same text did not help), to give the trust score more signal through account age and purchase verification, to
weigh reliability by account history rather than agreement alone (camouflage defeats agreement),
to keep the score scale wide enough that small shifts do not cross many phones, and to evaluate
the attacks end to end on search results instead of on reputation rank alone.
