# Conclusions and Limitations

> Report section R-C3 (owner C). Items marked `[TBD]` are filled from the frozen ablation table
> (owner A) and the kappa result (owner B); everything else is final.

## Conclusions

We built PhoneSense, a phone search system over 7,166 phones and 575,816 Amazon reviews, and
measured each of its components separately. Going back to the contributions listed in the
Introduction:

1. **Test collection.** We wrote 45 phone queries of four types (spec, aspect, model, vague),
   split 27 dev / 18 test, and labelled pooled results on a 0-3 scale with an LLM judge whose
   agreement with human labels is Cohen's kappa `[TBD]`.
2. **Retrieval comparison.** On the test queries the best system reaches nDCG@10 `[TBD]`,
   against 0.663 for tuned BM25. `[TBD: one sentence on which component helped most and
   which did not help, from the add-one and leave-one-out tables.]`
3. **HyDE with Rocchio weighting.** `[TBD: effect on nDCG@10 versus plain Rocchio.]`
4. **Graph and trust signals.**
   * The co-review graph links only 38.5% of the phones, because 88.4% of reviewers wrote a
     single review. PageRank on it acts as a popularity signal: its top phones are the most
     widely reviewed models. As reranking priors on tuned BM25, global PageRank, personalized
     PageRank and reviewer trust change test nDCG@10 by −0.003, +0.008 and +0.011; none of
     these differences is significant, and on dev all three were slightly worse than BM25.
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

Taken together, the classic signals carry most of the retrieval quality `[TBD: confirm from
ablation]`. The reputation work gives a clearer negative result than a positive one: a
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

**Evaluation.**
* 45 queries (18 for testing) is a small set. Many differences between systems are not
  statistically significant, and the significance markers in the ablation table should be read
  with that in mind.
* Relevance labels come from an LLM judge checked against a small human-labelled sample. Where
  the judge and humans disagree, the scores inherit its errors.
* Only pooled results were judged. A phone that no pooled system retrieved counts as not
  relevant, which favours the systems in the pool and makes scores for other systems lower
  bounds (reported as judged@10 in the ablation table).

**Graph and trust.**
* The graph uses co-review links only. The phone data has no "bought together" field, and the
  semantic kNN edges were not built for the final run, so 61.5% of phones are isolated and get
  no help from the PageRank prior.
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

The most direct improvements are to add the semantic kNN edges so that every phone joins the
graph, to give the trust score more signal through account age and purchase verification, to
weigh reliability by account history rather than agreement alone (camouflage defeats agreement),
to keep the score scale wide enough that small shifts do not cross many phones, and to evaluate
the attacks
end to end on search results instead of on reputation rank alone.
