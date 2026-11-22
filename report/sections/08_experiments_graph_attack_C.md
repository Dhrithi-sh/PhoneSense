# 8.5 Graph and reputation signals, and the review-attack study

> Report section R-C2 (owner C). Every number comes from the real data release (data-v1) and
> the scripts named next to it; nothing here is from the toy data used in tests.

## 8.5.1 Data the graph and trust methods see

The processed collection has 7,166 phones and 575,816 reviews from 489,887 reviewers
(`scripts/build_trust.py`, `results/trust_stats.json`). Three properties of these reviews
shape everything that follows.

* **Ratings are skewed to the extremes.** 52% of reviews give 5 stars and 19% give 1 star;
  the middle three grades share the remaining 29%.
* **Most reviewers write once.** 88.4% of reviewers have a single review in the collection.
  The 56,969 repeat reviewers (11.6%) wrote 24.8% of all reviews, at most 41 each.
* **Coverage per phone is uneven.** The median phone has 20 reviews (quartiles 9 and 59);
  the most reviewed has 8,215. 3,664 phones have at least 20 reviews.

89.5% of reviews are verified purchases, and 62.1% have no helpful votes.

## 8.5.2 The co-review graph and PageRank

Linking two phones when at least two reviewers rated both (7.5) gives 15,323 undirected
pairs, written as 30,646 directed edges (`scripts/build_graph.py`,
`results/graph_stats.json`).

| Property | Value |
|---|---|
| Phones with at least one edge | 2,757 (38.5%) |
| Isolated phones (no edge) | 4,409 (61.5%) |
| Largest connected component | 2,636 phones |
| Out-degree of linked phones | median 3, maximum 625 |
| PageRank iterations ($d = 0.85$, tolerance $10^{-8}$) | 74, in 0.014 s |

The graph is sparse because few reviewers rate more than one phone (8.5.1). Isolated phones
still receive the teleport share of PageRank, so every phone has a score, but the prior
can only separate the 38.5% of phones that are linked. Adding the semantic kNN edges (7.5),
which need the dense embeddings, would connect every phone; this run uses co-review edges
only.

As a sanity check, the phones with the highest PageRank are widely reviewed, long-selling
models: the Samsung Galaxy S3 Mini (5,437 ratings), the BLU Advance 5.5 HD (3,960) and the
second-generation Motorola Moto G (2,370). PageRank here behaves as a popularity and
centrality signal, which is what the prior is meant to capture.

## 8.5.3 Reviewer trust on real data

The trust iteration (7.7) converged in 9 rounds. Because most reviewers have one review,
their reliability stays close to the prior of 0.5: the median reliability is 0.56 and 90%
of reviewers fall between 0.44 and 0.62. Trust-weighted phone quality differs from the
Bayesian average by 0.091 stars on average (largest difference 0.43 stars). In other words,
on this collection the trust score is a modest correction to the Bayesian average, driven by
the quarter of reviews written by repeat reviewers.

## 8.5.4 Effect of the priors on ranking

**[pending: needs relevance labels.]** The rows "PageRank rerank", "Personalized PR" and
"Trust-weighted rerank" of the ablation table (8.3) report P@10, MAP and nDCG@10 for BM25
reranked with each prior, with $\lambda$ tuned on the dev queries and applied once to the
test queries. Until the labels exist, we can only report how much the priors reorder
results. On the 27 dev queries with untuned BM25 and $\lambda = 0.2$, each prior changes on
average 4 to 5 of the top-10 positions per query. The top result stays the same for 25 of 27
queries with global PageRank, 26 of 27 with trust, and 19 of 27 with personalized PageRank,
which reacts most because its prior is built from each query's own top results.

## 8.5.5 Review-attack study

**Setup** (7.9; `scripts/run_attacks.py`, `results/attacks/`). For each attack type, 15
targets were drawn with each of 3 seeds (45 targets per type) from phones with at least 20
reviews: well-rated phones for bad-mouthing, poorly rated ones for ballot stuffing. Each target
was attacked with fake reviews equal to 0, 1, 5, 10, 20, 50 and 100% of its real review count,
and we recorded its score and its rank among all 7,166 phones under each reputation method. The
tables report the mean number of rank places the attacker gained, with the mean shift in stars
in brackets.

*Table 8.5a. Bad-mouthing (plain fake accounts): places the target dropped (stars lost).*

| Attack size | Raw average | Bayesian | Helpful-vote | Median | Reviewer trust |
|---:|---:|---:|---:|---:|---:|
| 1% | 238 (0.07) | 225 (0.05) | 151 (0.05) | 0 (0.00) | 160 (0.04) |
| 5% | 644 (0.18) | 662 (0.15) | 418 (0.13) | 143 (0.03) | 459 (0.11) |
| 10% | 1,237 (0.32) | 1,351 (0.27) | 800 (0.22) | 305 (0.08) | 999 (0.21) |
| 20% | 2,384 (0.56) | 2,824 (0.47) | 1,591 (0.40) | 696 (0.21) | 2,395 (0.38) |
| 50% | 4,515 (1.09) | 5,513 (0.95) | 3,511 (0.83) | 2,682 (0.94) | 5,682 (0.85) |

*Table 8.5b. Ballot stuffing (plain fake accounts): places the target rose (stars gained).*

| Attack size | Raw average | Bayesian | Helpful-vote | Median | Reviewer trust |
|---:|---:|---:|---:|---:|---:|
| 1% | 109 (0.05) | 101 (0.04) | 65 (0.03) | 264 (0.14) | 97 (0.03) |
| 5% | 282 (0.12) | 275 (0.09) | 175 (0.07) | 561 (0.37) | 266 (0.08) |
| 10% | 516 (0.20) | 512 (0.15) | 314 (0.13) | 891 (0.58) | 513 (0.14) |
| 20% | 988 (0.35) | 1,043 (0.28) | 609 (0.23) | 1,359 (0.83) | 1,125 (0.26) |
| 50% | 2,322 (0.67) | 2,719 (0.55) | 1,494 (0.47) | 3,770 (1.76) | 3,161 (0.53) |

![Places gained by the attacker against attack size, per reputation method](../figures/attack_rank_shift.png)

*Figure 8.5a. Rank places gained by the attacker, plain fake accounts (mean over 45 targets).*

**Reviewer trust protects the star rating, but not always the rank.** Against plain fake
accounts, trust lost fewer stars than the raw average for every bad-mouthed target up to 20%
attack size (100% of 45 targets) and for 93% at 50%; for ballot stuffing it did so for 82-87%
of targets. The rank tells a different story. Trust kept the target higher than the raw
average for small bad-mouthing attacks (91% of targets at 5%, 76% at 10%), was level at 20%,
and was worse for every target at 50%. The reason is the scale of the scores: trust, like the
Bayesian average, pulls phones toward the mean, so its middle 50% of phones span only 0.40
stars, against 0.81 for the raw average. A smaller loss in stars therefore crosses more
phones. The Bayesian average shows the same effect: for bad-mouthing it ranks worse than the
raw average from 5% upward.

**The simple baselines behave in opposite ways.** The median ignores small bad-mouthing
completely (no movement at 1%) and is the most resistant up to 50%, but it is the least
robust against ballot stuffing: a 1% attack already lifts a weak phone 264 places, because
the median of a phone with mixed ratings flips easily to a higher grade. The helpful-vote
weighted average is the most consistent: it is never the worst method, and it resists ballot
stuffing best at every size. Part of that is an artefact of the simulator: fake
reviews never receive helpful votes, so they always get the lowest weight, while 38% of real
reviews have at least one vote.

*Table 8.5c. Camouflaged attackers (5 consensus-copying reviews per fake account): places
gained under raw average and reviewer trust. Raw is unaffected by camouflage.*

| Attack size | Bad-mouthing: raw | Bad-mouthing: trust | Ballot stuffing: raw | Ballot stuffing: trust |
|---:|---:|---:|---:|---:|
| 5% | 644 | 830 | 282 | 386 |
| 10% | 1,237 | 1,851 | 517 | 762 |
| 20% | 2,386 | 4,039 | 988 | 1,668 |
| 50% | 4,512 | 6,353 | 2,321 | 4,150 |

![Places gained by the attacker, camouflaged fake accounts](../figures/attack_rank_shift_camo5.png)

*Figure 8.5b. As Figure 8.5a, with camouflaged fake accounts.*

**A smarter attacker defeats the trust score.** When each fake account first posts five
reviews that copy the consensus, the trust score rates it as reliable and gives its attack
review full weight. Trust then does worse than the raw average for most targets (69% at 5%
bad-mouthing, and every bad-mouthed target at 20% and 50%): its
advantage in stars disappears (it loses fewer stars than raw for only about half of the
targets), and the target moves 1.3 to 1.8 times as many places as under the raw average. This
is the attacker described in Lecture 9 who builds reputation before abusing it, and the
trust score as designed has no defence against it.

**Summary.** Reviewer trust is a modest defence against naive fake reviews: it reduces the
change in a phone's star rating, and for small bad-mouthing attacks also the change in its
rank. It does not protect rank against large attacks, because it compresses the score scale,
and it fails against fake accounts that first build a consensus-following history.
