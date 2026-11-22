# 1. Introduction

> **Status: final text (owner: B).** Every number below is from the frozen results: settings
> tuned on the dev queries, test run once (`results/ablation_*.md`, `results/kappa.json`).

## 1.1 Motivation

Buying a smartphone online means turning a loose need ("a phone under $300 with a battery that
lasts two days") into a choice among thousands of near-identical listings. Three things make
this harder than it looks.

1. **The need and the listing use different words.** A shopper writes "battery lasts two
   days"; a listing says "6000mAh, up to 48 hours". Keyword matching misses the link.
2. **Many needs are hard constraints on numbers.** "At least 8 GB RAM and under $400" is not a
   matter of word overlap. The values sit in free text and have to be parsed, and an unknown
   value must never be mistaken for a match.
3. **Ratings can be manipulated.** Shoppers lean on star ratings and reviews, but a rating that
   is easy to push up or down by a few fake reviews is a weak signal.

## 1.2 Problem statement

Given a free-text smartphone need, return a ranked list of phones from the Amazon Reviews 2023
Cell Phones category that best satisfy it, and make that ranking robust to review manipulation.
We measure *ranking quality* with graded relevance judgements (precision, recall, F1, MAP and
nDCG@10) on a set of 45 queries of four types: spec, aspect, model and vague.

## 1.3 Approach in one paragraph

PhoneSense is a hybrid pipeline. A classic first stage (TF-IDF lnc.ltc and BM25 over a positional
inverted index, with Boolean spec filters and phrase search) is combined with neural retrieval
(sentence-embedding search, Rocchio pseudo-relevance feedback, and a HyDE-style expansion in
which an LLM writes a hypothetical listing for the query). Rankings are merged by rank fusion.
A graph signal (PageRank over a co-review graph, which links phones rated by the same reviewers) and a reviewer-trust score
(compared against raw, Bayesian and helpful-vote averages) feed the final ranking, and a
simulator measures how far fake reviews move a phone under each score. An LLM judge, checked
against human labels, supplies the relevance judgements we evaluate with.

## 1.4 Contributions

We list only what we can show. None of the retrieval ideas below is new; the contribution is
how they are combined, measured and made explainable. Test numbers are nDCG@10 on the 16 of the
18 test queries that have a relevant phone (the other two are excluded for every system).

1. **A phone-search test collection**: 45 queries of four types (spec, aspect, model, vague),
   split into 27 dev and 18 test queries, with graded 0-3 judgements from an LLM judge and a
   human-labelled sample on which agreement is measured: on 89 pairs the judge agrees with the
   human grades with Cohen's kappa 0.26 over the four grades (0.49 quadratic-weighted, 0.39 for
   relevant versus not relevant), which is fair, not strong. Scoring the systems with the human
   grades instead leaves their order unchanged.
2. **A controlled comparison** of lexical (TF-IDF, BM25), dense, feedback-based and fused
   retrieval on that collection, with an ablation showing what each component adds. Fusing tuned
   BM25 with dense retrieval gives 0.677 (reciprocal rank fusion) and 0.681 (weighted) against
   0.663 for tuned BM25 on test, a gain that is not statistically significant; field-weighted BM25F
   is significantly worse than BM25. No system beats tuned BM25 by more than 0.03.
3. **An application of HyDE with Rocchio term weighting to phone search**, following Jedidi and
   Lin (2025), with every LLM call logged and replayable so that results do not depend on an
   API key. Tuned on dev, the best setting uses the generated listing alone (the Rocchio feedback
   weight went to 0): 0.564 on test against 0.544 for plain TF-IDF, not significant. Rocchio
   feedback on its own did not help (0.526).
4. **A reviewer-trust score and a fake-review attack study** (bad-mouthing and ballot stuffing,
   simulated) comparing it with raw, Bayesian and helpful-vote averages. It reduces how many
   stars a phone gains or loses against plain fake accounts, but is worse than the raw average
   against accounts that first copy the consensus. As ranking priors on tuned BM25, PageRank,
   personalized PageRank and reviewer trust change test nDCG@10 by -0.003, +0.008 and +0.011,
   none significant.
5. **An explainable demo**: per-component switches, a per-result score breakdown, and an optional
   buying guide that may only restate listed facts and flags figures it cannot find in the data.

## 1.5 Scope and honest limits

Data is the 2023 Amazon snapshot with USD prices; the phone filter is a heuristic; there are
only 45 queries; the LLM judge is validated on a small human sample; and the attacks are
simulated, not observed. The report states these again in Limitations.

## 1.6 Organisation of the report

Section 2 reviews related work, Section 3 states the objective, Section 4 describes the proposed
model, Section 5 the methodology, Section 6 the experiments (retrieval, then graph and attacks),
and Section 7 the conclusions and limitations.
