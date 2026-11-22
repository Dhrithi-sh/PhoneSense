# CONTRIBUTIONS

Fill at Gate 4. Numbers come from the task board ("Done by" and "WU done") and are checked against
`git shortlog -sn --no-merges` and the merged PRs. If the real split is uneven, write the real
split. Each member fills **their own row**; the WU column is the planned WU of the tasks counted as
completed (a task counts only when its code, tests and run files exist and are pushed).

| Member | Name | GitHub | Completed WU | Task IDs | Report sections written |
|---|---|---|---|---|---|
| A | | | | | |
| B | Palleti Viraja | virajapalleti | 30.5 (details below) | T08, T12, T13, T14, T15, T09a, T09b, L-B, T23; T24 in progress | Introduction, Literature Review, Objective, Proposed Model (drafts: result numbers still to fill) |
| C | Rishit Kamboj | rishitkam | 44.5 (details below) | T16, T17, T18, T19, T20, T21, T25, L-C, R-C1, R-C2, R-C3; R-C4 in progress | Methodology; Experiments 8.5 (graph, trust, reranking, kNN and sensitivity studies, attacks); Conclusions and Limitations; References |

## Member B: details

| Task | Planned WU | Status | Where |
|---|---|---|---|
| T08 45 queries, dev/test split | 4 | done, merged (PR #2) | `queries/queries.jsonl` |
| T12 dense retrieval with embedding cache | 4 | done, PR #3 | `phonesense/retrieval/dense.py` |
| T14 Rocchio pseudo-relevance feedback | 3 | done, PR #4; tuned on dev | `phonesense/retrieval/rocchio.py` |
| T15 HyDE expansion (every LLM call logged) | 5 | done, tuned on dev | `phonesense/retrieval/hyde.py`, `phonesense/llm/` |
| T09a LLM judge and pooling | 3 | done; round 1 on dev (495 pairs) | `phonesense/llm/judge.py`, `pooling.py` |
| T09b round 2 on test (305 pairs), kappa | 2 | done; kappa on 89 human labels | PR #6, `results/kappa.json` |
| L-B hand-labelling share | 1.5 | done (30 pairs) | `qrels/human.txt` |
| T13 fusion: RRF and weighted linear, tuned on dev | 4 | done; fused runs written | `phonesense/retrieval/fusion.py`, `scripts/run_fusion.py` |
| T23 buying-guide generator | 4 | done (figures checked against the data; wrong attribution of a real figure is not caught) | `phonesense/llm/buying_guide.py` |
| T24 demo UI | 8 | in progress: tabs, system toggles, score breakdown, priors, guide button and the attack tab are built; Boolean filters and phrase search are not wired yet | `phonesense/ui/` |
| R-B1 Introduction and literature review | 5 | drafted (six core citations checked, two not re-checked); result numbers to fill | `report/sections/` |
| R-B2 Objective and proposed model | 2 | drafted with a pipeline figure; result numbers to fill | `report/sections/`, `report/figures/pipeline.png` |

Work beyond the plan: an LLM client with Gemini, OpenAI, Mistral and Groq backends (logging,
replay, throttling), the pooling and labelling pipeline (adaptive re-pooling of queries without a
relevant phone, sheets and a terminal helper for hand labels), the LLM-vs-human agreement
analysis and a label-sensitivity check, and tuning of Rocchio, HyDE and fusion on dev only.

## Member C: details

A detailed write-up, with the methods, full result tables and how to reproduce them, is in
[docs/contributions_C.md](docs/contributions_C.md).

### Summary

PhoneSense combines three things: classic IR, neural IR, and trust-aware graph ranking. I was
responsible for the third part. I built the phone graph, PageRank and personalized PageRank,
the reviewer-trust model and its baselines, the fake-review attack simulator, and the
reranking of search results with these signals. I also ran all of their experiments and built
the attack tab for the demo. In the report I wrote the Methodology, the graph, trust and attack
part of the Experiments (8.5), the Conclusions and Limitations, and the References.

| | |
|---|---|
| Planned share | 46 work units (about a third of the project), 12 tasks |
| Completed | 11 of 12 tasks merged; the final report pass is the last one, once the PDF is assembled |
| Pull requests | 11 opened and merged (#7, #8, #9, #10, #15, #16, #17, #18, #27, #28, #29) |
| Team integration | Reviewed #3 and #4; tested and merged #20 to #26 together |
| Commits | 59 on `main` |
| Code | about 1,000 lines in `phonesense/`, 350 in scripts, 700 in tests (61 test functions) |
| Report | about 700 lines across Methodology, Experiments 8.5, Conclusions and Limitations, References |

### Tasks

| Task | Planned WU | Status | Where |
|---|---|---|---|
| T17 PageRank (power iteration, dangling nodes) and personalized PageRank | 6 | done, merged (PR #7); matches networkx to 1e-9 | `phonesense/graph/pagerank.py` |
| T19 reviewer trust (HITS-inspired) and baselines (raw, Bayesian, helpful-vote, median) | 6 | done, merged (PR #7); run on data-v1 | `phonesense/trust/`, `scripts/build_trust.py` |
| T16 phone graph and edge statistics | 4 | done, merged (PR #7). The data has no "bought together" field, so the graph uses co-review edges; kNN edges were tested as a second graph in a follow-up experiment (PR #28) | `phonesense/graph/build_graph.py`, `scripts/build_graph.py` |
| T20 attack simulator (bad-mouthing, ballot stuffing, seeds, non-mutating) | 5 | done, merged (PR #7) | `phonesense/attacks/simulator.py` |
| T18 reranking with the priors and lambda sweep | 3 | done, merged (PR #7, results in PR #10); lambda tuned on dev, test run once; damping and kNN size checked in a sensitivity analysis (PR #29) | `phonesense/graph/rerank.py`, `scripts/run_rerank.py`, `runs/bm25_tuned+*` |
| T21 attack experiments (sizes 0-100%, 3 seeds, 45 targets per type) | 4 | done (PRs #7, #9); rank and star shift per method, plus camouflaged attackers. Measured on reputation rank, not end-to-end search nDCG | `scripts/run_attacks.py`, `phonesense/attacks/report.py`, `report/figures/attack_rank_shift*.png` |
| L-C hand-labelling share | 1.5 | done (share 1/3) | `qrels/human.txt` |
| T25 attack tab for the demo | 3 | done, merged (PR #7); standalone page and `render_attack_tab()` for the UI | `phonesense/ui/attack_tab.py`, `.streamlit/config.toml` |
| R-C1 Methodology | 6 | done, merged (PR #8) | `report/sections/methodology.md` |
| R-C2 Experiments: graph, trust, attacks, reranking | 3 | done, merged (PRs #9, #10, #28, #29) | `report/sections/08_experiments_graph_attack_C.md` |
| R-C3 Conclusions, Limitations, References | 3 | done, merged (PRs #9, #15, #28) | `report/sections/09_*`, `10_references_C.md` |
| R-C4 final pass (numbers against tables, numbering, references) | 1.5 | in progress | whole report |

### What I built

#### Reviewer trust and the attack study

Star ratings are easy to manipulate, so I wanted to see whether we could rank phones in a way
that holds up against fake reviews.

- **Trust model** (`phonesense/trust/`). Phone quality and reviewer reliability are estimated
  together, in the spirit of HITS. A phone's quality is a reliability-weighted, Bayesian-smoothed
  mean of its ratings. A reviewer's reliability depends on how close their ratings are to the
  consensus, and is smoothed for reviewers with few reviews. That smoothing matters here
  because 88.4% of the 489,887 reviewers wrote only one review. The first iteration equals the
  Bayesian average, which made the model straightforward to check. On the full data (575,816
  reviews) it converges in 9 iterations.
- **Baselines:** raw average, Bayesian average, helpful-vote weighted average and median.
- **Attack simulator** (`phonesense/attacks/`). It models bad-mouthing and ballot stuffing from
  new sybil accounts, is seeded, and never modifies the real data. I added camouflaged
  attackers that first copy the consensus to look reliable and only then attack, which is the
  strategy reputation systems are known to be weakest against.
- **Experiments** (8.5.5, Tables 8.5a to 8.5c, Figures 8.5a and 8.5b). I attacked 90 phones
  (15 targets, 3 seeds, 2 attack types) at sizes from 1% to 50% of their real reviews, and
  measured how far each one moved in rank and in stars under every method. Main results:
  - Trust reduced the star damage for every bad-mouthed phone up to a 20% attack.
  - For small attacks it also kept phones ranked higher: 91% of targets at 5%, 76% at 10%.
  - Its advantage in rank disappears for large attacks. I traced this to score compression:
    trust's middle half of phones spans 0.40 stars against 0.81 for the raw average, so a
    smaller drop crosses more phones.
  - Camouflaged attackers get past trust, moving their target 1.3 to 1.8 times as far.

#### PageRank and the phone graph

- **PageRank** (`phonesense/graph/pagerank.py`). I implemented power iteration myself rather than
  using a library, with dangling-node handling, weighted edges and personalized teleport
  vectors. It matches `networkx` to within 1e-9 for both variants and runs in about 0.01 s on
  the full graph. Its top phones are the most reviewed models, as expected.
- **Phone graph** (`phonesense/graph/build_graph.py`). The original plan assumed "bought
  together" links, but the dataset doesn't have them. I built a co-review graph instead:
  phones are linked when the same reviewers rated both, and very heavy reviewers are left out.
  I documented this in the team contract and updated the wording in the other report sections.
  I also added kNN edges over the dense embeddings, which connect all 7,166 phones.

#### Reranking and evaluation of the priors

- **Reranker** (`phonesense/graph/rerank.py`, `scripts/run_rerank.py`). It mixes any prior into a
  ranked run, and there is a separate per-query personalized-PageRank reranker. The output feeds
  directly into the team's ablation table.
- **Evaluation** (8.5.4, Table 8.5d). The prior weight was chosen on dev queries only, test was
  run once, and differences were tested with paired randomization tests and Holm correction.
  None of the three priors changes nDCG@10 significantly; trust comes closest at +0.011.
- **Two follow-up experiments I added.**
  - *Is the graph too sparse?* I reran everything on the fully connected kNN graph (Table 8.5e).
    It didn't help. The one setting that edged ahead on dev was significantly worse on test
    (p = 0.005), because kNN edges come from the same text the retrievers already use.
  - *Is the result sensitive to parameters?* I tried damping from 0.5 to 0.95 and kNN size from
    4 to 32 (Table 8.5f). None of the 22 settings made a significant difference.

#### Attack tab in the demo

`phonesense/ui/attack_tab.py`. You pick a phone, an attack type, an attack size and the number
of camouflage reviews, and the tab re-ranks all 7,166 phones under each of the five methods,
showing where the phone was and where it ends up. It also shows how reliable the trust model
thinks the fake accounts are (0.46) compared with a typical real reviewer (0.56). It runs as
a tab in the main app or on its own.

### Report sections

- **Methodology** (302 lines). I wrote this for the whole system, not just my part: TF-IDF, BM25,
  Boolean filters, dense retrieval, Rocchio, HyDE, rank fusion, the graph, PageRank, trust,
  reranking, the attack simulator, and the evaluation metrics and labelling, all in one
  notation.
- **Experiments 8.5** (226 lines). The data properties relevant to my part, the graph, trust,
  reranking, the kNN and sensitivity studies, and the attack study: six tables and two figures.
- **Conclusions and Limitations** (122 lines). Conclusions for the whole project, filled in from
  the team's final ablation table and the agreement results, and the limitations across data,
  evaluation, graph and trust, attacks and engineering.
- **References.** Twelve entries in the required format, with page numbers checked against the
  ACL Anthology and DBLP.

### Work with the team

- **Integration.** Before merging, I combined the remaining open PRs (#20 to #26) locally and ran
  the full test suite (663 passing). This brought the demo app, HyDE, the LLM judge, the kappa
  analysis and the buying guide onto `main`.
- **Demo check.** I ran the full demo on the real data and fixed what I found (#27):
  - an example query for a phone that isn't in our dataset;
  - a fusion label showing the wrong parameter;
  - the default system not being our best one;
  - a HyDE fix that had been pushed after its PR was merged and so never reached `main`.
- **Reviews.** I reviewed the dense retrieval and Rocchio PRs (#3, #4). While testing Rocchio on
  the real index I noticed its expansion terms are mostly generic listing words, which likely
  explains why it scores below TF-IDF.
- **Checks on shared results.** I verified the dense and Rocchio/HyDE run files and the embedding
  cache before they went into the ablation.
- **Data.** I looked into why so many price filters fail: 77% of phones have no price in the
  source data, mostly older listings. This went into the Limitations.
- **Consistency.** I aligned the nDCG definition in the report with the evaluation code, fixed
  the graph description in other sections and the pipeline figure (#17, #18), and filled in the
  final numbers.

### A note on the results

Most of my findings are limits rather than wins: the priors don't significantly improve
retrieval, trust helps against small attacks but not large or camouflaged ones, and adding kNN
edges doesn't help. I've reported each with a significance test and an explanation of the
cause, because I think that is more useful than tuning until something looks better on 23 dev
queries. Two changes from the plan are documented with reasons: the co-review graph replaced
"bought together", and the attacks were measured on reputation rank rather than end-to-end
search.

### Where to find it

| Area | Files | PRs |
|---|---|---|
| Graph, PageRank, reranking | `phonesense/graph/`, `scripts/build_graph.py`, `scripts/run_rerank.py` | #7, #28, #29 |
| Trust and baselines | `phonesense/trust/`, `scripts/build_trust.py` | #7 |
| Attacks | `phonesense/attacks/`, `scripts/run_attacks.py`, `report/figures/attack_rank_shift*.png` | #7, #9 |
| Attack tab | `phonesense/ui/attack_tab.py`, `phonesense/ui/attack_app.py` | #7 |
| Runs | `runs/bm25_tuned+{pagerank,ppr,trust,knnpr,knnppr}*` | #10, #28 |
| Report | `report/sections/methodology.md`, `08_experiments_graph_attack_C.md`, `09_conclusions_limitations_C.md`, `10_references_C.md` | #8, #9, #10, #15, #28, #29 |
| Tests | `tests/test_pagerank.py`, `test_build_graph.py`, `test_trust*.py`, `test_attacks.py`, `test_attack_report.py`, `test_rerank.py`, `test_attack_tab.py` | #7 |
| Team integration | demo fixes, merges of #20 to #26, reviews, consistency fixes | #17, #18, #27 |
