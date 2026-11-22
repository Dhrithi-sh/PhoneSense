# CONTRIBUTIONS

Fill at Gate 4. Numbers come from the task board ("Done by" and "WU done") and are checked against
`git shortlog -sn --no-merges` and the merged PRs. If the real split is uneven, write the real
split. Each member fills **their own row**; the WU column is the planned WU of the tasks counted as
completed (a task counts only when its code, tests and run files exist and are pushed).

| Member | Name | GitHub | Completed WU | Task IDs | Report sections written |
|---|---|---|---|---|---|
| A | Dhrithi Shashibushan | Dhrithi-sh | [from task board] (details below) | T01, T02, T03, T04, T05, T06, T07, T11, T22, final assembly | Abstract; Dataset; Experiments: test collection, main results, lexical baselines and tuning, filters, efficiency, failure cases (8.1-8.4); Appendix A; Methodology parts for BM25, filters and evaluation (7.2, 7.10); report assembly and numbering |
| B | Palleti Viraja | virajapalleti | 30.5 (details below) | T08, T12, T13, T14, T15, T09a, T09b, L-B, T23; T24 in progress | Introduction, Literature Review, Objective, Proposed Model (drafts: result numbers still to fill) |
| C | Rishit Kamboj | rishitkam | 44.5 (details below) | T16, T17, T18, T19, T20, T21, T25, L-C, R-C1, R-C2, R-C3; R-C4 in progress | Methodology; Experiments: graph, trust, attacks and reranking (8.5); Conclusions and Limitations; References |

## Member A: details

| Task | Planned WU | Status | Where |
|---|---|---|---|
| T01 repo, environment, data/IO contract | [board] | done, merged (PR #1) | `CONTRACT.md`, `phonesense/schemas.py`, `phonesense/io.py`, `phonesense/retrieval/base.py` |
| T02 data pipeline (raw Amazon 2023 to phones and reviews) | [board] | done, merged (PR #1); data shared as release `data-v1` of the private data repo. The hand-check sheets were generated but not filled, so no filter/parser accuracy is reported | `phonesense/data/`, `scripts/build_data.py`, `results/data_stats.json` |
| T03 positional inverted index (ported from the midterm), VB compression | [board] | done, merged (PR #1) | `phonesense/index/`, `legacy/PORTING.md` |
| T04 TF-IDF lnc.ltc | [board] | done, merged (PR #1) | `phonesense/retrieval/tfidf.py` |
| T05 BM25, BM25F-lite, dev-only tuning | [board] | done, merged (PRs #5, #19); k1 = 1.2, b = 0.9 | `phonesense/retrieval/bm25.py`, `scripts/tune_bm25.py`, `runs/bm25*` |
| T06 Boolean spec filters and constraint parser | [board] | done, merged (PRs #5, #19) | `phonesense/retrieval/filters.py`, `runs/bm25_tuned+filters*` |
| T07 Boolean, phrase and proximity search, phrase boost | [board] | done, merged (PRs #5, #19) | `phonesense/retrieval/{boolean,positional}.py`, `scripts/bench_boolean.py` |
| T11 evaluation harness (metrics, significance, kappa helpers, tables, CLI) | [board] | done, merged (PR #5) | `phonesense/eval/` |
| T22 ablation over all members' runs, report numbers, reproduction, submission ZIP | [board] | done, merged (PRs #5, #19); validated and integrated B's and C's run files | `scripts/ablation.py`, `ablation.yaml`, `scripts/report_numbers.py`, `scripts/reproduce.py`, `scripts/make_submission.py` |
| Report: abstract, dataset, retrieval experiments, appendix, assembly | [board] | done | `report/sections/01_*`, `06_*`, `08a_experiments_retrieval_A.md`, `11_*`, `scripts/build_report.py`, `report/PhoneSense_report.pdf` |

Work beyond the plan: the private data repository and releases for sharing the processed data,
validation of every teammate run file against the contract before it entered the ablation,
merging all pull requests into `main` (resolving the `config.yaml` conflicts of PRs #3 and #11),
and the report build (one PDF from the section files with consistent numbering).

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

| Task | Planned WU | Status | Where |
|---|---|---|---|
| T17 PageRank (power iteration, dangling nodes) and personalized PageRank | 6 | done, merged (PR #7); matches networkx to 1e-9 | `phonesense/graph/pagerank.py` |
| T19 reviewer trust (HITS-inspired) and baselines (raw, Bayesian, helpful-vote, median) | 6 | done, merged (PR #7); run on data-v1 | `phonesense/trust/`, `scripts/build_trust.py` |
| T16 phone graph and edge statistics | 4 | done, merged (PR #7). The data has no "bought together" field, so the graph uses co-review edges; kNN edges are built but not used in the final run | `phonesense/graph/build_graph.py`, `scripts/build_graph.py` |
| T20 attack simulator (bad-mouthing, ballot stuffing, seeds, non-mutating) | 5 | done, merged (PR #7) | `phonesense/attacks/simulator.py` |
| T18 reranking with the priors and lambda sweep | 3 | done, merged (PR #7, results in PR #10); lambda tuned on dev, test run once. No damping sweep | `phonesense/graph/rerank.py`, `scripts/run_rerank.py`, `runs/bm25_tuned+*` |
| T21 attack experiments (sizes 0-100%, 3 seeds, 45 targets per type) | 4 | done (PRs #7, #9); rank and star shift per method, plus camouflaged attackers. Measured on reputation rank, not end-to-end search nDCG | `scripts/run_attacks.py`, `phonesense/attacks/report.py`, `report/figures/attack_rank_shift*.png` |
| L-C hand-labelling share | 1.5 | done (share 1/3) | `qrels/human.txt` |
| T25 attack tab for the demo | 3 | done, merged (PR #7); standalone page and `render_attack_tab()` for the UI | `phonesense/ui/attack_tab.py`, `.streamlit/config.toml` |
| R-C1 Methodology | 6 | done, merged (PR #8) | `report/sections/07_methodology_C.md` |
| R-C2 Experiments: graph, trust, attacks, reranking | 3 | done, merged (PRs #9, #10) | `report/sections/08b_experiments_graph_attack_C.md` |
| R-C3 Conclusions, Limitations, References | 3 | done, merged (PRs #9, #15) | `report/sections/09_*`, `10_references_C.md` |
| R-C4 final pass (numbers against tables, numbering, references) | 1.5 | in progress | whole report |

Work beyond the plan: camouflaged sybil accounts that copy the consensus before attacking (the
attack the trust score fails against), an attack report with per-method rank and score shift,
the demo theme, the missing-price analysis (77% of phones), validation of B's run files and dense
cache, reviews of PRs #3 and #4, and checking the reference page numbers.

