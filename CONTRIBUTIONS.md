# CONTRIBUTIONS

Fill at Gate 4. Numbers come from the task board ("Done by" and "WU done") and are checked against
`git shortlog -sn --no-merges` and the merged PRs. If the real split is uneven, write the real
split. Each member fills **their own row**; the WU column is the planned WU of the tasks counted as
completed (a task counts only when its code, tests and run files exist and are pushed).

| Member | Name | GitHub | Completed WU | Task IDs | Report sections written |
|---|---|---|---|---|---|
| A | | | | | |
| B | Palleti Viraja | virajapalleti | 30.5 (details below) | T08, T12, T13, T14, T15, T09a, T09b, L-B, T23; T24 in progress | Introduction, Literature Review, Objective, Proposed Model (drafts: result numbers still to fill) |
| C | Rishit Kamboj | rishitkam | 44.5 (details below) | T16, T17, T18, T19, T20, T21, T25, L-C, R-C1, R-C2, R-C3; R-C4 in progress | Methodology; Experiments: graph, trust, attacks and reranking (8.5); Conclusions and Limitations; References |

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
| R-C1 Methodology | 6 | done, merged (PR #8) | `report/sections/methodology.md` |
| R-C2 Experiments: graph, trust, attacks, reranking | 3 | done, merged (PRs #9, #10) | `report/sections/08_experiments_graph_attack_C.md` |
| R-C3 Conclusions, Limitations, References | 3 | done, merged (PRs #9, #15) | `report/sections/09_*`, `10_references_C.md` |
| R-C4 final pass (numbers against tables, numbering, references) | 1.5 | in progress | whole report |

Work beyond the plan: camouflaged sybil accounts that copy the consensus before attacking (the
attack the trust score fails against), an attack report with per-method rank and score shift,
the demo theme, the missing-price analysis (77% of phones), validation of B's run files and dense
cache, reviews of PRs #3 and #4, and checking the reference page numbers.

