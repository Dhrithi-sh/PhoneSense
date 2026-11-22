# Queries

`queries.jsonl` holds the 45 evaluation queries (CONTRACT.md 4.3). Owner: B.

| Type | Count | What it tests | Filters |
|------|-------|---------------|---------|
| `spec`   | 12 | Hard numeric or brand constraints (price, RAM, battery, ...) | yes (CONTRACT 4.4) |
| `aspect` | 12 | A quality described in prose (night photos, battery life, ...) | no |
| `model`  | 11 | A named phone model | no |
| `vague`  | 10 | An open need with no hard spec | no |

The type is stored at the start of `intent_notes` as `type=<type>.` because the `Query`
schema has no separate type field.

## Split

27 dev and 18 test queries, stratified by type (dev / test): spec 7/5, aspect 7/5, model 7/4,
vague 6/4. Within a type the last queries in file order are the test ones.

* **dev**: for development and tuning only (BM25 k1/b, fusion weights, damping, ...).
* **test**: evaluated once per final configuration, never tuned on.

Read them with `read_queries(path, split="dev")`. The file is frozen once all three of us agree;
after that a change needs everyone to re-run their numbers.
