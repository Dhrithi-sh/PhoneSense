"""Edit distance for phonetic matching: plain Levenshtein and a learned version.

Plain Levenshtein charges every edit 1.0. The learned version charges each edit
the negative log of how often it actually happens in Hinglish, estimated from
Aksharantar: frequent slips (`au`↔`o`, `v`↔`w`, a dropped schwa) cost little,
rare ones cost a lot. Costs are learned by aligning each *(canonical spelling,
human spelling)* pair with a Levenshtein backtrace, counting the edits, turning
counts into −log P, then **re-aligning** with the new costs 2–3 times (EM-style,
after Ristad & Yianilos 1998).

Edit operations are character pairs `(src, dst)`:
  ``("a","o")`` substitution · ``("a","")`` deletion · ``("","o")`` insertion.
A match ``("a","a")`` ends up near-free because it is by far the most common op.

Everything runs on *Roman* strings — we compare the canonical romanisation of a
vocabulary term (see ``roman.py``) against the user's romanised spelling.
"""

import json
import math
from collections import defaultdict

INS = ""  # the empty side of an insertion/deletion


def levenshtein(a, b):
    """Plain unit-cost Levenshtein distance (one of the four matchers)."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _align(a, b, cost):
    """Return the min-cost edit ops aligning ``a`` → ``b`` under ``cost(src, dst)``."""
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cost(a[i - 1], INS)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + cost(INS, b[j - 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost(a[i - 1], b[j - 1]),  # substitute / match
                dp[i - 1][j] + cost(a[i - 1], INS),           # delete
                dp[i][j - 1] + cost(INS, b[j - 1]),           # insert
            )
    # Backtrace, preferring substitution, then deletion, then insertion.
    ops, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost(a[i - 1], b[j - 1]):
            ops.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + cost(a[i - 1], INS):
            ops.append((a[i - 1], INS)); i -= 1
        else:
            ops.append((INS, b[j - 1])); j -= 1
    ops.reverse()
    return ops


def _unit_cost(src, dst):
    return 0.0 if (src == dst and src != INS) else 1.0


def _make_cost(table, default):
    """A cost function over an edit table; identical characters are always free.

    Keeping a match at 0 (instead of its −log P value) makes this a proper
    distance: identical strings score 0 and only real edits add cost. The
    learned table still governs every substitution, insertion and deletion.
    """
    def cost(src, dst):
        if src == dst and src != INS:
            return 0.0
        return table.get((src, dst), default)
    return cost


def learn_costs(pairs, iterations=3):
    """Learn an edit-cost table from (canonical, variant) Roman string pairs.

    Returns ``(table, default)`` where ``table`` maps ``(src, dst)`` → cost and
    ``default`` is the cost of an unseen edit. ``iterations`` is the number of
    align-count-reestimate passes (the plan's "re-aligned 2 or 3 times").
    """
    cost = _unit_cost
    table, default = {}, 1.0
    for _ in range(max(1, iterations)):
        counts = defaultdict(int)
        src_total = defaultdict(int)
        dsts = set()
        for a, b in pairs:
            for src, dst in _align(a, b, cost):
                counts[(src, dst)] += 1
                src_total[src] += 1
                dsts.add(dst)
        vocab = len(dsts) + 1  # +1 for the unseen symbol (add-1 smoothing)
        table = {
            (src, dst): -math.log((c + 1) / (src_total[src] + vocab))
            for (src, dst), c in counts.items()
        }
        # Unseen edit: one pseudo-count against the largest source total.
        default = -math.log(1.0 / (max(src_total.values(), default=0) + vocab))
        cost = _make_cost(table, default)
    return table, default


def distance(a, b, table, default):
    """Weighted edit distance between ``a`` and ``b`` using learned costs."""
    cost = _make_cost(table, default)
    m, n = len(a), len(b)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + cost(a[i - 1], INS)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + cost(INS, b[j - 1])
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost(a[i - 1], b[j - 1]),
                dp[i - 1][j] + cost(a[i - 1], INS),
                dp[i][j - 1] + cost(INS, b[j - 1]),
            )
    return dp[m][n]


# --- persistence (tuples aren't JSON keys, so pack as "src>dst") ------------

def save_costs(path, table, default):
    data = {"default": default, "costs": {f"{s}>{d}": c for (s, d), c in table.items()}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=0)


def load_costs(path):
    """Load a cost table saved by ``save_costs``; returns ``(table, default)``."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    table = {}
    for key, cost in data["costs"].items():
        src, dst = key.split(">", 1)
        table[(src, dst)] = cost
    return table, data["default"]


# --- CLI: learn costs from a prepared pairs file ----------------------------

def _main(argv):
    """`python -m dhvani.query.editdist <pairs.tsv> <out.json> [iters]`

    ``pairs.tsv`` has one ``canonical<TAB>variant`` Roman pair per line
    (produced by ``dhvani.query.aksharantar``).
    """
    if len(argv) < 2:
        print(_main.__doc__)
        return 1
    pairs_path, out_path = argv[0], argv[1]
    iters = int(argv[2]) if len(argv) > 2 else 3
    pairs = []
    with open(pairs_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    print(f"learning edit costs from {len(pairs)} pairs, {iters} passes...")
    table, default = learn_costs(pairs, iterations=iters)
    save_costs(out_path, table, default)
    cheap = sorted(((c, s, d) for (s, d), c in table.items() if s != d), key=lambda x: x[0])[:10]
    print(f"saved {len(table)} edit costs to {out_path} (default={default:.3f})")
    print("cheapest non-identity edits (most common real variations):")
    for c, s, d in cheap:
        print(f"  {s or '∅'!r} -> {d or '∅'!r}   cost {c:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
