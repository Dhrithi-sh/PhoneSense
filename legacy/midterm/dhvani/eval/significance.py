"""Are differences between two systems real? Paired significance tests.

With a few dozen queries, a difference in MAP can easily be luck. Both tests
compare the two systems query by query (paired), on per-query AP or any
per-query score:

- Paired randomization (permutation) test (Smucker, Allan and Carterette,
  CIKM 2007, recommended for IR): if the two systems were really the same,
  swapping their scores on any query would be equally likely. We swap at
  random many times and count how often the mean difference is at least as
  big as the one we saw. That share is the p-value.
- Paired t-test on the per-query differences, the classic test, for
  comparison. The p-value uses the t distribution (computed here, so no SciPy
  is needed).

p < 0.05 is the usual line for calling a difference significant.
"""

import math
import random

PERMUTATIONS = 10000


def _paired(a, b):
    qids = sorted(set(a) & set(b))
    return [a[q] for q in qids], [b[q] for q in qids]


def randomization_test(a, b, permutations=PERMUTATIONS, seed=0):
    """Two-sided p-value for mean(a) - mean(b) over the queries both have."""
    xs, ys = _paired(a, b)
    n = len(xs)
    if n == 0:
        return 1.0
    diffs = [x - y for x, y in zip(xs, ys)]
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(permutations):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total / n) >= observed - 1e-12:
            at_least += 1
    return (at_least + 1) / (permutations + 1)


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c if abs(c) > 1e-30 else 1e30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c if abs(c) > 1e-30 else 1e30
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _incomplete_beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_test(a, b):
    """(t, two-sided p) for the paired t-test on mean(a) - mean(b)."""
    xs, ys = _paired(a, b)
    n = len(xs)
    if n < 2:
        return 0.0, 1.0
    diffs = [x - y for x, y in zip(xs, ys)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return (0.0, 1.0) if mean == 0 else (math.copysign(math.inf, mean), 0.0)
    t = mean / math.sqrt(var / n)
    df = n - 1
    p = _incomplete_beta(df / 2, 0.5, df / (df + t * t))
    return t, p


def compare(a, b, permutations=PERMUTATIONS):
    """Summary for two per-query score dicts: means, difference and both p-values."""
    xs, ys = _paired(a, b)
    n = len(xs)
    t, p_t = t_test(a, b)
    return {
        "queries": n,
        "mean_a": sum(xs) / n if n else 0.0,
        "mean_b": sum(ys) / n if n else 0.0,
        "difference": (sum(xs) - sum(ys)) / n if n else 0.0,
        "p_randomization": randomization_test(a, b, permutations),
        "t": t,
        "p_t": p_t,
    }
