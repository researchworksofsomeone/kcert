"""Certify how many inputs may violate the rules, from a few labelled inputs (paper, Section 7, "Imperfect rules").

Label m inputs drawn at random from the n audited inputs, count how many violate assumption (K) (``Knowledge.check``), and get a
number k such that, with probability at least 1 - delta, at most k of the n inputs violate the rules. Pass that k to
``brackets(..., k=k)`` or ``pairwise_certificate(..., k=k)``: every bound then holds simultaneously with probability 1 - delta.
"""
from __future__ import annotations

import math


def _log_binom_pmf(x, m, p):
    if p <= 0:
        return 0.0 if x == 0 else -math.inf
    if p >= 1:
        return 0.0 if x == m else -math.inf
    return math.lgamma(m + 1) - math.lgamma(x + 1) - math.lgamma(m - x + 1) + x * math.log(p) + (m - x) * math.log1p(-p)


def binom_cdf(x: int, m: int, p: float) -> float:
    """P(Bin(m, p) <= x), summed in log space."""
    if x < 0:
        return 0.0
    if x >= m:
        return 1.0
    terms = [_log_binom_pmf(i, m, p) for i in range(x + 1)]
    top = max(terms)
    return 0.0 if top == -math.inf else min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson_upper(violations: int, m: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper confidence bound on a rate: the largest p with P(Bin(m, p) <= violations) >= delta."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if violations >= m:
        return 1.0
    lo, hi = violations / m, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if binom_cdf(violations, m, mid) > delta else (lo, mid)
    return hi


def _log_comb(a, b):
    if b < 0 or b > a:
        return -math.inf
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def hypergeom_cdf(x: int, N: int, Kbad: int, m: int) -> float:
    """P(X <= x) for X = violations among m draws without replacement from N inputs of which Kbad violate."""
    terms = [_log_comb(Kbad, i) + _log_comb(N - Kbad, m - i) - _log_comb(N, m) for i in range(0, min(x, m, Kbad) + 1)]
    terms = [t for t in terms if t > -math.inf]
    if not terms:
        return 0.0
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def certify_k(violations: int, m: int, n: int, delta: float = 0.05, method: str = "hypergeometric") -> int:
    """Upper bound k on the number of rule violations among the n audited inputs, valid with probability >= 1 - delta.

    violations: number of the m labelled inputs that violate (K). The m inputs must be a uniform random sample (without replacement
    for ``method='hypergeometric'``, the exact and tighter choice; with replacement for ``'clopper-pearson'``)."""
    if not (0 <= violations <= m <= n):
        raise ValueError("need 0 <= violations <= m <= n")
    if method == "clopper-pearson":
        return min(n, math.ceil(clopper_pearson_upper(violations, m, delta) * n))
    if method != "hypergeometric":
        raise ValueError("method must be 'hypergeometric' or 'clopper-pearson'")
    if m == n:
        return violations
    lo, hi = violations, n - (m - violations)                                   # largest Kbad with P(X <= violations | Kbad) > delta
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if hypergeom_cdf(violations, n, mid, m) > delta else (lo, mid - 1)
    return lo
