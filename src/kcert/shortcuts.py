"""Reasoning shortcuts of additive programs (MNIST-Add family): exact count and minimal repair (Theorem 6 of the paper)."""
from __future__ import annotations

import itertools


def count_shortcuts(pairs, n_values=10):
    """Number of deterministic maps alpha on {0..n_values-1} with alpha(a) + alpha(b) = a + b for every (a, b) in `pairs`
    (1 = the identity only, i.e. no shortcut). Exact, linear time: one value per connected component fixes the rest."""
    adj = {d: [] for d in range(n_values)}
    for a, b in pairs:
        adj[a].append(b); adj[b].append(a)
    seen, total = set(), 1
    for d0 in range(n_values):
        if d0 in seen:
            continue
        if not adj[d0]:
            seen.add(d0); total *= n_values; continue
        stack = [d0]
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u); stack.extend(adj[u])
        good = 0
        for v0 in range(n_values):
            al, q, ok = {d0: v0}, [d0], True
            while q and ok:
                u = q.pop()
                for w in adj[u]:
                    vw = u + w - al[u]
                    if not 0 <= vw < n_values or (w in al and al[w] != vw):
                        ok = False; break
                    if w not in al:
                        al[w] = vw; q.append(w)
            good += ok
        total *= good
    return total


def min_repair(pairs, n_values=10, kmax=3):
    """Theorem 6: smallest sets of extra pairs whose sum labels leave only the identity map. Returns (k, list of k-tuples)."""
    have = set(pairs) | {(b, a) for a, b in pairs}
    cand = [(a, b) for a in range(n_values) for b in range(a, n_values) if (a, b) not in have]
    for k in range(1, kmax + 1):
        sols = [c for c in itertools.combinations(cand, k) if count_shortcuts(list(have) + list(c), n_values) == 1]
        if sols:
            return k, sols
    return None, []
