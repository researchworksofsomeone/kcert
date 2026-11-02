"""Knowledge-stratified evaluation (KSE): when some fact labels can be bought, spend them only where the rules are silent and the
models disagree. Slots the rules force are known exactly; slots where all models agree cannot change which model is best.
"""
from __future__ import annotations

import numpy as np

from .bounds import _check


def _strata(A, kn, cons):
    A, C = _check(A, kn, cons)
    f = kn.forced(C)                                                              # (n, K), -1 = not forced
    unforced = f < 0
    disagree = ~np.all(A == A[:1], axis=0)                                        # (n, K)
    return A, f, unforced & disagree


def kse_candidates(A, kn, cons) -> np.ndarray:
    """Indices of the inputs worth labelling: at least one fact is unforced by the rules and the models disagree on it."""
    _, _, D = _strata(A, kn, cons)
    return np.flatnonzero(D.any(1))


def kse_sample(A, kn, cons, m: int, rng=None) -> np.ndarray:
    """Draw m inputs to label, uniformly from ``kse_candidates`` (all of them if there are at most m)."""
    cand = kse_candidates(A, kn, cons)
    rng = np.random.default_rng(rng)
    return cand if len(cand) <= m else np.sort(rng.choice(cand, m, replace=False))


def kse_estimate(A, kn, cons, idx, Y_idx) -> np.ndarray:
    """Error estimate (R,) from the labelled inputs ``idx`` with true fact vectors ``Y_idx`` (from ``kse_sample``).

    Forced slots are counted exactly; unforced disagreement slots are estimated from the sample. Slots where every model agrees and
    the rules are silent are left out: they add the same unknown amount to every model, so differences err_r - err_s (and hence
    model selection) are estimated without bias, but the absolute level is a lower estimate."""
    A, f, D = _strata(A, kn, cons)
    n, K = f.shape
    idx = np.asarray(idx, int); Y = np.asarray(Y_idx, np.int8)
    forced_err = ((A != f[None]) & (f[None] >= 0)).sum((1, 2))
    cand = np.flatnonzero(D.any(1))
    if len(idx) == 0 or len(cand) == 0:
        return forced_err / (n * K)
    if not np.isin(idx, cand).all():
        raise ValueError("idx must come from kse_candidates / kse_sample")
    sampled = ((A[:, idx] != Y[None]) & D[idx][None]).sum((1, 2))
    return (forced_err + len(cand) / len(idx) * sampled) / (n * K)
