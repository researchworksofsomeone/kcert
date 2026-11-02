"""Standard label-free accuracy estimators, for use with ``project`` and ``select`` (Proposition 4: select by any estimator,
certify by the rules). They work on per-fact probabilities P (R, n, K) in [0, 1] and return an estimated fact error per model.

These are the baselines of the paper (ATC: Garg et al., 2022; DoC: Guillory et al., 2021). Their outputs carry no guarantee;
``project(estimate, lo, hi)`` repairs them and flags the provably impossible ones.
"""
from __future__ import annotations

import numpy as np


def _probs(P):
    P = np.asarray(P, np.float64)
    if P.ndim == 2:
        P = P[None]
    if P.ndim != 3 or P.min() < 0 or P.max() > 1:
        raise ValueError("probabilities must be (R, n, K) in [0, 1]")
    return P


def _conf(P):
    return np.maximum(P, 1 - P)


def average_confidence_error(P_target) -> np.ndarray:
    """1 - mean confidence on the target (R,)."""
    return 1 - _conf(_probs(P_target)).mean((1, 2))


def atc_error(P_source, Y_source, P_target) -> np.ndarray:
    """Average Thresholded Confidence: pick the confidence threshold whose exceedance rate on the labelled source equals the source
    accuracy, then predict target error as the share of target facts below it (R,)."""
    Ps, Pt = _probs(P_source), _probs(P_target); Y = np.asarray(Y_source)
    out = np.empty(len(Ps))
    for r in range(len(Ps)):
        cs, ct = _conf(Ps[r]).ravel(), _conf(Pt[r]).ravel()
        acc = np.mean((Ps[r] > 0.5) == Y)
        t = np.quantile(cs, 1 - acc) if 0 < acc < 1 else 0.5
        out[r] = 1 - np.mean(ct > t)
    return out


def doc_error(P_source, Y_source, P_target) -> np.ndarray:
    """Difference of Confidences: source error plus the drop in mean confidence from source to target (R,)."""
    Ps, Pt = _probs(P_source), _probs(P_target); Y = np.asarray(Y_source)
    err_s = ((Ps > 0.5) != Y[None]).mean((1, 2))
    return np.clip(err_s + _conf(Ps).mean((1, 2)) - _conf(Pt).mean((1, 2)), 0, 1)
