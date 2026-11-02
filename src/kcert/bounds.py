"""Sharp label-free bounds (Theorems 1-3 of the paper) and their k-robust versions (paper, Section 7, "Imperfect rules").

Notation: A is an (R, n, K) 0/1 array of R perception models' fact predictions on n inputs; ``cons`` is the (n, W) mask of the worlds
consistent with each input (``Knowledge.consistent``). Errors are normalised Hamming distances, err_r = mean_i d(a_ri, c_i).
"""
from __future__ import annotations

import numpy as np


def _as_preds(A) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim == 2:
        A = A[None]
    if A.ndim != 3:
        raise ValueError("predictions must be (R, n, K) or (n, K)")
    if not np.isin(A, (0, 1)).all():
        raise ValueError("predictions must be 0/1 fact vectors; threshold probabilities first")
    return A.astype(np.float64)


def hamming(A, worlds) -> np.ndarray:
    """(R, n, K) predictions -> (R, n, W) normalised Hamming distance of every prediction to every world."""
    A = _as_preds(A); V = np.asarray(worlds, np.float64)
    return (A @ (1 - V).T + (1 - A) @ V.T) / A.shape[-1]


def _check(A, kn, cons):
    A = _as_preds(A); C = np.asarray(cons, bool)
    if C.shape != (A.shape[1], kn.n_worlds):
        raise ValueError(f"cons must be (n, W) = ({A.shape[1]}, {kn.n_worlds}), got {C.shape}")
    if not C.any(1).all():
        raise ValueError("an input has no consistent world: assumption (K) is violated by the rule base itself")
    if A.shape[2] != kn.n_concepts:
        raise ValueError("predictions and knowledge disagree on the number of facts")
    return A, C


def _topk_sum(h, k):
    """Sum of the k largest entries along the last axis (h >= 0)."""
    k = int(k)
    if k <= 0:
        return np.zeros(h.shape[:-1])
    if k >= h.shape[-1]:
        return h.sum(-1)
    return np.partition(h, h.shape[-1] - k, axis=-1)[..., -k:].sum(-1)


# ---------------------------------------------------------------------------------------------------------- Theorem 1
def frame_bounds(A, kn, cons):
    """Per-input floor and ceiling (R, n) of every model's error, exact under (K)."""
    A, C = _check(A, kn, cons)
    H = hamming(A, kn.worlds); Cm = C[None]
    return np.where(Cm, H, np.inf).min(-1), np.where(Cm, H, -np.inf).max(-1)


def brackets(A, kn, cons, k: int = 0, universe: str = "worlds"):
    """Certified bracket [LO, HI] (R,) of each model's mean fact error. Sharp: both ends are attained by a truth allowed by (K).

    k > 0 gives the k-robust bracket: valid when the rules may be wrong on up to k unknown inputs, whose truth is then
    any legal world (``universe='worlds'``) or any fact vector at all (``universe='cube'``)."""
    lo, hi = frame_bounds(A, kn, cons)
    n = lo.shape[1]
    if k <= 0:
        return lo.mean(1), hi.mean(1)
    if universe == "cube":
        lo_u, hi_u = np.zeros_like(lo), np.ones_like(hi)
    elif universe == "worlds":
        lo_u, hi_u = frame_bounds(A, kn, np.ones_like(np.asarray(cons, bool)))
    else:
        raise ValueError("universe must be 'worlds' or 'cube'")
    return (lo.sum(1) - _topk_sum(lo - lo_u, k)) / n, (hi.sum(1) + _topk_sum(hi_u - hi, k)) / n


# ---------------------------------------------------------------------------------------------------------- Theorem 1 (pairwise)
def pairwise_certificate(A, kn, cons, k: int = 0, universe: str = "worlds"):
    """Sharp certificate UP (R, R): UP[r, s] >= err_r - err_s for every truth allowed by (K), with equality attained.
    UP[r, s] < 0 certifies that model r is better than model s, with no fact labels.

    k > 0: k-robust version; the certificate widens by at most 2k/n."""
    A, C = _check(A, kn, cons)
    H = hamming(A, kn.worlds); Cm = C[None]; R, n = H.shape[:2]; UP = np.zeros((R, R))
    if k > 0 and universe not in ("worlds", "cube"):
        raise ValueError("universe must be 'worlds' or 'cube'")
    for r in range(R):
        D = H[r][None] - H                                                        # (R, n, W): d(a_r, v) - d(a_s, v)
        m = np.where(Cm, D, -np.inf).max(-1)                                      # (R, n) per-input sharp term
        tot = m.sum(1)
        if k > 0:
            mu = (A[r][None] != A).mean(-1) if universe == "cube" else D.max(-1)  # max over the universe
            tot = tot + _topk_sum(mu - m, k)
        UP[r] = tot / n
    np.fill_diagonal(UP, 0.0)
    return UP


# ---------------------------------------------------------------------------------------------------------- Theorem 2
def certified_regret(UP):
    """CR (R,): max_s max(0, UP[r, s]), a guaranteed upper bound on err_r - min_s err_s (regret within the pool)."""
    return np.maximum(np.asarray(UP), 0).max(1)


def minimax_select(UP, lo=None) -> int:
    """r_dagger = argmin_r CR(r) (Theorem 2), ties broken by the certified floor if given."""
    CR = certified_regret(UP)
    return int(np.lexsort((np.zeros_like(CR) if lo is None else np.asarray(lo), CR))[0])


def top_set(UP, tol: float = 1e-12):
    """Bool (R,): models that no other model certifiably beats (the candidates that survive the rules' veto)."""
    return ~np.any(np.asarray(UP) < -tol, axis=0)


def certified_pairs(UP, tol: float = 1e-12):
    """Share of ordered pairs (r, s), r != s, with a certified ordering UP[r, s] < 0."""
    UP = np.asarray(UP); off = ~np.eye(len(UP), dtype=bool)
    return float(((UP < -tol) & off).sum() / max(1, off.sum()))


# ---------------------------------------------------------------------------------------------------------- Theorem 3
def shortcut_indistinguishable(A, kn, cons, r: int, s: int) -> float:
    """Theorem 3 check: on inputs where both models' predictions satisfy the rules, the per-input sharp certificate equals their
    Hamming distance, so a rule-consistent shortcut can never be certified worse. Returns the max deviation (0 up to rounding)."""
    A, C = _check(A, kn, cons)
    H = hamming(A, kn.worlds)
    inside = np.where(C[None], H, np.inf).min(-1) == 0
    per = np.where(C, H[r] - H[s], -np.inf).max(-1); ham = (A[r] != A[s]).mean(-1)
    both = inside[r] & inside[s]
    return float(np.abs(per - ham)[both].max()) if both.any() else 0.0
