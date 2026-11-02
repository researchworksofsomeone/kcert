"""Proposition 4: select by any estimator, certify by the rules; projection of any estimate onto the certified bracket."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import brackets, certified_regret, minimax_select, pairwise_certificate, top_set


def project(estimate, lo, hi):
    """Clip any error estimate to the certified bracket. Never increases |estimate - true error|.
    Estimates outside [lo, hi] are provably impossible (see ``impossible``)."""
    return np.clip(np.asarray(estimate, float), lo, hi)


def impossible(estimate, lo, hi, tol: float = 1e-12) -> np.ndarray:
    """Bool (R,): estimates that no truth allowed by the rules can produce."""
    e = np.asarray(estimate, float)
    return (e < np.asarray(lo) - tol) | (e > np.asarray(hi) + tol)


@dataclass
class Selection:
    index: int            # chosen model
    certificate: float    # guaranteed bound on its regret within the pool, whatever the estimate was
    method: str
    top_set: np.ndarray   # models no other model certifiably beats
    lo: np.ndarray
    hi: np.ndarray

    def __repr__(self):
        return (f"Selection(index={self.index}, regret <= {self.certificate:.4g}, method={self.method!r}, "
                f"candidates={int(self.top_set.sum())}/{len(self.top_set)})")


def select(A, kn, cons, estimate=None, veto: bool = True, k: int = 0, universe: str = "worlds") -> Selection:
    """Pick a model.

    With ``estimate`` (any error estimate, e.g. ATC, DoC, confidence, a few labels): argmin of the projected estimate, restricted to
    the certified top set if ``veto``. Without: the minimax-optimal r_dagger (Theorem 2). The certificate holds in both cases."""
    lo, hi = brackets(A, kn, cons, k=k, universe=universe)
    UP = pairwise_certificate(A, kn, cons, k=k, universe=universe); CR = certified_regret(UP); top = top_set(UP)
    if estimate is None:
        i = minimax_select(UP, lo); m = "minimax"
    else:
        e = project(estimate, lo, hi)
        if e.shape != lo.shape:
            raise ValueError("estimate must have one value per model")
        pool = np.flatnonzero(top) if veto else np.arange(len(e))
        i = int(pool[np.argmin(e[pool])]); m = "estimate+veto" if veto else "estimate"
    return Selection(i, float(CR[i]), m, top, lo, hi)
