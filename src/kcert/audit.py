"""One call that runs the whole certified audit of a perception layer (Algorithm 1 of the paper) and returns a readable report."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bounds import brackets, certified_pairs, certified_regret, minimax_select, pairwise_certificate, top_set
from .select import impossible, project


@dataclass
class AuditReport:
    names: list
    lo: np.ndarray                 # certified floor of each model's fact error
    hi: np.ndarray                 # certified ceiling
    UP: np.ndarray                 # sharp pairwise certificate, UP[r, s] >= err_r - err_s
    certified_regret: np.ndarray   # CR(r)
    top_set: np.ndarray            # models no other model certifiably beats
    minimax_choice: int            # r_dagger
    forced_fraction: float         # share of fact slots the rules force
    k: int = 0                     # number of inputs on which the rules may be wrong
    estimate: np.ndarray | None = None
    estimate_choice: int | None = None
    flagged: np.ndarray | None = None
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        """Plain-text table of the audit."""
        lines = [f"Certified audit of {len(self.lo)} models (forced fraction {self.forced_fraction:.3f}"
                 + (f", rules may be wrong on up to {self.k} inputs" if self.k else "") + ")",
                 f"certified pairwise orderings: {certified_pairs(self.UP):.1%}",
                 f"{'model':<24}{'floor':>8}{'ceiling':>9}{'CR':>8}  top-set" + ("   estimate  projected  impossible" if self.estimate is not None else "")]
        proj = None if self.estimate is None else project(self.estimate, self.lo, self.hi)
        imp = None if self.estimate is None else impossible(self.estimate, self.lo, self.hi)
        for r, nm in enumerate(self.names):
            row = f"{str(nm)[:23]:<24}{self.lo[r]:>8.4f}{self.hi[r]:>9.4f}{self.certified_regret[r]:>8.4f}  {'yes' if self.top_set[r] else 'no ':>7}"
            if proj is not None:
                row += f"   {self.estimate[r]:>8.4f}  {proj[r]:>9.4f}  {'YES' if imp[r] else 'no':>10}"
            if self.flagged is not None and self.flagged[r]:
                row += "   <- floor above tolerance"
            lines.append(row)
        lines.append(f"minimax choice: {self.names[self.minimax_choice]} (regret <= {self.certified_regret[self.minimax_choice]:.4f})")
        if self.estimate_choice is not None:
            lines.append(f"estimator + veto choice: {self.names[self.estimate_choice]} (regret <= {self.certified_regret[self.estimate_choice]:.4f})")
        lines += self.notes
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def audit(A, kn, labels=None, estimate=None, names=None, k: int = 0, universe: str = "worlds", floor_tolerance=None) -> AuditReport:
    """Certified audit with no fact labels.

    A: (R, n, K) 0/1 fact predictions of R models. labels: logged decisions (indices or names), or None if nothing is logged.
    estimate: optional (R,) error estimate from any estimator; it is projected and the pick is certified.
    k: inputs on which the rules may be wrong (from ``certify_k``). floor_tolerance: flag models whose certified floor exceeds it
    (on inputs outside the training support this is the shortcut check of the paper)."""
    A = np.asarray(A)
    if A.ndim == 2:
        A = A[None]
    cons = kn.consistent(labels, A.shape[1]) if labels is None else kn.consistent(labels)
    lo, hi = brackets(A, kn, cons, k=k, universe=universe)
    UP = pairwise_certificate(A, kn, cons, k=k, universe=universe)
    CR = certified_regret(UP); top = top_set(UP)
    names = list(names) if names is not None else [f"model_{r}" for r in range(len(A))]
    rep = AuditReport(names, lo, hi, UP, CR, top, minimax_select(UP, lo), kn.forced_fraction(cons), k)
    if estimate is not None:
        e = np.asarray(estimate, float); rep.estimate = e
        pe = project(e, lo, hi); pool = np.flatnonzero(top)
        rep.estimate_choice = int(pool[np.argmin(pe[pool])])
        n_imp = int(impossible(e, lo, hi).sum())
        if n_imp:
            rep.notes.append(f"{n_imp} estimate(s) are provably impossible under the rules")
    if floor_tolerance is not None:
        rep.flagged = lo > floor_tolerance
        rep.notes.append(f"{int(rep.flagged.sum())} model(s) have a certified error above {floor_tolerance} (each flag is a proof)")
    if labels is None:
        rep.notes.append("nothing logged: bounds use the legal set only; logging decisions makes them much tighter")
    return rep
