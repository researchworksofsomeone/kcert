"""Keep your favourite label-free estimator to pick a model; let the rules certify the pick and repair the estimate.

Run:  python examples/select_and_certify.py
"""
import numpy as np

import kcert

rng = np.random.default_rng(1)
kn, sums = kcert.Knowledge.from_program(lambda a, b: a + b, 10, 2)
n, R = 3000, 8
digits = rng.integers(0, 10, (n, 2)); labels = [sums.index(a + b) for a, b in digits]
Y = kcert.onehot(digits, 10)

# R digit readers with different error rates, as per-fact probabilities
P = []
for r in range(R):
    p = rng.uniform(0.02, 0.3)
    wrong = rng.random((n, 2)) < p
    pred = np.where(wrong, rng.integers(0, 10, (n, 2)), digits)
    P.append(np.clip(kcert.onehot(pred, 10) * 0.85 + 0.075 + rng.normal(0, 0.05, (n, 20)), 0, 1))
P = np.stack(P); A = (P > 0.5).astype(np.int8)
err = (A != Y[None]).mean((1, 2))

est = kcert.average_confidence_error(P)                     # any estimator works here
sel = kcert.select(A, kn, kn.consistent(labels), estimate=est)
print(sel)
print(f"true regret of the pick: {err[sel.index] - err.min():.4f}  (certified bound {sel.certificate:.4f})")
print("estimates that are provably impossible:", np.flatnonzero(kcert.impossible(est, sel.lo, sel.hi)).tolist())
print("mean |error| raw vs projected:", np.abs(est - err).mean().round(4), np.abs(kcert.project(est, sel.lo, sel.hi) - err).mean().round(4))
