"""MNIST-EvenOdd: count reasoning shortcuts exactly, find the minimal repair, and catch a shortcut model with no digit labels.

Run:  python examples/reasoning_shortcuts.py
"""
import numpy as np

import kcert

SUPPORT = [(0, 6), (2, 8), (4, 6), (4, 8), (1, 5), (3, 7), (1, 9), (3, 9)]          # training pairs of MNIST-EvenOdd
print("maps alpha with alpha(a)+alpha(b)=a+b on the support:", kcert.count_shortcuts(SUPPORT))   # 36 -> 35 shortcuts
k, fixes = kcert.min_repair(SUPPORT)
print(f"minimal repair: {k} extra pair type; {len(fixes)} choices, e.g. {fixes[0]}")

# A shortcut that is perfect on the training sums: swap within the even and odd digits consistently
alpha = {0: 2, 6: 4, 4: 6, 8: 6, 2: 4, 1: 3, 5: 3, 9: 7, 3: 5, 7: 5}
assert all(alpha[a] + alpha[b] == a + b for a, b in SUPPORT)

kn, sums = kcert.Knowledge.from_program(lambda a, b: a + b, 10, 2)
rng = np.random.default_rng(0)
digits = rng.integers(0, 10, (4000, 2))                                            # audit on all digit pairs
labels = [sums.index(a + b) for a, b in digits]
correct = kcert.onehot(digits, 10)
shortcut = kcert.onehot(np.vectorize(alpha.get)(digits), 10)
A = np.stack([correct, shortcut])

in_support = np.array([(a, b) in SUPPORT for a, b in digits])
rep_in = kcert.audit(A[:, in_support], kn, np.array(labels)[in_support], names=["correct", "shortcut"])
# Facts are 20 one-hot bits; one wrong digit costs 2 of 20 bits, so digit error = 5 x fact error. Flag digit error >= 0.2:
rep_all = kcert.audit(A, kn, labels, names=["correct", "shortcut"], floor_tolerance=0.2 / 5)
print("\nInside the training support (Theorem 3: indistinguishable):\n" + rep_in.summary())
print("\nOn all digit pairs (the shortcut is caught, no digit labels used):\n" + rep_all.summary())
print(f"certified digit-error floor of the shortcut model: {5 * rep_all.lo[1]:.3f}; true digit error: {(np.vectorize(alpha.get)(digits) != digits).mean():.3f}")
