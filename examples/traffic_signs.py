"""A small driver-assistance rule base: certify three simulated sign readers with no fact labels, then watch for drift.

Run:  python examples/traffic_signs.py
"""
import numpy as np

import kcert

# Facts read by the camera model: red rim, round, octagon, triangle, number, blue background
FACTS = ["red", "round", "octagon", "triangle", "number", "blue", "pictogram"]
# Legal sign designs (the sign catalogue) and the action the law attaches to each
SIGNS = {
    (1, 1, 0, 0, 1, 0, 0): "obey speed limit",
    (1, 0, 1, 0, 0, 0, 0): "stop",
    (1, 0, 0, 1, 0, 0, 0): "give way",
    (1, 0, 0, 1, 0, 0, 1): "slow down (warning)",
    (1, 1, 0, 0, 0, 0, 0): "prohibition",          # no entry
    (1, 1, 0, 0, 0, 0, 1): "prohibition",          # no overtaking: same action, different design
    (0, 1, 0, 0, 0, 1, 1): "follow direction",
    (0, 1, 0, 0, 1, 1, 0): "minimum speed",
}
kn = kcert.Knowledge.from_rules(list(SIGNS), lambda v: SIGNS[tuple(int(x) for x in v)], concept_names=FACTS)
print(kn)

rng = np.random.default_rng(0)
n = 3000
truth_w = rng.integers(0, kn.n_worlds, n)
Y = kn.worlds[truth_w]                                                   # true facts (unknown to the audit)
actions = [SIGNS[tuple(int(x) for x in y)] for y in Y]                   # logged decisions (known)


def reader(noise, n_inputs, Ytrue):
    flip = rng.random((n_inputs, len(FACTS))) < noise
    return np.where(flip, 1 - Ytrue, Ytrue).astype(np.int8)


noise = [0.02, 0.06, 0.15]
A = np.stack([reader(p, n, Y) for p in noise])
names = [f"reader (noise {p})" for p in noise]

rep = kcert.audit(A, kn, actions, names=names)
print(rep.summary())
print("true errors:", np.round((A != Y[None]).mean((1, 2)), 4), "(never seen by the audit)")

# Drift: the same readers in fog read worse. Nothing is logged on the target.
Yt = kn.worlds[rng.integers(0, kn.n_worlds, n)]
At = np.stack([reader(min(0.5, 4 * p), n, Yt) for p in noise])
alarm = kcert.drift_alarm(A, kn.consistent(actions), At, kn)
print(alarm, "| floors on target:", np.round(alarm.target_floor, 4), "| source ceilings:", np.round(alarm.source_ceiling, 4))
