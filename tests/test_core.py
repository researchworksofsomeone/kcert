import itertools

import numpy as np
import pytest

import kcert


def random_setup(seed, R=6, n=60, K=5, W=7, L=3):
    rng = np.random.default_rng(seed)
    worlds = np.unique(rng.integers(0, 2, (W, K)), axis=0)
    allowed = rng.random((len(worlds), L)) < 0.5
    allowed[np.arange(len(worlds)), rng.integers(0, L, len(worlds))] = True            # every world allows some label
    kn = kcert.Knowledge(worlds, allowed)
    truth_w = rng.integers(0, len(worlds), n)
    labels = np.array([rng.choice(np.flatnonzero(allowed[w])) for w in truth_w])
    Y = worlds[truth_w]
    A = np.where(rng.random((R, n, K)) < rng.uniform(0.05, 0.4, (R, 1, 1)), 1 - Y[None], Y[None]).astype(np.int8)
    return kn, A, Y, labels


@pytest.mark.parametrize("seed", range(20))
def test_brackets_and_certificate_valid(seed):
    kn, A, Y, labels = random_setup(seed); cons = kn.consistent(labels)
    err = (A != Y[None]).mean((1, 2))
    lo, hi = kcert.brackets(A, kn, cons)
    assert np.all(lo - 1e-9 <= err) and np.all(err <= hi + 1e-9)
    lof, _ = kcert.brackets(A, kn, kn.consistent(None, A.shape[1]))
    assert np.all(lof <= lo + 1e-9)                                                    # law-only floor is weaker but valid
    UP = kcert.pairwise_certificate(A, kn, cons)
    assert np.all(UP + 1e-9 >= err[:, None] - err[None])


@pytest.mark.parametrize("seed", range(10))
def test_certificate_is_sharp(seed):
    kn, A, Y, labels = random_setup(seed); cons = kn.consistent(labels); UP = kcert.pairwise_certificate(A, kn, cons)
    V = kn.worlds.astype(float)
    for r, s in [(0, 1), (2, 3), (4, 5)]:
        H = (A[:, :, None, :] != V[None, None]).mean(-1)                                # (R, n, W)
        w = np.where(cons, H[r] - H[s], -np.inf).argmax(1); Yw = kn.worlds[w]                  # adversarial truth inside the knowledge
        gap = (A[r] != Yw).mean() - (A[s] != Yw).mean()
        assert abs(gap - UP[r, s]) < 1e-6


@pytest.mark.parametrize("seed", range(10))
def test_selection_certificate_covers_regret(seed):
    kn, A, Y, labels = random_setup(seed); cons = kn.consistent(labels); err = (A != Y[None]).mean((1, 2))
    rng = np.random.default_rng(seed)
    for est in (None, rng.random(len(A)), err + rng.normal(0, 0.05, len(A))):
        for veto in (True, False):
            sel = kcert.select(A, kn, cons, estimate=est, veto=veto)
            assert err[sel.index] - err.min() <= sel.certificate + 1e-9


def test_projection_never_hurts():
    kn, A, Y, labels = random_setup(3); cons = kn.consistent(labels); err = (A != Y[None]).mean((1, 2))
    lo, hi = kcert.brackets(A, kn, cons); est = np.random.default_rng(0).random(len(A))
    assert np.all(np.abs(kcert.project(est, lo, hi) - err) <= np.abs(est - err) + 1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_drift_alarm_no_false_alarm(seed):
    kn, A, Y, labels = random_setup(seed, n=80)
    # target: new truths from the same worlds, readers with arbitrary (often heavier) noise
    rng = np.random.default_rng(seed + 7); tw = rng.integers(0, kn.n_worlds, 80); Yt = kn.worlds[tw]
    At = np.where(rng.random((A.shape[0], 80, kn.n_concepts)) < rng.uniform(0.0, 0.6, (A.shape[0], 1, 1)), 1 - Yt[None], Yt[None])
    alarm, LO, HI = kcert.drift_alarm(A, kn.consistent(labels), At, kn)
    e_s = (A != Y[None]).mean((1, 2)); e_t = (At != Yt[None]).mean((1, 2))
    assert not np.any(alarm & (e_t <= e_s))


def test_theorem9_identity():
    kn, outs = kcert.Knowledge.from_program(lambda a, b: a + b, 4, 2)
    rng = np.random.default_rng(0); n = 50; t = rng.integers(0, 4, (n, 2)); labels = np.array([outs.index(a + b) for a, b in t])
    def enc(p):
        x = np.zeros((len(p), 8), np.int8); x[np.arange(len(p)), p[:, 0]] = 1; x[np.arange(len(p)), 4 + p[:, 1]] = 1; return x
    A = np.stack([enc(t), enc(t[:, ::-1]), enc(rng.integers(0, 4, (n, 2)))])
    assert kcert.shortcut_indistinguishable(A, kn, kn.consistent(labels), 0, 1) < 1e-9


def test_shortcut_count_evenodd_and_repair():
    P = [(0, 6), (2, 8), (4, 6), (4, 8), (1, 5), (3, 7), (1, 9), (3, 9)]
    assert kcert.count_shortcuts(P) == 36
    ev = [0, 2, 4, 6, 8]; brute = sum(all(dict(zip(ev, g))[a] + dict(zip(ev, g))[b] == a + b for a, b in P if a % 2 == 0)
                                     for g in itertools.product(range(10), repeat=5))
    assert brute == 6
    assert kcert.count_shortcuts([(a, b) for a in range(10) for b in range(10)]) == 1
    k, sols = kcert.min_repair(P)
    assert k == 1 and all(kcert.count_shortcuts(P + list(s)) == 1 for s in sols)
