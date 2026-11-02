import itertools

import numpy as np
import pytest

import kcert
from test_core import random_setup


def violate(kn, labels, Y, k, rng, universe):
    """Replace the truth of k inputs by a vector the rules forbid for that decision."""
    Y = Y.copy(); cons = kn.consistent(labels)
    for i in rng.choice(len(Y), k, replace=False):
        if universe == "worlds":
            bad = np.flatnonzero(~cons[i])
            if len(bad):
                Y[i] = kn.worlds[rng.choice(bad)]
        else:
            Y[i] = rng.integers(0, 2, kn.n_concepts)
    return Y


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("universe", ["worlds", "cube"])
def test_robust_bounds_valid_under_k_violations(seed, universe):
    kn, A, Y, labels = random_setup(seed); rng = np.random.default_rng(seed + 100); k = 4
    Yb = violate(kn, labels, Y, k, rng, universe); cons = kn.consistent(labels)
    err = (A != Yb[None]).mean((1, 2))
    lo, hi = kcert.brackets(A, kn, cons, k=k, universe=universe)
    assert np.all(lo - 1e-9 <= err) and np.all(err <= hi + 1e-9)
    UP = kcert.pairwise_certificate(A, kn, cons, k=k, universe=universe)
    assert np.all(UP + 1e-9 >= err[:, None] - err[None])
    UP0 = kcert.pairwise_certificate(A, kn, cons)
    off = ~np.eye(len(UP), dtype=bool)
    assert np.all((UP - UP0)[off] <= 2 * k / A.shape[1] + 1e-9)                        # widening is at most 2k/n
    lo0, hi0 = kcert.brackets(A, kn, cons, k=0)
    assert np.all(lo <= lo0 + 1e-12) and np.all(hi >= hi0 - 1e-12)


def test_robust_k0_equals_plain():
    kn, A, Y, labels = random_setup(1); cons = kn.consistent(labels)
    assert np.allclose(kcert.pairwise_certificate(A, kn, cons, k=0), kcert.pairwise_certificate(A, kn, cons))


def test_certify_k_coverage():
    rng = np.random.default_rng(0); n, m, delta, misses = 400, 60, 0.1, 0
    for bad in (0, 5, 20, 60):
        for _ in range(150):
            pop = np.zeros(n, bool); pop[:bad] = True
            v = int(pop[rng.choice(n, m, replace=False)].sum())
            misses += kcert.certify_k(v, m, n, delta) < bad
    assert misses / 600 <= delta + 0.03
    assert kcert.certify_k(0, n, n) == 0 and kcert.certify_k(3, n, n) == 3
    assert kcert.certify_k(2, 50, 1000, 0.05, method="clopper-pearson") >= kcert.certify_k(2, 50, 1000, 0.05)


def test_clopper_pearson_known_value():
    # zero failures in m trials: upper bound = 1 - delta^(1/m)
    assert abs(kcert.clopper_pearson_upper(0, 30, 0.05) - (1 - 0.05 ** (1 / 30))) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_forced_matches_bruteforce(seed):
    kn, A, Y, labels = random_setup(seed); cons = kn.consistent(labels); f = kn.forced(cons)
    for i in range(len(labels)):
        W = kn.worlds[cons[i]]
        for j in range(kn.n_concepts):
            vals = set(W[:, j].tolist())
            assert f[i, j] == (vals.pop() if len(vals) == 1 else -1)
    assert 0 <= kn.forced_fraction(cons) <= 1


def test_constructors():
    # traffic-law style rule base: (red, round, number) -> action
    def decide(v):
        red, rnd, num = v
        return "limit" if (red and rnd and num) else ("stop" if red and not rnd else "info")
    worlds = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]
    kn = kcert.Knowledge.from_rules(worlds, decide, concept_names=["red", "round", "number"])
    assert kn.n_worlds == 4 and set(kn.label_names) == {"limit", "stop", "info"}
    cons = kn.consistent(["limit", "info"])
    assert cons[0].sum() == 1 and cons[1].sum() == 2
    assert np.array_equal(kn.forced(cons)[0], [1, 1, 1])
    assert kn.check(np.array([[1, 1, 1], [1, 1, 0]]), ["limit", "limit"]).tolist() == [True, False]
    kt = kcert.Knowledge.from_table([[1, 0], [1, 0], [0, 1]], label_names=["a", "b", "c"])
    assert kt.n_worlds == 2 and kt.consistent(["a"])[0].sum() == 1
    kp = kcert.Knowledge.from_predicate(lambda v: v.sum() == 1, 4)
    assert kp.n_worlds == 4
    kn2, outs = kcert.Knowledge.from_program(lambda a, b: a + b, 3, 2)
    vals = np.array([[0, 2], [1, 1]])
    oh = kcert.onehot(vals, 3)
    labels = [outs.index(2), outs.index(2)]
    assert kn2.check(oh, labels).all()


@pytest.mark.parametrize("seed", range(8))
def test_kse_exact_differences_with_full_budget(seed):
    kn, A, Y, labels = random_setup(seed); cons = kn.consistent(labels); err = (A != Y[None]).mean((1, 2))
    idx = kcert.kse_candidates(A, kn, cons)
    est = kcert.kse_estimate(A, kn, cons, idx, Y[idx])
    d_est = est[:, None] - est[None]; d_true = err[:, None] - err[None]
    assert np.allclose(d_est, d_true, atol=1e-12)                                       # model differences are exact
    sub = kcert.kse_sample(A, kn, cons, 5, rng=seed)
    assert len(sub) <= 5 and np.isin(sub, idx).all()


def test_kse_unbiased_differences():
    kn, A, Y, labels = random_setup(4, n=200); cons = kn.consistent(labels); err = (A != Y[None]).mean((1, 2))
    rng = np.random.default_rng(1); ests = []
    for _ in range(400):
        idx = kcert.kse_sample(A, kn, cons, 20, rng=rng)
        ests.append(kcert.kse_estimate(A, kn, cons, idx, Y[idx]))
    m = np.mean(ests, 0)
    assert np.allclose(m - m[0], err - err[0], atol=0.01)


def test_estimators():
    rng = np.random.default_rng(0); Y = rng.integers(0, 2, (100, 4))
    P = np.clip(np.where(Y[None] == 1, 0.8, 0.2) + rng.normal(0, 0.1, (3, 100, 4)), 0, 1)
    for f in (kcert.atc_error(P, Y, P), kcert.doc_error(P, Y, P), kcert.average_confidence_error(P)):
        assert f.shape == (3,) and np.all((0 <= f) & (f <= 1))
    true_src = ((P > .5) != Y[None]).mean((1, 2))
    assert np.allclose(kcert.doc_error(P, Y, P), true_src)                                # no shift -> DoC = source error


@pytest.mark.parametrize("seed", range(6))
def test_audit_report(seed):
    kn, A, Y, labels = random_setup(seed); err = (A != Y[None]).mean((1, 2))
    est = err + np.random.default_rng(seed).normal(0, 0.1, len(err))
    rep = kcert.audit(A, kn, labels, estimate=est, floor_tolerance=0.1)
    assert err[rep.estimate_choice] - err.min() <= rep.certified_regret[rep.estimate_choice] + 1e-9
    assert err[rep.minimax_choice] - err.min() <= rep.certified_regret[rep.minimax_choice] + 1e-9
    assert not np.any(rep.flagged & (err <= 0.1))                                        # no false flag
    s = rep.summary(); assert "minimax choice" in s and len(s.splitlines()) >= len(err) + 3
    rep0 = kcert.audit(A, kn, None)
    assert np.all(rep0.lo <= rep.lo + 1e-12)


def test_input_validation():
    kn, A, Y, labels = random_setup(0)
    with pytest.raises(ValueError):
        kcert.brackets(A.astype(float) * 0.5, kn, kn.consistent(labels))
    with pytest.raises(ValueError):
        kcert.brackets(A, kn, kn.consistent(labels)[:3])
    with pytest.raises(ValueError):
        kcert.Knowledge([[0, 1], [0, 1]])


def test_drift_alarm_object():
    kn, A, Y, labels = random_setup(2)
    res = kcert.drift_alarm(A, kn.consistent(labels), 1 - A, kn)
    alarm, LO, HI = res
    assert alarm.shape == (A.shape[0],) and "certainly degraded" in repr(res)
