# kcert

**Certified, label-free auditing of the perception layer of rule-based systems.**

Many expert systems read facts with a neural model and then apply rules (a driver-assistance system reads a sign's shape, colour and symbol, then applies the traffic law). `kcert` uses those same rules to audit the perception model **without any fact labels**, and every number it returns is a guarantee, not an estimate.

Its single assumption, (K): the world obeys the rules, so every input's true fact vector is a legal combination consistent with the decision that was observed.

```bash
pip install kcert            # NumPy is the only dependency
```

## What it gives you

| question | function | needs | guarantee |
|---|---|---|---|
| How wrong can this model's facts be? | `brackets` | logged decisions (or nothing) | interval contains the true error; sharp |
| Is model r better than model s? | `pairwise_certificate` | logged decisions | `UP[r, s] < 0` is never wrong; sharp |
| Which model should I deploy? | `select`, `audit` | logged decisions, any estimator | regret of the pick is at most its certificate |
| Is my accuracy estimate even possible? | `project`, `impossible` | logged decisions or nothing | projection never increases the error |
| Did the model degrade after deployment? | `drift_alarm` | nothing on the target | cannot raise a false alarm, whatever the shift |
| Did it learn a reasoning shortcut? | `audit(..., floor_tolerance=...)` on inputs outside the training support | logged decisions | every flag is a proof |
| How do I remove the shortcuts? | `count_shortcuts`, `min_repair` | the training support | exact count; provably smallest repair |
| What if my rules are sometimes wrong? | `brackets(k=...)`, `certify_k` | a few labelled inputs | all bounds hold with probability 1 - delta |
| Where should I spend a labelling budget? | `kse_sample`, `kse_estimate` | a label budget | unbiased model differences, far fewer labels |
| Baselines to compare or to certify | `atc_error`, `doc_error`, `average_confidence_error` | source labels | none (that is what the certificates add) |

## Quick start

```python
import numpy as np
import kcert

# 1. The rule base: legal fact vectors and the action the rules attach to each one
FACTS = ["red", "round", "octagon", "triangle", "number"]
SIGNS = {(1, 1, 0, 0, 1): "obey speed limit", (1, 0, 1, 0, 0): "stop",
         (1, 0, 0, 1, 0): "give way", (1, 1, 0, 0, 0): "prohibition"}
kn = kcert.Knowledge.from_rules(list(SIGNS), lambda v: SIGNS[tuple(int(x) for x in v)], concept_names=FACTS)

# 2. A: fact predictions of R models on n inputs, shape (R, n, K), 0/1.  actions: the n logged decisions.
report = kcert.audit(A, kn, actions, names=["model A", "model B", "model C"])
print(report.summary())          # floor, ceiling, certified regret, top set, minimax choice

# 3. Pick with your favourite estimator, certify with the rules
sel = kcert.select(A, kn, kn.consistent(actions), estimate=my_error_estimates)
print(sel)                       # Selection(index=..., regret <= ..., candidates=...)

# 4. Monitor: does a new, unlabelled batch prove that a model got worse?
alarm = kcert.drift_alarm(A, kn.consistent(actions), A_new_batch, kn)
print(alarm)                     # DriftAlarm(k of R models certainly degraded)

# 5. Reasoning shortcuts of additive rules (MNIST-EvenOdd)
support = [(0, 6), (2, 8), (4, 6), (4, 8), (1, 5), (3, 7), (1, 9), (3, 9)]
kcert.count_shortcuts(support)   # 36 maps, i.e. 35 shortcuts
kcert.min_repair(support)        # (1, [((0, 1),), ...]): one extra pair type removes them all
```

Runnable examples: [`examples/traffic_signs.py`](examples/traffic_signs.py), [`examples/reasoning_shortcuts.py`](examples/reasoning_shortcuts.py), [`examples/select_and_certify.py`](examples/select_and_certify.py).

## Describing your rules

| your rules look like | constructor |
|---|---|
| a function from facts to an action, over a list of legal fact vectors | `Knowledge.from_rules(worlds, decide)` |
| a predicate saying which fact vectors are legal (K <= 22) | `Knowledge.from_predicate(legal, K, decide=None)` |
| a class-to-attribute table or a catalogue (several classes may share a design) | `Knowledge.from_table(table, label_names)` |
| a program over categorical values (sum of digits, ...) | `Knowledge.from_program(f, n_values, arity)` with `onehot()` |
| explicit arrays | `Knowledge(worlds, allowed)` |

Useful checks before training anything: `kn.forced_fraction(kn.consistent(actions))` tells you how much the rules pin down (high means tight certificates and little room for shortcuts), and `kn.check(concepts, actions)` tests the rules on a few labelled inputs.

## The theory in brief

Under (K), the truth of input i lies in its fibre V(l_i). For any additive quantity, such as a model's error or the difference of two models' errors, the set of values compatible with the rules is computed input by input, and its ends are attained (partial identification). That gives sharp brackets and a sharp pairwise certificate; the minimax-optimal selector and its certified regret; a projection that repairs any estimator; a drift alarm whose floor uses only the legal set; and, for additive rules, an exact count of reasoning shortcuts with the provably smallest repair. Every statement and proof is written out in the README of the [experiments repository](https://github.com/researchworksofsomeone/kcert-experiments).

## Tests

```bash
pip install .[test]
pytest
```

110 tests, including brute-force checks that the certificate is attained (sharpness), that every bound holds on random rule bases, that k-robust bounds hold under injected rule violations, and that the certified number of violations has the stated coverage.

## Citation

If you use `kcert`, please cite the paper (see [`CITATION.cff`](CITATION.cff)). MIT licence.
