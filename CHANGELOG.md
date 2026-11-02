# Changelog

## 0.2.0
- `audit()`: the whole certified audit in one call, with a printable report.
- k-robust brackets and certificates (`k=`, `universe=`) for rule bases that may be wrong on a few inputs, and `certify_k()` to certify k from a small labelled sample (exact hypergeometric or Clopper-Pearson).
- Knowledge constructors for expert systems: `Knowledge.from_rules` (facts -> action), `from_table` (class-attribute or sign catalogue), `from_predicate`; `forced`, `forced_fraction`, `check`.
- Knowledge-stratified labelling: `kse_candidates`, `kse_sample`, `kse_estimate`.
- Baseline estimators `atc_error`, `doc_error`, `average_confidence_error`; `impossible()` flags estimates the rules exclude.
- `drift_alarm` returns a `DriftAlarm` object (still unpacks as `alarm, LO, HI`).
- 110 tests, including brute-force sharpness and validity under injected rule violations; runnable examples; CI.

## 0.1.0
- Brackets, sharp pairwise certificate, certified selection, projection, drift alarm, shortcut count and minimal repair.
