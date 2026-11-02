"""Certified drift alarm (Theorem 5 of the paper): an alarm that cannot be false, whatever the distribution shift.

The rules alone give a floor on the target error (nothing logged on the target), or a sharper floor when the target decisions are
logged; the source decisions give a ceiling on the source error. Floor(target) > ceiling(source) proves the error went up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import frame_bounds


@dataclass
class DriftAlarm:
    alarm: np.ndarray       # (R,) bool: error certainly increased
    target_floor: np.ndarray
    source_ceiling: np.ndarray

    def __repr__(self):
        return f"DriftAlarm({int(self.alarm.sum())} of {len(self.alarm)} models certainly degraded)"

    def __iter__(self):                                                           # allows: alarm, LO, HI = drift_alarm(...)
        return iter((self.alarm, self.target_floor, self.source_ceiling))


def drift_alarm(A_src, cons_src, A_tgt, kn, cons_tgt=None, delta=None) -> DriftAlarm:
    """Per model: True when err(target) > err(source) is certain under (K), whatever the shift.

    A_src, A_tgt: (R, n_src, K), (R, n_tgt, K) predictions. cons_src: source consistency mask (logged decisions).
    cons_tgt: target consistency; None = nothing logged on the target (legal set only).
    delta: if given, Hoeffding margins make the statement about population errors, with probability >= 1 - 2 delta."""
    n_t = np.asarray(A_tgt).shape[-2]
    lo_t, _ = frame_bounds(A_tgt, kn, kn.consistent(None, n_t) if cons_tgt is None else cons_tgt)
    _, hi_s = frame_bounds(A_src, kn, cons_src)
    LO, HI = lo_t.mean(1), hi_s.mean(1)
    if delta is not None:
        LO = LO - np.sqrt(np.log(2 / delta) / (2 * lo_t.shape[1])); HI = HI + np.sqrt(np.log(2 / delta) / (2 * hi_s.shape[1]))
    return DriftAlarm(LO > HI, LO, HI)
