"""kcert: certified, label-free auditing of the perception layer of rule-based systems.

The rule base that turns facts into decisions also bounds the errors of the model that reads those facts. Under the single
assumption (K) that the world obeys the rules, every function here returns a guarantee, not an estimate.
"""
from .audit import AuditReport, audit
from .bounds import (brackets, certified_pairs, certified_regret, frame_bounds, hamming, minimax_select, pairwise_certificate,
                     shortcut_indistinguishable, top_set)
from .drift import DriftAlarm, drift_alarm
from .estimators import atc_error, average_confidence_error, doc_error
from .knowledge import Knowledge, onehot
from .robust import certify_k, clopper_pearson_upper
from .sampling import kse_candidates, kse_estimate, kse_sample
from .select import Selection, impossible, project, select
from .shortcuts import count_shortcuts, min_repair

__all__ = ["Knowledge", "onehot", "audit", "AuditReport",
           "frame_bounds", "hamming", "brackets", "pairwise_certificate", "certified_regret", "minimax_select", "top_set",
           "certified_pairs", "shortcut_indistinguishable",
           "project", "impossible", "select", "Selection",
           "drift_alarm", "DriftAlarm",
           "certify_k", "clopper_pearson_upper",
           "kse_candidates", "kse_sample", "kse_estimate",
           "atc_error", "doc_error", "average_confidence_error",
           "count_shortcuts", "min_repair"]
__version__ = "0.2.0"
