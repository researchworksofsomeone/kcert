"""Exact symbolic knowledge: the legal fact combinations of a rule base and the decisions each one allows.

A rule base is reduced to two finite objects:

* ``worlds``  (W, K) 0/1 array: every legal combination of the K binary facts (the set V),
* ``allowed`` (W, L) bool array: ``allowed[w, l]`` is True when world w is consistent with decision l (the sets V(l)).

Assumption (K), the only assumption behind every guarantee in kcert: the true fact vector of every input is a world
allowed by that input's true decision.
"""
from __future__ import annotations

import itertools
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np


class Knowledge:
    """Legal worlds and, optionally, the decision table.

    Parameters
    ----------
    worlds : array-like (W, K) of 0/1
        One legal fact vector per row.
    allowed : array-like (W, L) of bool, optional
        ``allowed[w, l]`` means world ``w`` is consistent with decision label ``l``.
    concept_names, label_names : optional sequences used only for reports.
    """

    def __init__(self, worlds, allowed=None, concept_names: Sequence[str] | None = None, label_names: Sequence[Hashable] | None = None):
        self.worlds = np.asarray(worlds, dtype=np.int8)
        if self.worlds.ndim != 2 or not np.isin(self.worlds, (0, 1)).all():
            raise ValueError("worlds must be a (W, K) 0/1 array")
        if len(np.unique(self.worlds, axis=0)) != len(self.worlds):
            raise ValueError("worlds must be distinct")
        self.allowed = None if allowed is None else np.asarray(allowed, dtype=bool)
        if self.allowed is not None:
            if self.allowed.ndim != 2 or self.allowed.shape[0] != len(self.worlds):
                raise ValueError("allowed must be a (W, L) array with one row per world")
        self.concept_names = list(concept_names) if concept_names is not None else [f"c{j}" for j in range(self.n_concepts)]
        if len(self.concept_names) != self.n_concepts:
            raise ValueError("concept_names must have one name per fact")
        self.label_names = list(label_names) if label_names is not None else (
            None if self.allowed is None else list(range(self.allowed.shape[1])))

    # ------------------------------------------------------------------ basic properties
    @property
    def n_worlds(self) -> int:
        return len(self.worlds)

    @property
    def n_concepts(self) -> int:
        return self.worlds.shape[1]

    @property
    def n_labels(self) -> int:
        return 0 if self.allowed is None else self.allowed.shape[1]

    def __repr__(self) -> str:
        return f"Knowledge(worlds={self.n_worlds}, facts={self.n_concepts}, decisions={self.n_labels})"

    # ------------------------------------------------------------------ consistency masks
    def label_index(self, labels) -> np.ndarray:
        """Map decision names (as given in ``label_names``) to column indices; integer arrays pass through."""
        labels = np.asarray(labels)
        if labels.dtype.kind in "iu":
            return labels.astype(int)
        pos = {name: i for i, name in enumerate(self.label_names)}
        return np.array([pos[x] for x in labels.tolist()], dtype=int)

    def consistent(self, labels=None, n: int | None = None) -> np.ndarray:
        """(n, W) bool mask of the worlds consistent with each input.

        ``labels`` = observed decisions (indices or names). ``labels=None`` uses the legal set alone (nothing logged); then give ``n``."""
        if labels is None:
            if n is None:
                raise ValueError("give n when labels is None")
            return np.ones((int(n), self.n_worlds), dtype=bool)
        if self.allowed is None:
            raise ValueError("this Knowledge has no decision table; build it with `allowed` or use labels=None")
        idx = self.label_index(labels)
        if idx.min(initial=0) < 0 or idx.max(initial=0) >= self.n_labels:
            raise ValueError("label index out of range")
        return self.allowed[:, idx].T.copy()

    def forced(self, cons) -> np.ndarray:
        """(n, K) int array: the value (0/1) the knowledge forces for each fact of each input, or -1 when it is not forced."""
        cons = np.asarray(cons, bool)
        if not cons.any(1).all():
            raise ValueError("an input has no consistent world: the knowledge contradicts its decision")
        V = self.worlds.astype(np.int8)
        any1 = (cons[:, :, None] & (V[None] == 1)).any(1)
        any0 = (cons[:, :, None] & (V[None] == 0)).any(1)
        return np.where(any1 & ~any0, 1, np.where(any0 & ~any1, 0, -1)).astype(np.int8)

    def forced_fraction(self, cons) -> float:
        """Share of (input, fact) slots that the knowledge forces. Computable before any training: high values (e.g. 0.95 for the
        traffic law) mean tight certificates and little room for reasoning shortcuts; low values mean the opposite."""
        return float(np.mean(self.forced(cons) >= 0))

    def check(self, concepts, labels=None) -> np.ndarray:
        """Bool (n,): does each labelled input satisfy assumption (K)? Use on a few labelled inputs to test the rule base."""
        C = np.asarray(concepts, np.int8)
        match = (C[:, None, :] == self.worlds[None]).all(-1)                    # (n, W)
        if labels is None:
            return match.any(1)
        return (match & self.consistent(labels)).any(1)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_rules(cls, worlds, decide: Callable[[np.ndarray], Hashable | Iterable[Hashable]], concept_names=None) -> "Knowledge":
        """Rule base given as a function from a legal fact vector to its decision (or to the set of decisions it allows).

        This is the usual expert-system form: ``decide(facts) -> action``."""
        worlds = np.asarray(worlds, np.int8)
        outs = []
        for w in worlds:
            d = decide(w)
            outs.append({d} if isinstance(d, (str, bytes, int, np.integer)) or not isinstance(d, Iterable) else set(d))
        labels = sorted({x for o in outs for x in o}, key=lambda x: (str(type(x)), x))
        allowed = np.array([[lab in o for lab in labels] for o in outs], bool)
        return cls(worlds, allowed, concept_names=concept_names, label_names=labels)

    @classmethod
    def from_predicate(cls, legal: Callable[[np.ndarray], bool], n_concepts: int, decide=None, concept_names=None) -> "Knowledge":
        """Enumerate all 2^K fact vectors and keep those for which ``legal(v)`` is True (K <= 22). ``decide`` as in ``from_rules``."""
        if n_concepts > 22:
            raise ValueError("enumeration of 2^K vectors is limited to K <= 22; pass the worlds explicitly")
        cube = np.array(list(itertools.product((0, 1), repeat=n_concepts)), np.int8)
        worlds = cube[np.array([bool(legal(v)) for v in cube])]
        if decide is None:
            return cls(worlds, None, concept_names=concept_names)
        return cls.from_rules(worlds, decide, concept_names=concept_names)

    @classmethod
    def from_table(cls, table, label_names=None, concept_names=None) -> "Knowledge":
        """Class-to-fact table (L, K): decision l has fact vector table[l] (e.g. a sign catalogue or a class-attribute table).
        Several classes may share one fact vector."""
        T = np.asarray(table, np.int8)
        worlds, inv = np.unique(T, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        allowed = np.zeros((len(worlds), len(T)), bool)
        allowed[inv, np.arange(len(T))] = True
        return cls(worlds, allowed, concept_names=concept_names, label_names=label_names)

    @classmethod
    def from_program(cls, program, n_values: int, arity: int, encode: str = "onehot"):
        """Knowledge of a program over ``arity`` categorical concepts with ``n_values`` values each (e.g. the sum of two digits).
        Worlds are one-hot encoded (arity * n_values bits); decisions are the distinct program outputs, sorted.
        Returns ``(Knowledge, output_values)`` so that label index l means output ``output_values[l]``."""
        if encode != "onehot":
            raise ValueError("only one-hot encoding is supported")
        tuples = list(itertools.product(range(n_values), repeat=arity))
        outs = sorted({program(*t) for t in tuples})
        W = np.zeros((len(tuples), arity * n_values), np.int8)
        for i, t in enumerate(tuples):
            for j, v in enumerate(t):
                W[i, j * n_values + v] = 1
        allowed = np.zeros((len(tuples), len(outs)), bool)
        for i, t in enumerate(tuples):
            allowed[i, outs.index(program(*t))] = True
        names = [f"x{j}={v}" for j in range(arity) for v in range(n_values)]
        return cls(W, allowed, concept_names=names, label_names=outs), outs


def onehot(values, n_values: int) -> np.ndarray:
    """(…, arity) integer predictions -> (…, arity * n_values) one-hot fact vectors, matching ``Knowledge.from_program``."""
    v = np.asarray(values, int)
    out = np.zeros(v.shape[:-1] + (v.shape[-1] * n_values,), np.int8)
    idx = v + n_values * np.arange(v.shape[-1])
    np.put_along_axis(out, idx, 1, axis=-1)
    return out
