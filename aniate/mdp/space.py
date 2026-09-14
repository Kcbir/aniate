"""Finite, ordered, labelled sets: the state space and the action space.

A label is whatever the user wrote -- ``"has_stock"``, ``(2, 3)``, ``7`` --
and its position is what the arrays are indexed by.  Everything public in
this library accepts a label; integers are accepted as positions only when no
label is itself an integer, so ``index(3)`` can never mean two things.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Space"]


def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


class Space:
    """A finite set of labels with a fixed order.

    >>> s = Space(["low", "high"], kind="state")
    >>> s.index("high"), s[0], len(s)
    (1, 'low', 2)
    """

    def __init__(self, labels, kind="item"):
        labels = list(labels)
        if not labels:
            raise ValueError(f"the {kind} space is empty")
        lookup = {}
        for i, x in enumerate(labels):
            try:
                hash(x)
            except TypeError:
                raise TypeError(f"{kind} labels must be hashable; got {x!r}") from None
            if x in lookup:
                raise ValueError(f"duplicate {kind} label {x!r}")
            lookup[x] = i
        self.kind = kind
        self._labels = tuple(labels)
        self._lookup = lookup
        self._int_labels = any(_is_int(x) for x in labels)

    @classmethod
    def default(cls, n, prefix, kind):
        return cls([f"{prefix}{i}" for i in range(n)], kind=kind)

    @property
    def labels(self):
        return self._labels

    def index(self, x):
        """Position of a label.  An int is a position only if no label is an int."""
        try:
            return self._lookup[x]
        except (KeyError, TypeError):
            pass
        if _is_int(x) and not self._int_labels:
            i = int(x)
            if 0 <= i < len(self._labels):
                return i
            raise IndexError(f"{self.kind} index {i} out of range [0, {len(self._labels)})")
        raise KeyError(f"unknown {self.kind} {x!r}")

    def __getitem__(self, i):
        return self._labels[i]

    def __iter__(self):
        return iter(self._labels)

    def __len__(self):
        return len(self._labels)

    def __contains__(self, x):
        try:
            return x in self._lookup
        except TypeError:
            return False

    def __eq__(self, other):
        return isinstance(other, Space) and self._labels == other._labels

    def __hash__(self):
        return hash(self._labels)

    def __repr__(self):
        shown = ", ".join(repr(x) for x in self._labels[:4])
        more = f", ... ({len(self)} total)" if len(self) > 4 else ""
        return f"Space({self.kind}s: {shown}{more})"
