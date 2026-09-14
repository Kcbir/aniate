"""Memory policies: call with a world state, get an action; memory advances inside.

    pi.reset()
    a = pi.step(state)          # every step, with the state you are in
    pi.memory                   # 'refuels_used=2' -- a name, never an index
    pi.explain()                # what that memory state means, in words
"""

from __future__ import annotations

import numpy as np

__all__ = ["MemoryPolicy"]


class MemoryPolicy:
    """A policy over world states that carries named memory."""

    def __init__(self, product, solution):
        self.product = product
        self.machine = product.machine
        self.labels = product.labels
        self.world = product.base
        self.solution = solution if hasattr(solution, "policy") else None
        table = np.asarray(solution.policy if hasattr(solution, "policy") else solution, dtype=int)
        if table.shape[-1] != product.S or table.ndim not in (1, 2):
            raise ValueError(f"expected a product policy over {product.S} states; got shape {table.shape}")
        self._flat = table
        self.table = table.reshape(table.shape[:-1] + (product.nQ, product.nS))
        self.reset()

    def reset(self):
        self._q = self.machine.initial
        self._prev = None
        self.t = 0
        return self

    @property
    def memory(self):
        """Current memory state, by name."""
        return self._q

    def explain(self):
        """Plain-English meaning of the current memory state."""
        return self.machine.meaning.get(self._q, self._q)

    def step(self, state):
        """Action label for ``state``; first updates memory from the previous step."""
        s = self.world.index(state)
        if self._prev is not None:
            ps, pa = self._prev
            if not self.world.terminal[ps]:
                self._q = self.machine.step(self._q, self.labels.step_events(ps, pa, s))[0]
        q = self.machine.index(self._q)
        row = self.table if self.table.ndim == 2 else self.table[self.t]
        a = int(row[q, s])
        self._prev = (s, a)
        self.t += 1
        return self.world.actions[a]

    def action(self, state, memory=None, t=0):
        """Look up an action without advancing anything."""
        q = self.machine.index(self._q if memory is None else memory)
        row = self.table if self.table.ndim == 2 else self.table[t]
        return self.world.actions[int(row[q, self.world.index(state)])]

    def product_table(self, m):
        if m.S != self.product.S:
            raise ValueError("this memory policy belongs to a different model")
        return self._flat

    @property
    def value(self):
        return None if self.solution is None else self.solution.value

    @property
    def start_value(self):
        return None if self.solution is None else self.solution.start_value

    def describe(self):
        d = {"kind": "MemoryPolicy", "machine": self.machine.name, "memory": self._q,
             "meaning": self.explain(), "start_value": self.start_value}
        if self.table.ndim == 2:
            d["states_where_memory_changes_the_action"] = self.product.policy_disagreements(self._flat)[1]
        return d

    def _repr_html_(self):
        from aniate.notebook import to_html

        return to_html(self)

    def __repr__(self):
        extra = ""
        if self.table.ndim == 2:
            _, n = self.product.policy_disagreements(self._flat)
            extra = f", memory changes the action in {n} world states"
        return f"MemoryPolicy({self.machine.name!r}, memory={self._q!r}{extra})"
