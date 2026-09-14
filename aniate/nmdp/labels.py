"""Labelling: which named events are true on each step.

An automaton never looks at the world's states directly; it reads *events*.
The rule for when an event is read is fixed:

    events of the step  s --a--> s'   =   on(s, a)  +  at(s')

``at`` events are read on **arrival** -- you have visited ``A`` when you step
into it, so a goal that ends the episode is still seen.  ``on`` events belong
to taking an action somewhere.  Nothing is read at time 0 and nothing after
the episode ends.  This is the reward-machine convention.
"""

from __future__ import annotations

import numpy as np

from aniate import log as _log

__all__ = ["Labels"]


class Labels:
    """Attach events to a world model.

    >>> labels = Labels(world)
    >>> labels.at("r3c3", "A")                    # arriving in r3c3
    >>> labels.on("r1c1", "north", "refuel")      # moving north from r1c1
    >>> labels.action("jump", "jumped")           # jumping anywhere
    >>> labels.where(lambda s: s.startswith("r0"), "top_row")
    """

    def __init__(self, world):
        self.mdp = world
        self._at = {}          # s -> set of events
        self._on = {}          # (s, a) -> set of events
        self._events = []      # declaration order

    def _register(self, events):
        if not events:
            raise ValueError("name at least one event")
        for e in events:
            if not isinstance(e, str):
                raise TypeError(f"event names must be strings; got {e!r}")
            if e not in self._events:
                self._events.append(e)

    def at(self, state, *events):
        """``events`` are true on arriving in ``state``."""
        self._register(events)
        self._at.setdefault(self.mdp.index(state), set()).update(events)
        return self

    def on(self, state, action, *events):
        """``events`` are true when ``action`` is taken in ``state``."""
        self._register(events)
        self._on.setdefault((self.mdp.index(state), self.mdp.action_index(action)), set()).update(events)
        return self

    def action(self, action, *events):
        """``events`` are true whenever ``action`` is taken."""
        self._register(events)
        a = self.mdp.action_index(action)
        for s in range(self.mdp.S):
            self._on.setdefault((s, a), set()).update(events)
        return self

    def where(self, predicate, *events):
        """``events`` are true on arriving in any state whose label satisfies ``predicate``."""
        self._register(events)
        hits = [s for s, label in enumerate(self.mdp.states) if predicate(label)]
        if not hits:
            _log.warning("labels", f"where() matched no state, so {list(events)} are never true")
        for s in hits:
            self._at.setdefault(s, set()).update(events)
        return self

    @property
    def events(self):
        return tuple(self._events)

    def step_events(self, s, a, s_next):
        """Events true on the step ``s --a--> s_next`` (positions)."""
        return frozenset(self._on.get((int(s), int(a)), ())) | frozenset(self._at.get(int(s_next), ()))

    def at_bits(self, events):
        """``(S,)`` bitmask of ``at`` events, bit ``i`` for ``events[i]``."""
        bit = {e: 1 << i for i, e in enumerate(events)}
        out = np.zeros(self.mdp.S, dtype=np.int64)
        for s, group in self._at.items():
            out[s] = sum(bit[e] for e in group if e in bit)
        return out

    def on_bits(self, events):
        """``(S, A)`` bitmask of ``on`` events."""
        bit = {e: 1 << i for i, e in enumerate(events)}
        out = np.zeros((self.mdp.S, self.mdp.A), dtype=np.int64)
        for (s, a), group in self._on.items():
            out[s, a] = sum(bit[e] for e in group if e in bit)
        return out

    def summary(self):
        lines = [f"Labels: events {list(self._events)}  (a step s -a-> s' reads on(s, a) + at(s'))"]
        for e in self._events:
            ats = [str(self.mdp.states[s]) for s, g in self._at.items() if e in g]
            ons = [f"{self.mdp.states[s]}/{self.mdp.actions[a]}" for (s, a), g in self._on.items() if e in g]
            parts = []
            if ats:
                parts.append("arriving in " + ", ".join(ats[:6]) + (" ..." if len(ats) > 6 else ""))
            if ons:
                parts.append("doing " + ", ".join(ons[:6]) + (" ..." if len(ons) > 6 else ""))
            lines.append(f"  {e}: " + "; ".join(parts))
        return "\n".join(lines)

    def describe(self):
        return {
            "kind": "Labels",
            "events": list(self._events),
            "at": {e: [str(self.mdp.states[s]) for s, g in self._at.items() if e in g] for e in self._events},
            "on": {e: [f"{self.mdp.states[s]} / {self.mdp.actions[a]}" for (s, a), g in self._on.items() if e in g]
                   for e in self._events},
        }

    def __repr__(self):
        return f"Labels(events={list(self._events)})"
