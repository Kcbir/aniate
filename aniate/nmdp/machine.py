"""Mealy machines: the memory a non-Markovian objective needs, with names.

A machine has named states ``Q``, reads a set of events ``sigma`` each step,
moves by ``delta(q, sigma)`` and emits a reward ``out(q, sigma)``.  Both are
tabulated over the full alphabet up front, so the product construction,
drawing and ``summary()`` all read the same table.

The alphabet is ordered by bitmask: symbol ``k`` is the set of events whose
bit is set in ``k``.
"""

from __future__ import annotations

import itertools
import math

__all__ = ["Mealy", "compose", "MAX_EVENTS"]

MAX_EVENTS = 12      # 2**MAX_EVENTS symbols get tabulated


def _alphabet(events):
    if len(events) > MAX_EVENTS:
        raise ValueError(f"{len(events)} events would tabulate 2**{len(events)} symbols; "
                         f"the limit is {MAX_EVENTS}")
    return [frozenset(e for i, e in enumerate(events) if k >> i & 1) for k in range(2 ** len(events))]


def _fmt_symbol(sigma):
    return "+".join(sorted(sigma)) if sigma else "none"


class Mealy:
    """A finite Mealy machine with named states.

    Parameters
    ----------
    name : str
    states : sequence of str
    initial : str
    events : sequence of str
    transition : ``f(q, sigma) -> q'``, with ``sigma`` a frozenset of events
    output : ``f(q, sigma) -> float``, optional (default 0)
    accepting : states where the objective is satisfied, optional
    meaning : ``{state: plain-English description}``, optional
    """

    def __init__(self, name, states, initial, events, transition, output=None,
                 accepting=(), meaning=None):
        self.name = str(name)
        self.states = [str(q) for q in states]
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"{name}: state names must be unique")
        if initial not in self.states:
            raise ValueError(f"{name}: initial state {initial!r} is not in states")
        self.initial = initial
        self.events = tuple(dict.fromkeys(events))
        self.alphabet = _alphabet(self.events)
        self.accepting = set(accepting)
        if self.accepting - set(self.states):
            raise ValueError(f"{name}: accepting states not in states: "
                             f"{sorted(self.accepting - set(self.states))}")
        self.meaning = dict(meaning or {})
        self._index = {q: i for i, q in enumerate(self.states)}

        self.delta, self.out = {}, {}
        for q in self.states:
            for sigma in self.alphabet:
                nxt = transition(q, sigma)
                if nxt not in self._index:
                    raise ValueError(f"{name}: transition({q!r}, {set(sigma) or '{}'}) returned "
                                     f"{nxt!r}, which is not a declared state")
                r = float(output(q, sigma)) if output else 0.0
                if not math.isfinite(r):
                    raise ValueError(f"{name}: output({q!r}, {set(sigma)}) is not finite")
                self.delta[(q, sigma)] = nxt
                self.out[(q, sigma)] = r

    def __len__(self):
        return len(self.states)

    def index(self, q):
        try:
            return self._index[q]
        except KeyError:
            raise KeyError(f"{self.name}: unknown memory state {q!r}") from None

    def step(self, q, sigma):
        """``(next_state, output)`` on reading the event set ``sigma``."""
        sigma = frozenset(sigma) & frozenset(self.events)
        return self.delta[(q, sigma)], self.out[(q, sigma)]

    def run(self, trace):
        """Feed a sequence of event sets; returns ``(final_state, outputs)``."""
        q, outs = self.initial, []
        for sigma in trace:
            q, r = self.step(q, sigma)
            outs.append(r)
        return q, outs

    def is_sink(self, q):
        return all(self.delta[(q, s)] == q for s in self.alphabet)

    def edges(self):
        """``(src, dst, label)`` with parallel symbols merged; the majority reads 'otherwise'."""
        out = []
        for q in self.states:
            groups = {}
            for sigma in self.alphabet:
                groups.setdefault((self.delta[(q, sigma)], self.out[(q, sigma)]), []).append(sigma)
            biggest = max(groups, key=lambda k: len(groups[k]))
            for (dst, r), symbols in groups.items():
                if len(groups) == 1:
                    label = "any"
                elif (dst, r) == biggest and len(symbols) > len(self.alphabet) / 2:
                    label = "otherwise"
                else:
                    label = ", ".join(_fmt_symbol(s) for s in symbols[:3]) + (" ..." if len(symbols) > 3 else "")
                if r:
                    label += f" / {r:+g}"
                out.append((q, dst, label))
        return out

    def summary(self):
        """The machine as a transition table, in words."""
        lines = [f"Mealy {self.name!r}: {len(self.states)} memory states, events {list(self.events)}",
                 f"  initial: {self.initial}"]
        if self.accepting and self.accepting != set(self.states):
            lines.append(f"  accepting: {', '.join(q for q in self.states if q in self.accepting)}")
        edges = self.edges()
        for q in self.states:
            gloss = f"  -- {self.meaning[q]}" if q in self.meaning else ""
            lines.append(f"  {q}{gloss}")
            lines += [f"      on {label:<28} -> {dst}" for src, dst, label in edges if src == q]
        return "\n".join(lines)

    def describe(self):
        return {"kind": "Mealy", "name": self.name, "states": list(self.states), "initial": self.initial,
                "events": list(self.events), "accepting": sorted(self.accepting),
                "meaning": dict(self.meaning)}

    def __repr__(self):
        return f"Mealy({self.name!r}, {len(self.states)} states, events={list(self.events)})"


def compose(machines, name=None):
    """Run machines in lockstep as one; outputs add, names join with ``+``.

    A composite state accepts when every component that has an accepting set accepts.
    """
    machines = list(machines)
    if not machines:
        raise ValueError("compose() needs at least one machine")
    if len(machines) == 1:
        return machines[0]
    events = tuple(dict.fromkeys(e for mm in machines for e in mm.events))
    combos = list(itertools.product(*(mm.states for mm in machines)))
    join = " + ".join
    lookup = {join(c): c for c in combos}

    def project(sigma, mm):
        return frozenset(sigma) & frozenset(mm.events)

    def transition(q, sigma):
        return join(tuple(mm.delta[(p, project(sigma, mm))] for mm, p in zip(machines, lookup[q])))

    def output(q, sigma):
        return sum(mm.out[(p, project(sigma, mm))] for mm, p in zip(machines, lookup[q]))

    has_acc = [bool(mm.accepting) for mm in machines]
    accepting = [] if not any(has_acc) else [
        join(c) for c in combos
        if all(p in mm.accepting for mm, p, h in zip(machines, c, has_acc) if h)
    ]
    meaning = {}
    for c in combos:
        glosses = [mm.meaning[p] for mm, p in zip(machines, c) if p in mm.meaning]
        if glosses:
            meaning[join(c)] = "; ".join(glosses)
    return Mealy(name or " + ".join(mm.name for mm in machines), [join(c) for c in combos],
                 join(tuple(mm.initial for mm in machines)), events, transition,
                 output=output, accepting=accepting, meaning=meaning)
