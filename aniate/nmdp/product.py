"""The product construction: a non-Markovian problem as an ordinary MDP.

For a step ``s --a--> s'`` with events ``sigma = on(s, a) + at(s')``:

    P'((s', q') | (s, q), a) = P(s' | s, a)        if q' = delta(q, sigma), else 0
    r'((s, q), a, (s', q'))  = r(s, a, s') + out(q, sigma)     (combine="add")

Once the world is in a terminal state the episode is over, so ``(s, q)``
absorbs with zero reward and the machine reads nothing more; with
``violation_terminal`` a non-accepting sink of the machine ends it too.

Product states are ordered ``q * S + s`` so each memory state's slice of a
vector is contiguous: ``by_memory(v)`` is a reshape.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from aniate import log as _log
from aniate.mdp.core import MDP, _entries

__all__ = ["ProductMDP", "product"]


class ProductMDP(MDP):
    """An MDP over ``(world state, memory state)`` pairs, remembering how it was built."""

    def __init__(self, P, R, gamma, *, base, machine, labels, **kwargs):
        self.base = base
        self.machine = machine
        self.labels = labels
        self.nS = base.S
        self.nQ = len(machine.states)
        self._skip_reachability = True      # unreachable (s, q) pairs are expected
        self._quiet_build = True            # NMDP logs the build instead
        super().__init__(P, R, gamma, **kwargs)

    def flat(self, state, memory):
        """Product position of ``(world state label, memory name)``."""
        return self.machine.index(memory) * self.nS + self.base.index(state)

    def unflat(self, i):
        """``(world state label, memory name)`` of a product position."""
        q, s = divmod(int(i), self.nS)
        return self.base.states[s], self.machine.states[q]

    def step_events(self, i, a, j):
        """Events the machine reads on the product step ``i --a--> j``."""
        return self.labels.step_events(int(i) % self.nS, a, int(j) % self.nS) & frozenset(self.machine.events)

    def by_memory(self, vector):
        """Reshape a product-sized vector to ``(memory states, world states)``."""
        return np.asarray(vector).reshape(self.nQ, self.nS)

    def policy_disagreements(self, policy, k=8):
        """World states where the chosen action depends on memory: ``(examples, count)``.

        Only reachable, non-terminal ``(state, memory)`` pairs are compared.
        """
        from aniate.mdp.validate import reachable

        pol = policy.policy if hasattr(policy, "policy") else np.asarray(policy)
        if pol.ndim != 1:
            raise ValueError("policy_disagreements needs a stationary policy")
        table = self.by_memory(pol)
        live = self.by_memory(reachable(self) & ~self.terminal)
        found = []
        for s in range(self.nS):
            qs = np.flatnonzero(live[:, s])
            if len({int(table[q, s]) for q in qs}) > 1:
                found.append({"state": self.base.states[s],
                              "action_by_memory": {self.machine.states[q]: self.actions[int(table[q, s])]
                                                   for q in qs}})
        return found[:k], len(found)

    def describe(self):
        d = super().describe()
        d.update(world_states=self.nS, memory_states=self.nQ, machine=self.machine.name)
        return d

    def __repr__(self):
        return (f"ProductMDP(world S={self.nS} x memory Q={self.nQ} = {self.S}, A={self.A}, "
                f"gamma={self.gamma}, machine={self.machine.name!r})")


def product(world, machine, labels, *, combine="add", violation_terminal=False):
    """Cross a world model with a Mealy machine.  Used by :class:`NMDP`."""
    if labels.mdp is not world:
        raise ValueError("these labels were built for a different world model")
    if combine not in ("add", "replace"):
        raise ValueError(f"combine must be 'add' or 'replace'; got {combine!r}")
    missing = set(machine.events) - set(labels.events)
    if missing:
        raise ValueError(f"the machine reads {sorted(missing)}, which the labels never make true; "
                         "declare them with Labels.at / on / action")

    S, A, Q = world.S, world.A, len(machine.states)
    at = labels.at_bits(machine.events)
    on = labels.on_bits(machine.events)
    delta = np.array([[machine.index(machine.delta[(q, sym)]) for sym in machine.alphabet]
                      for q in machine.states], dtype=np.int64)
    out = np.array([[machine.out[(q, sym)] for sym in machine.alphabet] for q in machine.states])

    starts = np.flatnonzero(world.initial > 0)
    if at[starts].any():
        names = [str(world.states[s]) for s in starts[at[starts] > 0][:3]]
        _log.warning("nmdp", f"events at start state(s) {', '.join(names)} are not read at time 0 "
                             "(events are read on arrival)")

    terminal = np.tile(world.terminal, Q)
    if violation_terminal and machine.accepting:
        for q in machine.states:
            if q not in machine.accepting and machine.is_sink(q):
                i = machine.index(q)
                terminal[i * S:(i + 1) * S] = True
    stop = np.flatnonzero(terminal)

    P, R = [], []
    for a in range(A):
        rows, cols = _entries(world.P[a])
        prob, r_world = world.P[a].data, world.r_next[a]
        sym = on[rows, a] | at[cols]
        pr, pc, pp, rr = [stop], [stop], [np.ones(stop.size)], [np.zeros(stop.size)]
        for q in range(Q):
            src = q * S + rows
            keep = ~terminal[src]
            reward = out[q, sym] + (r_world if combine == "add" else 0.0)
            pr.append(src[keep])
            pc.append(delta[q, sym][keep] * S + cols[keep])
            pp.append(prob[keep])
            rr.append(reward[keep])
        coords = (np.concatenate(pr), np.concatenate(pc))
        P.append(sp.csr_array((np.concatenate(pp), coords), shape=(S * Q, S * Q)))
        R.append(sp.csr_array((np.concatenate(rr), coords), shape=(S * Q, S * Q)))

    initial = np.zeros(S * Q)
    q0 = machine.index(machine.initial)
    initial[q0 * S:(q0 + 1) * S] = world.initial
    states = [(world.states[s], q) for q in machine.states for s in range(S)]

    m = ProductMDP(P, R, world.gamma, base=world, machine=machine, labels=labels,
                   states=states, actions=world.actions, initial=initial,
                   terminal=terminal, horizon=world.horizon)
    m.layout = world.layout
    return m
