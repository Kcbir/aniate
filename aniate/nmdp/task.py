"""NMDP: a decision problem whose rewards or constraints depend on history.

    world  = problems.gridworld(...)               # Markovian dynamics
    labels = nmdp.Labels(world).at("r3c3", "A").at("r0c3", "B")
    task   = aniate.NMDP(world, labels, nmdp.Ordering(["A", "B"]))

An agent that sees only the world state faces a non-Markovian problem: the
same state can call for different actions depending on what came before.
Crossing the world with the automaton restores the Markov property, so the
problem can be simulated, checked and solved exactly -- and the automaton's
named states are the memory the resulting policy carries.
"""

from __future__ import annotations

from aniate import log as _log
from aniate.nmdp.machine import Mealy, compose
from aniate.nmdp.product import product

__all__ = ["NMDP"]


class NMDP:
    """World model + labels + automaton.

    Parameters
    ----------
    world : MDP
    labels : Labels built on ``world``
    spec : Mealy, or a list of them (run in lockstep)
    combine : "add" (world reward + machine output) or "replace" (machine output only)
    violation_terminal : bool
        End the episode when the machine enters a non-accepting sink.
    """

    def __init__(self, world, labels, spec, *, combine="add", violation_terminal=False):
        self.world = world
        self.labels = labels
        self.machine = spec if isinstance(spec, Mealy) else compose(list(spec))
        self.combine = combine
        self.product = product(world, self.machine, labels, combine=combine,
                               violation_terminal=violation_terminal)
        _log.info("nmdp", f"{world.S:,} world states x {len(self.machine)} memory states "
                          f"= {self.product.S:,}", f"events {', '.join(self.machine.events)}",
                  f"machine {self.machine.name}")

    @property
    def states(self):
        return self.world.states

    @property
    def actions(self):
        return self.world.actions

    @property
    def memory(self):
        return list(self.machine.states)

    @property
    def gamma(self):
        return self.world.gamma

    @property
    def horizon(self):
        return self.world.horizon

    def env(self, seed=None, max_steps=None, labels=True, hide_memory=True, log=True):
        """Simulator.  By default observations are world-state labels and memory is hidden."""
        from aniate.mdp.env import Env

        return Env(self.product, seed=seed, max_steps=max_steps, labels=labels,
                   hide_memory=hide_memory, log=log)

    def solve(self, method="auto", **kwargs):
        """Solve exactly; returns a :class:`MemoryPolicy` (``.solution`` has values and bound)."""
        from aniate.nmdp.policy import MemoryPolicy

        pi = MemoryPolicy(self.product, self.product.solve(method, **kwargs))
        if pi.table.ndim == 2:
            _, n = self.product.policy_disagreements(pi.solution)
            _log.info("nmdp", f"memory changes the action in {n} of {self.world.S} world states")
        return pi

    def evaluate(self, policy):
        """Exact value over product states of any policy (memory or memoryless)."""
        return self.product.evaluate(policy)

    def simulate(self, policy=None, episodes=100, **kwargs):
        from aniate.mdp.env import simulate

        return simulate(self.product, policy, episodes, **kwargs)

    def check(self, env=None, policy=None, **kwargs):
        from aniate.mdp.check import check

        return check(self.product, env=env, policy=policy, **kwargs)

    def summary(self, max_states=8):
        p = self.product
        return "\n".join([
            f"NMDP: {self.world.S} world states x {len(self.machine)} memory states "
            f"= {p.S} product states, {self.world.A} actions, gamma={self.gamma:g}",
            f"  rewards: {'world reward + automaton output' if self.combine == 'add' else 'automaton output only'}",
            "", self.labels.summary(), "", self.machine.summary(), "",
            "World " + self.world.summary(max_states=max_states),
        ])

    def describe(self):
        return {"kind": "NMDP", "world": self.world.describe(), "labels": self.labels.describe(),
                "machine": self.machine.describe(), "combine": self.combine,
                "product_states": self.product.S}

    def _repr_html_(self):
        from aniate.notebook import to_html

        return to_html(self)

    def __repr__(self):
        return (f"NMDP(world S={self.world.S}, memory Q={len(self.machine)}, "
                f"A={self.world.A}, machine={self.machine.name!r})")
