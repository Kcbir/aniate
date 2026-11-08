"""Finite Markov decision processes: define, inspect, simulate, check, solve.

    from aniate import mdp

    model = mdp(states, actions, transition, reward, gamma=0.9)
    best = model.solve()
"""

import sys
from types import ModuleType

from aniate.mdp.builder import Builder, from_functions
from aniate.mdp.check import CheckReport, check
from aniate.mdp.core import MDP
from aniate.mdp.env import Env, Episode, Episodes, simulate
from aniate.mdp.solution import Solution
from aniate.mdp.solve import (
    backward_induction,
    evaluate,
    policy_iteration,
    solve,
    value_iteration,
    values_by_time,
)
from aniate.mdp.space import Space

__all__ = [
    "MDP", "Space", "Builder", "from_functions",
    "Env", "simulate", "Episode", "Episodes", "check", "CheckReport",
    "solve", "evaluate", "values_by_time", "policy_iteration", "value_iteration",
    "backward_induction", "Solution",
]


class _Callable(ModuleType):
    def __call__(self, *args, **kwargs):
        """Build a model: the same as `from_functions`."""
        return from_functions(*args, **kwargs)


sys.modules[__name__].__class__ = _Callable
