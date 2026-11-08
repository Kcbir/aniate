"""Non-Markovian decision problems: history-dependent rewards and constraints.

"Visit A before B", "refuel at most three times", "finish within 50 steps" --
none of these is Markov in the world's own states.  Declare the history you
care about as an automaton over named events, and ``NMDP`` turns the problem
into an exact product MDP whose memory keeps your names.
"""

import sys
from types import ModuleType

from aniate.nmdp.builders import Budget, Deadline, Ordering
from aniate.nmdp.labels import Labels
from aniate.nmdp.machine import Mealy, compose
from aniate.nmdp.policy import MemoryPolicy
from aniate.nmdp.task import NMDP

__all__ = ["NMDP", "Labels", "Mealy", "compose", "Ordering", "Budget", "Deadline", "MemoryPolicy"]


class _Callable(ModuleType):
    def __call__(self, *args, **kwargs):
        """Build a task: the same as `NMDP`."""
        return NMDP(*args, **kwargs)


sys.modules[__name__].__class__ = _Callable
