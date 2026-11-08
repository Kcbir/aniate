"""aniate: open source combinatorics solvers.

    from aniate import cube, ncube, mdp, nmdp

    nodes = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]   # depot first
    print(cube(nodes, drone=True))

cube plans routes with a metaheuristic, cube.milp with the exact model, and ncube with a neural accelerator.
mdp solves decisions that depend only on the present, and nmdp those that depend on the past.
PDF plots live in ``aniate.vis`` (needs matplotlib).
"""

from aniate import art as _art

_art._on_import()

from aniate import log, mdp, nmdp, problems  # noqa: E402
from aniate.log import verbosity  # noqa: E402
from aniate.mdp import MDP, Builder, Env, check, evaluate, from_functions, simulate, solve  # noqa: E402
from aniate.nmdp import NMDP  # noqa: E402
from aniate.route import Plan, cube, ncube  # noqa: E402
from aniate.selfcheck import ant  # noqa: E402

__version__ = "2.2.0"

__all__ = ["ant", "cube", "ncube", "Plan", "MDP", "NMDP", "Builder", "from_functions", "Env", "simulate", "check",
           "solve", "evaluate", "verbosity", "log", "mdp", "nmdp", "problems", "__version__"]
