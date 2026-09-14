"""aniate: Approximate Numerics in Applied Transition Environments.

Write down a finite decision problem, then simulate it, check it, solve it, and plot it.

    import aniate as an
    m = an.problems.gridworld()
    runs = an.simulate(m, m.solve(), episodes=10_000)

PDF plots live in ``aniate.vis`` (needs matplotlib).
"""

from aniate import art as _art

_art._on_import()

from aniate import log, mdp, nmdp, problems  # noqa: E402
from aniate.log import verbosity  # noqa: E402
from aniate.mdp import MDP, Builder, Env, check, evaluate, from_functions, simulate, solve  # noqa: E402
from aniate.nmdp import NMDP  # noqa: E402
from aniate.selfcheck import ant  # noqa: E402

__version__ = "2.1.0"

__all__ = ["ant", "MDP", "NMDP", "Builder", "from_functions", "Env", "simulate", "check",
           "solve", "evaluate", "verbosity", "log", "mdp", "nmdp", "problems", "__version__"]
