"""Plots, saved as PDF.  Needs matplotlib (``pip install 'aniate[vis]'``).

Black Computer Modern text on white.  Every plot takes ``ax=`` and returns
it; ``save(figure_or_ax, "name.pdf")`` writes the PDF.

Models      ``graph(model, policy=None)``, ``automaton(machine)``, ``grid(model, solution)``
Episodes    ``returns(runs)``, ``paths(runs)``, ``martingale(runs)``, ``convergence(runs)``,
            ``overview(runs)``
"""

from __future__ import annotations

__all__ = ["graph", "automaton", "grid", "returns", "paths", "martingale", "convergence",
           "overview", "save"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("aniate.vis needs matplotlib:  pip install 'aniate[vis]'") from exc
    return plt


from aniate.vis.diagrams import automaton, graph  # noqa: E402
from aniate.vis.grid import grid  # noqa: E402
from aniate.vis.stats import convergence, martingale, overview, paths, returns  # noqa: E402
from aniate.vis.style import save  # noqa: E402
