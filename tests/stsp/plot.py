"""Figures for the stochastic TSP.   python tests/stsp/plot.py

route.pdf     the map: every road in light gray, the optimal tour as arrows,
              each leg labelled with its blocking probability
overview.pdf  20,000 simulated tours: cost distribution, running mean, paths, martingale
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from route import XY, blocked, follow, route_model  # noqa: E402

import aniate as an  # noqa: E402
from aniate import vis  # noqa: E402
from aniate.vis.style import ACCENT, INK, LIGHT, axes, text  # noqa: E402

HERE = Path(__file__).resolve().parent


def route_map(solution, ax=None):
    ax = axes(ax, figsize=(4.0, 4.2))
    for u, v in itertools.combinations(XY, 2):
        (x0, y0), (x1, y1) = XY[u], XY[v]
        ax.plot([x0, x1], [y0, y1], color=LIGHT, lw=0.6, zorder=0)
    stops = ["depot", *follow(solution)]
    for u, v in zip(stops, stops[1:]):
        ax.annotate("", xy=XY[v], xytext=XY[u], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.1, mutation_scale=9,
                                    shrinkA=7, shrinkB=7))
        (x0, y0), (x1, y1) = XY[u], XY[v]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{blocked(u, v):.2f}", ha="center", va="center",
                zorder=3, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), **text(7))
    cx = sum(x for x, _ in XY.values()) / len(XY)
    cy = sum(y for _, y in XY.values()) / len(XY)
    for name, (x, y) in XY.items():
        ax.plot(x, y, "s" if name == "depot" else "o", ms=6, mfc="white", mec=INK, mew=0.8, zorder=4)
        dx, dy = x - cx, y - cy                   # label on the far side from the centre, off the arrows
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ax.text(x + 0.45 * dx / norm, y + 0.45 * dy / norm, name, ha="center", va="center", zorder=4,
                **text(8))
    ax.set_aspect("equal")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-2.8, 4.2)
    ax.set_axis_off()
    return ax


def main():
    m = route_model()
    sol = m.solve()
    vis.save(route_map(sol), HERE / "route.pdf")
    vis.save(vis.overview(an.simulate(m, sol, episodes=20_000, seed=0)), HERE / "overview.pdf")


if __name__ == "__main__":
    main()
