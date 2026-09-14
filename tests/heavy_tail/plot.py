"""Figures for the heavy-tailed robot.   python tests/heavy_tail/plot.py

tail.pdf      P(trip takes more than t steps) on log-log axes: 200,000 simulated trips, the exact
              distribution from the model, and a Gaussian with the same mean and sd
overview.pdf  20,000 trips: return distribution, running mean, paths, martingale
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from warehouse import warehouse_robot  # noqa: E402

import aniate as an  # noqa: E402
from aniate import vis  # noqa: E402
from aniate.vis.style import ACCENT, GRAY, INK, axes, labels, legend, number, ticks  # noqa: E402

HERE = Path(__file__).resolve().parent
DASHED = (0, (4, 3))


def exact_survival(model, t_max):
    """P(T > t) for t = 0..t_max, by pushing the start distribution through the chain."""
    P = model.P[0].T.tocsr()
    mass = model.initial.copy()
    live = ~model.terminal
    out = np.empty(t_max + 1)
    for t in range(t_max + 1):
        out[t] = mass[live].sum()
        mass = P @ mass
    return out


def tail(model, runs, ax=None):
    ax = axes(ax, figsize=(4.4, 3.2))
    T = np.sort(runs.lengths)
    t = np.arange(int(T.min()), int(T.max()) + 1)
    simulated = 1 - np.searchsorted(T, t, side="right") / T.size
    keep = simulated > 0
    ax.step(t[keep], simulated[keep], where="post", color=ACCENT, lw=1.1)
    exact = exact_survival(model, int(T.max()))
    ax.plot(t[keep], exact[t[keep]], color=INK, lw=0.8, ls=DASHED)
    mu, sd = float(T.mean()), float(T.std())
    gauss = np.array([0.5 * math.erfc((x - mu) / (sd * math.sqrt(2))) for x in t])
    show = gauss > 1e-5
    ax.plot(t[show], gauss[show], color=GRAY, lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(number))
        axis.set_minor_formatter(NullFormatter())
    ticks(ax)
    ax.set_ylim(0.5 / T.size, 1.5)
    labels(ax, "steps t", "P(trip longer than t)")
    legend(ax, [("simulated", ACCENT, "-"), ("exact", INK, DASHED), ("Gaussian, same mean and sd", GRAY, "-")],
           loc="lower left")
    return ax


def main():
    m, _, _ = warehouse_robot()
    # The far tail needs many trips: with 20,000 the curve past t = 100 rests on a handful of them.
    vis.save(tail(m, an.simulate(m, None, episodes=200_000, seed=2)), HERE / "tail.pdf")
    vis.save(vis.overview(an.simulate(m, None, episodes=20_000, seed=2)), HERE / "overview.pdf")


if __name__ == "__main__":
    main()
