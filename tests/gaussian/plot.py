"""Figures for the Gaussian plane.   python tests/gaussian/plot.py

plane.pdf     exact P(pizza before pool) from every cell, calm air and wind 0.3
overview.pdf  20,000 windy Monte Carlo walks: outcome mass, running mean, paths, martingale
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from plane import gaussian_plane, pizza_probability  # noqa: E402

import aniate as an  # noqa: E402
from aniate import vis  # noqa: E402
from aniate.vis.style import INK, axes, labels, number, text, ticks  # noqa: E402

HERE = Path(__file__).resolve().parent
GRAYS = ListedColormap(colormaps["Greys"](np.linspace(0.02, 0.6, 256)))


def board(plane, ax, xlabel):
    p = pizza_probability(plane)
    n = p.shape[0]
    im = ax.imshow(p, cmap=GRAYS, vmin=0, vmax=1, origin="lower")
    for y in range(n):
        for x in range(n):
            label = f"{p[y, x]:.2f}"
            label = "1" if label == "1.00" else "0" if label == "0.00" else label[1:]
            ax.text(x, y, label, ha="center", va="center", **text(4.5))
    ax.plot(n // 2, n // 2, "s", ms=9, mfc="none", mec=INK, mew=0.8)
    ax.set_xticks([0, n - 1], ["pool", "pizza"])
    ax.set_yticks([])
    ticks(ax)
    labels(ax, xlabel)
    return im


def main():
    calm, windy = gaussian_plane(), gaussian_plane(drift=0.3)
    ax = axes(figsize=(7.2, 3.4))
    fig = ax.figure
    fig.delaxes(ax)
    left, right = fig.subplots(1, 2)
    for a in (left, right):
        axes(a)
    board(calm, left, "calm: P(pizza first) = 0.5")
    im = board(windy, right, f"wind 0.3: P(pizza first) = {windy.initial @ windy.evaluate(None):.3f}")
    bar = fig.colorbar(im, ax=[left, right], fraction=0.025, pad=0.02)
    bar.outline.set_linewidth(0.6)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(number))
    ticks(bar.ax)
    vis.save(fig, HERE / "plane.pdf")
    vis.save(vis.overview(an.simulate(windy, None, episodes=20_000, seed=1)), HERE / "overview.pdf")


if __name__ == "__main__":
    main()
