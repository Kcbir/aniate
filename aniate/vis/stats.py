"""What many episodes look like.

    runs = aniate.simulate(model, policy, episodes=10_000)
    vis.save(vis.overview(runs), "overview.pdf")

* ``returns``      distribution of discounted returns, normal fit, the model's exact value
* ``paths``        return collected so far along sample episodes, with a 10-90% band and the mean
* ``martingale``   ``M_t = G_<t + gamma^t V(s_t)``; its mean stays at the model value
                   exactly when the model's values agree with the simulation
* ``convergence``  running mean return with a 95% band
"""

from __future__ import annotations

import math

import numpy as np

from aniate.vis.style import ACCENT, GRAY, INK, LIGHT, axes, labels, legend, number, ticks

__all__ = ["returns", "paths", "martingale", "convergence", "overview"]

DASHED = (0, (4, 3))


def _runs(runs):
    from aniate.mdp.env import Episodes

    if not isinstance(runs, Episodes):
        raise TypeError("pass the Episodes returned by aniate.simulate(...)")
    return runs


def returns(runs, ax=None, bins=60):
    """Distribution of discounted returns, a normal fit, and the model's exact value."""
    runs = _runs(runs)
    ax = axes(ax)
    x = runs.returns
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    items = [("simulated", ACCENT, "-")]
    values, counts = np.unique(x, return_counts=True)
    if 1 < values.size <= bins:
        # Few distinct returns: probability mass per value, and the normal
        # fit's mass over the same cells, so the two are comparable.
        mids = (values[1:] + values[:-1]) / 2
        edges = np.concatenate([[2 * values[0] - mids[0]], mids, [2 * values[-1] - mids[-1]]])
        ax.vlines(values, 0, counts / x.size, color=ACCENT, lw=2.0)
        if sd > 0:
            cdf = np.array([0.5 * (1 + math.erf((e - mu) / (sd * math.sqrt(2)))) for e in edges])
            ax.plot(values, np.diff(cdf), color=GRAY, lw=0.9, ls=DASHED)
            items.append(("normal fit", GRAY, DASHED))
        labels(ax, "discounted return", "probability")
    else:
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(x, bins=edges, density=True, histtype="stepfilled", color=ACCENT, alpha=0.25, lw=0)
        ax.hist(x, bins=edges, density=True, histtype="step", color=ACCENT, lw=0.8)
        if sd > 0:
            g = np.linspace(lo, hi, 300)
            ax.plot(g, np.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
                    color=GRAY, lw=0.9, ls=DASHED)
            items.append(("normal fit", GRAY, DASHED))
        labels(ax, "discounted return", "density")
    v = runs.model_value
    if v is not None:
        ax.axvline(v, color=INK, lw=0.9)
        items.append(("model value", INK, "-"))
    legend(ax, items)
    return ax


def _fan(ax, curves, lengths, n, band=True):
    """Thin gray sample paths, optionally a 10-90% band; returns t."""
    from matplotlib.collections import LineCollection

    N, W = curves.shape
    t = np.arange(W)
    pick = np.random.default_rng(0).choice(N, size=min(n, N), replace=False)
    segs = [np.column_stack([t[:lengths[i] + 1], curves[i, :lengths[i] + 1]]) for i in pick]
    ax.add_collection(LineCollection(segs, colors=GRAY, linewidths=0.4,
                                     alpha=max(0.05, min(0.5, 20 / len(pick)))))
    if band:
        lo, hi = np.percentile(curves, [10, 90], axis=0)
        ax.fill_between(t, lo, hi, color=LIGHT, lw=0, zorder=0)
    ax.autoscale_view()
    ax.set_xlim(0, max(1, min(W - 1, int(np.percentile(lengths, 99)) + 1)))
    return t


def paths(runs, n=200, ax=None):
    """Discounted return collected so far, step by step."""
    runs = _runs(runs)
    ax = axes(ax)
    G = runs.partial_returns()
    t = _fan(ax, G, runs.lengths, n)
    ax.plot(t, G.mean(axis=0), color=ACCENT, lw=1.2)
    labels(ax, "step", "return so far")
    legend(ax, [("sample paths", GRAY, "-"), ("10-90%", LIGHT, "-"), ("mean", ACCENT, "-")])
    return ax


def martingale(runs, n=200, ax=None):
    """``M_t = G_<t + gamma^t V(s_t)``, whose mean stays at the model value when model and simulation agree."""
    runs = _runs(runs)
    M = runs.martingale()
    ax = axes(ax)
    t = _fan(ax, M, runs.lengths, n, band=False)
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    ax.fill_between(t, mean - 1.96 * se, mean + 1.96 * se, color=ACCENT, alpha=0.2, lw=0)
    ax.plot(t, mean, color=ACCENT, lw=1.2)
    level = runs.model_value if runs.model_value is not None else float(M[:, 0].mean())
    ax.axhline(level, color=INK, lw=0.8, ls=DASHED)
    labels(ax, "step", r"$G_{<t} + \gamma^t V(s_t)$")
    ax.yaxis.label.set_math_fontfamily("cm")
    legend(ax, [("sample paths", GRAY, "-"), ("mean, 95%", ACCENT, "-"), ("model value", INK, DASHED)])
    return ax


def convergence(runs, ax=None):
    """Running mean return as episodes accumulate, with a 95% band."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    runs = _runs(runs)
    ax = axes(ax)
    x = runs.returns
    k = np.arange(1, x.size + 1)
    mean = np.cumsum(x) / k
    var = np.maximum((np.cumsum(x * x) - k * mean ** 2) / np.maximum(k - 1, 1), 0.0)
    band = 1.96 * np.sqrt(var / k)
    ax.fill_between(k, mean - band, mean + band, color=ACCENT, alpha=0.2, lw=0)
    ax.plot(k, mean, color=ACCENT, lw=1.1)
    items = [("running mean, 95%", ACCENT, "-")]
    v = runs.model_value
    if v is not None:
        ax.axhline(v, color=INK, lw=0.8, ls=DASHED)
        items.append(("model value", INK, DASHED))
    if x.size > 1:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(number))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ticks(ax)
        tail = k >= min(x.size, max(10, x.size // 200))
        lo, hi = float((mean - band)[tail].min()), float((mean + band)[tail].max())
        if v is not None:
            lo, hi = min(lo, v), max(hi, v)
        pad = 0.08 * (hi - lo or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    labels(ax, "episodes", "mean return")
    legend(ax, items)
    return ax


def overview(runs, figsize=(8.0, 5.6)):
    """Returns, running mean, paths and martingale as one 2 x 2 figure.  Returns the Figure."""
    from aniate.vis import _plt
    from aniate.vis.style import text

    runs = _runs(runs)
    fig, axs = _plt().subplots(2, 2, figsize=figsize)
    returns(runs, ax=axs[0, 0])
    convergence(runs, ax=axs[0, 1])
    paths(runs, ax=axs[1, 0])
    try:
        martingale(runs, ax=axs[1, 1])
    except ValueError as exc:
        axs[1, 1].set_axis_off()
        axs[1, 1].text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True,
                       transform=axs[1, 1].transAxes, **text(8))
    fig.tight_layout()
    return fig
