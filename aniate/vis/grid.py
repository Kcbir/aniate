"""Gridworld view: values in grayscale, policy arrows, and a trajectory."""

from __future__ import annotations

import numpy as np

from aniate.vis.style import ACCENT, INK, axes, number, plain, text, ticks

__all__ = ["grid"]

_ARROWS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def grid(model, solution=None, *, values=None, policy=None, memory=None, episode=None,
         ax=None, annotate=True):
    """Draw a gridworld (any model with ``.layout``, e.g. from ``problems.gridworld``).

    Parameters
    ----------
    solution : Solution or MemoryPolicy, optional
        Supplies both ``values`` and ``policy``.
    values : (S,) array, optional
        Anything per state: values, or ``runs.occupancy()`` for visit frequencies.
    policy : (S,) action indices, optional
    memory : str, optional
        For a non-Markovian model, which memory state's slice to show
        (default: the automaton's initial state).
    episode : Episode, or a sequence of state labels, optional
        Drawn as a path.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    from aniate.mdp.env import _unwrap

    m = _unwrap(model)
    product = hasattr(m, "machine") and hasattr(m, "base")
    world = m.base if product else m
    layout = getattr(m, "layout", None) or getattr(world, "layout", None)
    if layout is None:
        raise ValueError("vis.grid needs a model with a .layout (build it with problems.gridworld); "
                         "for other models use vis.graph")

    if solution is not None:
        sol = getattr(solution, "solution", None) or solution
        if values is None and hasattr(sol, "value"):
            values = sol.value
        if policy is None and hasattr(sol, "policy"):
            policy = sol.policy
    q = None
    if product:
        q = m.machine.index(memory if memory is not None else m.machine.initial)
    elif memory is not None:
        raise ValueError("memory= only applies to non-Markovian models")

    def world_slice(vec, name):
        vec = np.asarray(vec)
        if vec.ndim != 1:
            raise ValueError(f"vis.grid needs a stationary {name}; index a time step first")
        if product and vec.size == m.S:
            return m.by_memory(vec)[q]
        if vec.size != world.S:
            raise ValueError(f"{name} has {vec.size} entries; expected {world.S}")
        return vec

    from matplotlib import colormaps

    rows, cols, coords = layout["rows"], layout["cols"], layout["coords"]
    ax = axes(ax, figsize=(0.55 * cols + 1.2, 0.55 * rows + 0.3))
    # Light grays only, so black arrows and numbers stay readable on every cell.
    grays = ListedColormap(colormaps["Greys"](np.linspace(0.02, 0.5, 256)))

    v = world_slice(values, "values") if values is not None else None
    if v is not None:
        board = np.full((rows, cols), np.nan)
        for i, (r, cc) in enumerate(coords):
            board[r, cc] = v[i]
        im = ax.imshow(board, cmap=grays, origin="upper")
        bar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        bar.outline.set_linewidth(0.6)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(number))
        ticks(bar.ax)
    else:
        ax.imshow(np.zeros((rows, cols)), cmap=ListedColormap(["white"]), origin="upper")

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=INK, lw=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(INK)
        side.set_linewidth(0.6)

    for r, cc in layout.get("walls", []):
        ax.add_patch(Rectangle((cc - 0.5, r - 0.5), 1, 1, color="0.35", zorder=2))
    for r, cc in layout.get("goals", []):
        ax.plot(cc, r, "*", ms=11, color=INK, zorder=3)
    for r, cc in layout.get("traps", []):
        ax.plot(cc, r, "x", ms=8, mew=1.2, color=INK, zorder=3)
    if "start" in layout:
        sr, sc = layout["start"]
        ax.plot(sc, sr, "s", ms=8, mfc="none", mec=INK, mew=0.8, zorder=3)

    stops = {tuple(x) for x in layout.get("goals", [])} | {tuple(x) for x in layout.get("traps", [])}
    if policy is not None:
        pol = world_slice(policy, "policy")
        for i, (r, cc) in enumerate(coords):
            if (r, cc) in stops:
                continue
            name = str(world.actions[int(pol[i])])
            dx, dy = _ARROWS.get(name, (0, 0))
            if dx or dy:
                ax.annotate("", xy=(cc + 0.22 * dx, r - 0.08 + 0.22 * dy),
                            xytext=(cc - 0.22 * dx, r - 0.08 - 0.22 * dy),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=7), zorder=4)
            else:
                ax.text(cc, r - 0.08, plain(name), ha="center", va="center", zorder=4, **text(5.5))

    if annotate and v is not None and rows * cols <= 100:
        for i, (r, cc) in enumerate(coords):
            ax.text(cc, r + 0.3, number(round(float(v[i]), 2)), ha="center", va="center", zorder=5,
                    **text(5.5))

    if episode is not None:
        states = episode.states if hasattr(episode, "states") else list(episode)
        at = {world.states[i]: rc for i, rc in enumerate(coords)}
        path = [at[x[0] if product and isinstance(x, tuple) else x] for x in states]
        ys, xs = zip(*path)
        jitter = np.linspace(-0.08, 0.08, len(path))
        ax.plot(np.array(xs) + jitter, np.array(ys) + jitter, "-", color=ACCENT, lw=1.1, zorder=6)

    if product:
        ax.set_xlabel(plain(f"memory: {m.machine.states[q]}"), **text(8))
    return ax
