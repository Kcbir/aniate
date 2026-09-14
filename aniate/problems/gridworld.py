"""Gridworld: the problem most tabular models start as."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from aniate.mdp.core import MDP

__all__ = ["gridworld"]

_MOVES = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
_PERP = {"north": ("west", "east"), "south": ("west", "east"),
         "east": ("north", "south"), "west": ("north", "south")}


def gridworld(rows=4, cols=4, goals=((0, 3),), traps=(), walls=(), goal_reward=1.0,
              trap_reward=-1.0, step_reward=-0.04, slip=0.2, gamma=0.95, start=(3, 0)):
    """A slippery gridworld with states named ``r{row}c{col}``.

    Each move goes the intended way with probability ``1 - slip`` and to
    either side with ``slip / 2``; bumping into a wall or edge stays put.
    Every move costs ``step_reward``; *entering* a goal or trap also pays
    ``goal_reward`` / ``trap_reward``, and goals and traps end the episode.
    The model carries ``.layout`` for ``vis.grid``.
    """
    walls = {tuple(w) for w in walls}
    goals = {tuple(g) for g in goals}
    traps = {tuple(t) for t in traps}
    coords = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in walls]
    index = {rc: i for i, rc in enumerate(coords)}
    for rc in goals | traps | {tuple(start)}:
        if rc not in index:
            raise ValueError(f"{rc} is a wall or outside the {rows}x{cols} grid")
    S, stop = len(coords), goals | traps

    def move(rc, d):
        nxt = (rc[0] + _MOVES[d][0], rc[1] + _MOVES[d][1])
        return nxt if nxt in index else rc

    P, R = [], []
    for name in _MOVES:
        outcome = {}
        for rc, s in index.items():
            if rc in stop:
                outcome[(s, s)] = [1.0, 0.0]
                continue
            left, right = _PERP[name]
            for d, p in ((name, 1.0 - slip), (left, slip / 2), (right, slip / 2)):
                if p <= 0:
                    continue
                nxt = move(rc, d)
                bonus = goal_reward if nxt in goals else trap_reward if nxt in traps else 0.0
                entry = outcome.setdefault((s, index[nxt]), [0.0, step_reward + bonus])
                entry[0] += p
        keys = list(outcome)
        ij = (np.array([k[0] for k in keys]), np.array([k[1] for k in keys]))
        P.append(sp.csr_array((np.array([outcome[k][0] for k in keys]), ij), shape=(S, S)))
        R.append(sp.csr_array((np.array([outcome[k][1] for k in keys]), ij), shape=(S, S)))

    m = MDP(P, R, gamma, states=[f"r{r}c{c}" for r, c in coords], actions=list(_MOVES),
            initial={f"r{start[0]}c{start[1]}": 1.0},
            terminal=[f"r{r}c{c}" for r, c in sorted(stop)] or None)
    m.layout = {"rows": rows, "cols": cols, "coords": coords, "walls": sorted(walls),
                "goals": sorted(goals), "traps": sorted(traps), "start": tuple(start)}
    return m
