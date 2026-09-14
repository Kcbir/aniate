"""Monte Carlo on a Gaussian plane: does a wandering robot reach the pizza or the pool first?

The robot starts in the middle of an n x n plane and takes discretised Gaussian steps
(offsets -2..2 in each direction, weight exp(-|d - wind|^2 / 2 sigma^2)), clipped at the
edges.  The west column is a pool, the east column a pizza; both end the episode.  Reward is
1 for reaching the pizza, so the value of a cell is exactly P(pizza before pool) from there.
"""

from __future__ import annotations

import math

import numpy as np

import aniate as an


def gaussian_plane(n=11, sigma=1.0, drift=0.0):
    """The plane as an MDP with one action, 'wander'.  ``drift`` is an eastward wind."""
    cells = [(x, y) for x in range(n) for y in range(n)]
    offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    w = np.array([math.exp(-((dx - drift) ** 2 + dy ** 2) / (2 * sigma ** 2)) for dx, dy in offsets])
    w /= w.sum()

    def transition(s, a):
        out = {}
        for (dx, dy), p in zip(offsets, w):
            t = (min(max(s[0] + dx, 0), n - 1), min(max(s[1] + dy, 0), n - 1))
            out[t] = out.get(t, 0.0) + p
        return out

    return an.from_functions(cells, ["wander"], transition,
                             reward=lambda s, a, s2: 1.0 if s2[0] == n - 1 else 0.0, gamma=1.0,
                             terminal=lambda s: s[0] in (0, n - 1), initial=(n // 2, n // 2))


def pizza_probability(plane):
    """Exact P(pizza first) for every cell, as an (n, n) board indexed [y, x].

    Inside the plane this is the value.  On the edges the episode is already over
    (value 0), so the pool column is 0 and the pizza column is 1 by definition.
    """
    v = plane.evaluate(None)
    n = int(round(math.sqrt(plane.S)))
    board = np.zeros((n, n))
    for i in range(plane.S):
        x, y = plane.states[i]
        board[y, x] = 1.0 if x == n - 1 else v[i]
    return board
