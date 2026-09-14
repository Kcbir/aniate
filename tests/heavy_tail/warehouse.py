"""A warehouse robot with heavy-tailed jams.

The robot drives L cells to a shelf.  Every move jams with probability p; a jam lasts
k steps with P(k) proportional to k^-alpha for k = 1..K (a truncated power law).  Each step
costs 1.  State: (cell, jam steps left).  The expected cost has a closed form:

    E[steps per cell] = (1 + p E[k]) / (1 - p),    V(start) = -L (1 + p E[k]) / (1 - p)
"""

from __future__ import annotations

import numpy as np

import aniate as an


def jam_lengths(K=100, alpha=1.5, leak=0.0):
    """(k, P(k)) for the truncated power law; ``leak`` removes that much probability mass."""
    ks = np.arange(1, K + 1)
    pk = ks ** -alpha
    return ks, pk / pk.sum() * (1 - leak)


def warehouse_robot(L=10, p=0.05, K=100, alpha=1.5, leak=0.0):
    """Returns (model, ks, pk).  A nonzero ``leak`` makes the model refuse to build."""
    ks, pk = jam_lengths(K, alpha, leak)
    states = [(x, j) for x in range(L) for j in range(K + 1)] + [(L, 0)]

    def transition(s, a):
        x, j = s
        if j > 0:
            return {(x, j - 1): 1.0}
        out = {(x + 1, 0): 1 - p}
        for k, q in zip(ks, pk):
            out[(x, int(k))] = p * q
        return out

    model = an.from_functions(states, ["drive"], transition, reward=lambda s, a: -1.0, gamma=1.0,
                              terminal=[(L, 0)], initial=(0, 0))
    return model, ks, pk


def closed_form(L=10, p=0.05, K=100, alpha=1.5):
    ks, pk = jam_lengths(K, alpha)
    return -L * (1 + p * float(ks @ pk)) / (1 - p)


class RealRobot:
    """A Gym-style simulator of the same robot, whose true jam rate is ``p`` (pass a wrong one)."""

    def __init__(self, L=10, p=0.10, K=100, alpha=1.5):
        self.L, self.p = L, p
        self.ks, self.pk = jam_lengths(K, alpha)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x, self.j = 0, 0
        return (0, 0), {}

    def step(self, action):
        if self.j > 0:
            self.j -= 1
        elif self.rng.random() < self.p:
            self.j = int(self.rng.choice(self.ks, p=self.pk))
        else:
            self.x += 1
        return (self.x, self.j), -1.0, (self.x, self.j) == (self.L, 0), False, {}
