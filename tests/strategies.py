"""Hypothesis strategies for random small MDPs, plus brute-force ground truth."""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aniate.mdp.core import MDP


def random_mdp(S, A, branching, rewards, gamma):
    """Build an MDP from raw draws.  Rows are normalised, so `P` is valid by
    construction -- these tests are about solvers, not about validation."""
    rng = np.random.default_rng(abs(hash((S, A, branching, gamma))) % (2**32))
    P = []
    for a in range(A):
        rows, cols, vals = [], [], []
        for s in range(S):
            k = min(branching, S)
            targets = rng.choice(S, size=k, replace=False)
            w = rng.random(k) + 1e-3
            w /= w.sum()
            rows.extend([s] * k)
            cols.extend(int(t) for t in targets)
            vals.extend(float(x) for x in w)
        mat = sp.csr_array(
            (np.array(vals), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
            shape=(S, S),
        )
        mat.sum_duplicates()
        P.append(mat)
    R = np.asarray(rewards, dtype=float).reshape(S, A)
    return MDP(P, R, gamma, validate=False)


@st.composite
def small_mdps(draw, max_s=4, max_a=3, min_gamma=0.1, max_gamma=0.95):
    """Small enough that every deterministic policy can be enumerated."""
    S = draw(st.integers(min_value=2, max_value=max_s))
    A = draw(st.integers(min_value=2, max_value=max_a))
    branching = draw(st.integers(min_value=1, max_value=min(3, S)))
    gamma = draw(
        st.floats(min_value=min_gamma, max_value=max_gamma,
                  allow_nan=False, allow_infinity=False)
    )
    R = draw(
        hnp.arrays(
            np.float64, (S, A),
            elements=st.floats(min_value=-10, max_value=10,
                               allow_nan=False, allow_infinity=False),
        )
    )
    return random_mdp(S, A, branching, R, gamma)


def brute_force(m):
    """Optimal value by enumerating every deterministic policy.

    ``v*(s) = max_pi v_pi(s)``, and for a discounted finite MDP that pointwise
    maximum is attained by a single deterministic stationary policy -- so the
    elementwise max over all of them *is* ``v*``, with no tie-breaking
    subtleties to get wrong.  Exponential, and deliberately so: this shares no
    code with anything in `aniate.mdp.solve`.
    """
    eye = np.eye(m.S)
    dense = [m.P[a].toarray() for a in range(m.A)]
    v_star = None
    for pi in itertools.product(range(m.A), repeat=m.S):
        P_pi = np.stack([dense[a][s] for s, a in enumerate(pi)])
        r_pi = m.R[np.arange(m.S), list(pi)]
        v = np.linalg.solve(eye - m.gamma * P_pi, r_pi)
        v_star = v if v_star is None else np.maximum(v_star, v)
    # The optimal policy is greedy with respect to v*.
    q = np.stack([m.R[:, a] + m.gamma * (dense[a] @ v_star) for a in range(m.A)], axis=1)
    return v_star, q.argmax(axis=1)
