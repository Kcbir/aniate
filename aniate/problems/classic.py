"""Chain, RiverSwim and inventory control: small, standard test problems.

Chain and RiverSwim have long thin reward gradients that punish sloppy
solvers and myopic policies; inventory has a non-trivial optimal policy.
"""

from __future__ import annotations

import math

import numpy as np

from aniate.mdp.builder import Builder

__all__ = ["chain", "river_swim", "inventory"]


def _poisson_pmf(lam, k_max):
    """Poisson pmf truncated at ``k_max``, with the tail folded into the last
    cell so the row still sums to one."""
    pmf = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(k_max + 1)])
    pmf[-1] += 1.0 - pmf.sum()
    return pmf


def chain(n=5, slip=0.2, forward_reward=10.0, reset_reward=2.0, gamma=0.95):
    """Strens' chain: `n` states in a line.

    ``forward`` walks toward the far end, which pays ``forward_reward``;
    ``back`` returns to the first state for a small immediate
    ``reset_reward``.  With probability ``slip`` the actions swap.  The point
    is that a myopic policy takes the small reward forever.
    """
    b = Builder(gamma=gamma)
    names = [f"s{i}" for i in range(n)]

    def outcome(i, act):
        if act == "forward":
            j = min(i + 1, n - 1)
            return j, (forward_reward if i == n - 1 else 0.0)
        return 0, reset_reward

    for i in range(n):
        for act in ("forward", "back"):
            j, r = outcome(i, act)
            b.transition(names[i], act, names[j], prob=1.0 - slip, reward=r)
            other = "back" if act == "forward" else "forward"
            j2, r2 = outcome(i, other)
            b.transition(names[i], act, names[j2], prob=slip, reward=r2)
    b.initial(names[0])
    return b.build()


def river_swim(n=6, gamma=0.95, left_reward=5e-3, right_reward=1.0):
    """RiverSwim: paddle upstream for a big payoff or drift to a small one.

    Swimming ``right`` succeeds with probability 0.35, stalls with 0.6, and is
    swept back with 0.05; swimming ``left`` always succeeds.  Nearly every
    tabular method that is tuned badly ends up sitting in the left-hand state.
    """
    b = Builder(gamma=gamma)
    s = [f"s{i}" for i in range(n)]
    for i in range(n):
        b.transition(s[i], "left", s[max(i - 1, 0)], prob=1.0,
                     reward=left_reward if i == 0 else 0.0)
    for i in range(n):
        if i == 0:
            b.transition(s[0], "right", s[1], prob=0.4)
            b.transition(s[0], "right", s[0], prob=0.6)
        elif i == n - 1:
            b.transition(s[i], "right", s[i], prob=0.6, reward=right_reward)
            b.transition(s[i], "right", s[i - 1], prob=0.4, reward=0.0)
        else:
            b.transition(s[i], "right", s[i + 1], prob=0.35)
            b.transition(s[i], "right", s[i], prob=0.6)
            b.transition(s[i], "right", s[i - 1], prob=0.05)
    b.initial(s[0])
    return b.build()


def inventory(max_stock=10, demand_mean=3.0, price=4.0, order_cost=2.0,
              holding_cost=0.5, gamma=0.95):
    """Single-item inventory control with Poisson demand.

    State is the stock on hand; action is the quantity ordered, capped at the
    remaining capacity.  The optimal policy is a base-stock (order-up-to) rule,
    which is a satisfying thing to recover from a solver.
    """
    b = Builder(gamma=gamma)
    pmf = _poisson_pmf(demand_mean, max_stock)
    names = [f"stock={i}" for i in range(max_stock + 1)]
    for stock in range(max_stock + 1):
        for order in range(max_stock + 1):
            q = min(order, max_stock - stock)  # capacity cap
            available = stock + q
            for d, p in enumerate(pmf):
                if p <= 0:
                    continue
                sold = min(available, d)
                nxt = available - sold
                reward = price * sold - order_cost * q - holding_cost * available
                b.transition(names[stock], f"order={order}", names[nxt],
                             prob=p, reward=reward)
    b.initial(names[0])
    return b.build()

