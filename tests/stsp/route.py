"""A stochastic TSP: one van, a depot, four customers, and roads that are sometimes blocked.

State: (where the van is, customers served so far).  Action: where to drive next.
Driving costs the distance.  A blocked road costs WAIT and leaves the van where it was,
so the van can retry or go elsewhere.  The episode ends back at the depot with everyone served.
"""

from __future__ import annotations

import itertools
import math

import aniate as an

CUSTOMERS = {"A": (0, 3), "B": (4, 3), "C": (4, 0), "D": (2, -2)}
XY = {"depot": (0, 0), **CUSTOMERS}
FULL = tuple(sorted(CUSTOMERS))
WAIT = 2.0            # cost of an attempt that finds the road blocked
START, DONE = ("depot", ()), ("depot", FULL)


def blocked(u, v):
    """Probability that the road between u and v is closed on an attempt."""
    return 0.35 if {u, v} == {"B", "C"} else 0.10      # the B-C bridge is often closed


def allowed(s, a):
    loc, served = s
    return a != loc and (len(served) == len(FULL) if a == "depot" else a not in served)


def arrive(s, a):
    loc, served = s
    return ("depot", served) if a == "depot" else (a, tuple(sorted(served + (a,))))


def route_model():
    """The exact MDP.  Illegal moves (revisits, going home early) keep the van put and cost 50."""
    states = ([START]
              + [(loc, served) for k in range(1, len(FULL) + 1)
                 for served in itertools.combinations(FULL, k) for loc in served]
              + [DONE])

    def transition(s, a):
        if not allowed(s, a):
            return {s: 1.0}
        p = blocked(s[0], a)
        return {arrive(s, a): 1 - p, s: p}

    def reward(s, a, s2):
        if not allowed(s, a):
            return -50.0
        return -WAIT if s2 == s else -math.dist(XY[s[0]], XY[a])

    return an.from_functions(states, list(XY), transition, reward, gamma=1.0,
                             terminal=[DONE], initial=START)


def expected_tour_cost(tour):
    """A fixed order, retrying each leg until it opens: distance + WAIT * p / (1 - p) per leg."""
    stops = ("depot",) + tuple(tour) + ("depot",)
    return sum(math.dist(XY[u], XY[v]) + WAIT * blocked(u, v) / (1 - blocked(u, v))
               for u, v in zip(stops, stops[1:]))


def brute_force():
    """Expected cost of every one of the 4! fixed tours."""
    return {tour: expected_tour_cost(tour) for tour in itertools.permutations(FULL)}


def follow(solution):
    """The stops a solution visits in order, ending with 'depot' (blocked attempts skipped)."""
    s, order = START, []
    while s != DONE:
        a = solution.action(s)
        order.append(a)
        s = arrive(s, a)
    return order
