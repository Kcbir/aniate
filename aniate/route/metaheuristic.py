"""The metaheuristic from the TTP-D paper.

It starts from a nearest-neighbour tour, descends through ten neighbourhoods, then keeps moving at random,
accepting worse plans less and less often, and restarting from the best plan whenever it goes cold.
"""
from __future__ import annotations

import math
import random
import time

from .core import _shake_once, check_solution, construct_nn, evaluate, vnd


def _candidates(n):
    """Cap each neighbourhood above 20 customers, so one descent stays bounded."""
    return None if n <= 20 else 30 * n


def _start_heat(P, truck, sorties, z, rng, accept=0.8, samples=80):
    """Pick a starting heat at which about `accept` of worsening moves pass."""
    base = evaluate(P, truck, sorties, z)
    worse = []
    for _ in range(samples):
        out = _shake_once(P, truck, sorties, z, rng)
        if out is not None and out[3] < base:
            worse.append(base - out[3])
    if worse:
        return max(sum(worse) / len(worse) / -math.log(accept), 1e-6)
    return max(abs(base) * 0.05, 1.0)


def run(P, seconds, seed=0, cooling=0.97, floor=1e-4):
    """Search for `seconds` and return the best plan found as (truck, sorties, z, objective)."""
    rng = random.Random(seed)
    start = time.perf_counter()
    deadline = start + seconds
    cap = _candidates(P.n)

    truck, sorties, z = construct_nn(P)
    obj = evaluate(P, truck, sorties, z)
    truck, sorties, z, obj = vnd(P, truck, sorties, z, obj, rng, deadline=deadline, max_cands=cap)
    current = best = (truck, sorties, z, obj)

    moves = max(20, 12 * (P.n + 1))
    hot = _start_heat(P, truck, sorties, z, rng)
    heat, cold = hot, hot * floor

    while time.perf_counter() < deadline:
        for _ in range(moves):
            if time.perf_counter() >= deadline:
                break
            out = _shake_once(P, current[0], current[1], current[2], rng)
            if out is None:
                continue
            change = out[3] - current[3]
            if change >= 0 or rng.random() < math.exp(change / heat):
                current = out
                if out[3] > best[3] + 1e-9:
                    best = out
        heat *= cooling
        if heat < cold:
            heat, current = hot, best

    polished = vnd(P, *best, rng, deadline=deadline + min(seconds * 0.1, 5.0), max_cands=cap)
    if polished[3] > best[3] + 1e-9:
        best = polished
    problem = check_solution(P, best[0], best[1], best[2])
    if problem is not None:
        raise RuntimeError(f"the search returned an infeasible plan: {problem}")
    return best
