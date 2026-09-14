"""Stochastic TSP: the exact solution equals brute force, and simulation agrees with it."""

from __future__ import annotations

import json

import pytest
from route import brute_force, follow, route_model

import aniate as an


def test_solution_equals_brute_force_over_every_tour():
    m = route_model()
    sol = m.solve()
    costs = brute_force()
    assert sol.start_value == pytest.approx(-min(costs.values()), abs=1e-6)
    assert costs[tuple(follow(sol)[:-1])] == pytest.approx(min(costs.values()))


def test_simulation_and_check_agree_with_the_model():
    m = route_model()
    sol = m.solve()
    runs = an.simulate(m, sol, episodes=20_000, seed=0)
    assert not runs.truncated.any()
    assert abs(runs.mean_return - sol.start_value) < 5 * runs.stderr
    assert an.check(m, policy=sol, episodes=2000).ok
    json.dumps(runs.describe())
