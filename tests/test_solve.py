"""Evaluation and solvers, against brute force and against each other."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import aniate as an
from aniate import mdp, problems
from tests.strategies import brute_force, small_mdps

SLOW = settings(max_examples=40, deadline=None,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

PROBLEMS = [problems.gridworld, problems.chain, problems.river_swim, problems.inventory]


@given(small_mdps())
@SLOW
def test_solvers_match_brute_force(m):
    v_star, _ = brute_force(m)
    for method in ("policy_iteration", "value_iteration"):
        sol = m.solve(method)
        assert np.allclose(sol.value, v_star, atol=1e-6), method
        assert np.allclose(sol.value, an.evaluate(m, sol.policy), atol=1e-9), method


@given(small_mdps(), st.integers(0, 2**32 - 1))
@SLOW
def test_stochastic_policy_evaluation_matches_dense_algebra(m, seed):
    M = np.random.default_rng(seed).random((m.S, m.A))
    M /= M.sum(axis=1, keepdims=True)
    dense = np.stack([m.P[a].toarray() for a in range(m.A)])
    P_pi = np.einsum("sa,asj->sj", M, dense)
    r_pi = (M * m.R).sum(axis=1)
    v = np.linalg.solve(np.eye(m.S) - m.gamma * P_pi, r_pi)
    assert np.allclose(an.evaluate(m, M), v, atol=1e-8)


@given(small_mdps())
@SLOW
def test_bound_covers_the_true_gap_of_truncated_solves(m):
    """Truncate hard so there is a real gap; the bound must still cover it."""
    v_star, _ = brute_force(m)
    for n in (1, 2, 3, 8):
        sol = mdp.value_iteration(m, tol=0.0, max_iter=n)
        gap = float((v_star - sol.value).max())
        assert gap <= sol.bound * (1 + 1e-9) + 1e-12, (n, gap, sol.bound)


def test_bound_is_not_vacuous():
    exercised = set()
    for fn in PROBLEMS:
        m = fn()
        exact = m.solve()
        for n in (1, 2, 5):
            sol = mdp.value_iteration(m, tol=0.0, max_iter=n)
            gap = float((exact.value - sol.value).max())
            assert gap <= sol.bound * (1 + 1e-9) + 1e-12
            if gap > 1e-6:
                exercised.add(fn.__name__)
                assert sol.bound > 0
    assert len(exercised) >= 3, exercised


def test_value_iteration_does_not_fall_into_the_span_trap():
    """chain: span-stopped iterates sit ~50 away from v* while the policy is right."""
    m = problems.chain()
    assert np.allclose(mdp.value_iteration(m).value, mdp.policy_iteration(m).value, atol=1e-8)


@pytest.mark.parametrize("fn", PROBLEMS)
def test_solvers_agree_on_bundled_problems(fn):
    m = fn()
    pi, vi = m.solve("policy_iteration"), m.solve("value_iteration")
    assert np.allclose(pi.value, vi.value, atol=1e-7)
    assert pi.bound < 1e-9


@pytest.mark.parametrize("fn", PROBLEMS)
def test_long_horizon_approaches_infinite_horizon(fn):
    m = fn()
    T = 400
    finite = mdp.backward_induction(m, horizon=T)
    tail = m.gamma**T * np.abs(m.R).max() / (1 - m.gamma)
    assert np.abs(finite.value - m.solve().value).max() <= tail + 1e-6


def test_finite_horizon_solve_and_evaluate_agree():
    w = problems.gridworld(gamma=0.9)
    m = an.MDP(w.P, w.R, w.gamma, states=w.states, actions=w.actions, initial=w.initial,
               terminal=w.terminal, horizon=6)
    sol = m.solve()
    assert sol.solver == "backward_induction" and sol.policy.shape == (6, m.S) and sol.bound == 0
    assert np.allclose(an.evaluate(m, sol), sol.value)
    assert np.allclose(an.evaluate(m, sol.policy), sol.value)
    # a stationary policy on a finite horizon is evaluated over that horizon
    stationary = an.evaluate(m, sol.policy[0])
    assert np.all(stationary <= sol.value + 1e-12)


def test_gamma_one_with_terminal_states():
    m = an.from_functions(range(4), ["stay", "go"],
                          lambda s, a: {s: 1.0} if a == "stay" else {s + 1: 1.0},
                          reward=lambda s, a: -1.0, gamma=1.0, terminal=[3], initial=0)
    sol = m.solve()
    assert sol.solver == "value_iteration"
    assert np.allclose(sol.value, [-3, -2, -1, 0])
    with pytest.raises(ValueError, match="undefined"):
        an.evaluate(m, np.zeros(4, dtype=int))          # 'stay' never terminates
    with pytest.raises(ValueError, match="gamma < 1"):
        mdp.policy_iteration(m)


def test_policy_forms_are_equivalent():
    m = problems.gridworld()
    sol = m.solve()
    as_dict = sol.as_dict()
    fn = lambda s: as_dict[s]  # noqa: E731
    onehot = np.eye(m.A)[sol.policy]
    for form in (sol, sol.policy, as_dict, fn, onehot):
        assert np.allclose(an.evaluate(m, form), sol.value)
    assert sol.action("r3c0") == as_dict["r3c0"]


def test_solving_logs_the_numbers_that_matter():
    from aniate import log

    m = problems.chain()
    with log.capture("info") as lines:
        m.solve()
    assert len(lines) == 1 and lines[0].startswith("aniate | solve    | policy_iteration | ")
    assert "start value 61.3795" in lines[0] and "bound" in lines[0]


def test_values_by_time_is_constant_for_stationary_and_zero_past_the_horizon():
    m = problems.chain()
    sol = m.solve()
    V = mdp.values_by_time(m, sol, 5)
    assert V.shape == (6, m.S) and np.allclose(V, sol.value)
    h = an.MDP(m.P, m.R, m.gamma, actions=m.actions, horizon=3)
    Vh = mdp.values_by_time(h, h.solve(), 5)
    assert np.allclose(Vh[0], h.solve().value) and np.allclose(Vh[3:], 0.0)


def test_solver_misuse_is_explained():
    m = problems.chain()
    with pytest.raises(ValueError, match="unknown solver"):
        m.solve("magic")
    h = an.MDP(m.P, m.R, m.gamma, horizon=3)
    with pytest.raises(ValueError, match="finite horizon"):
        mdp.policy_iteration(h)
    with pytest.raises(ValueError, match="stochastic policy rows"):
        an.evaluate(m, np.full((m.S, m.A), 0.3))
