"""Defining a model: spaces, rewards, builders, validation, and reporting."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import aniate as an
from aniate import log, problems
from aniate.mdp import MDP, Space


def codes(report):
    return {i.code for i in report.issues}


# -- orientation and Bellman arithmetic ---------------------------------------

def test_rows_are_sources_and_columns_are_successors():
    P = np.zeros((1, 2, 2))
    P[0, 0, 1] = 1.0
    P[0, 1, 1] = 1.0
    m = MDP(P, np.array([[0.0], [1.0]]), gamma=0.5, validate="warn")
    assert m.transitions("s0", "a0") == {"s1": 1.0}
    v = an.evaluate(m, [0, 0])
    assert np.isclose(v[1], 2.0) and np.isclose(v[0], 1.0)


def test_q_is_reward_plus_discounted_expectation():
    m = problems.chain()
    v = np.arange(m.S, dtype=float)
    dense = [m.P[a].toarray() for a in range(m.A)]
    expected = np.stack([m.R[:, a] + m.gamma * dense[a] @ v for a in range(m.A)], axis=1)
    assert np.allclose(m.q_from_v(v), expected)


# -- spaces -------------------------------------------------------------------

def test_space_labels_positions_and_ambiguity_rule():
    s = Space(["low", "high"], kind="state")
    assert s.index("high") == 1 and s.index(1) == 1 and s[0] == "low"
    ints = Space([10, 20, 30], kind="state")
    assert ints.index(20) == 1
    with pytest.raises(KeyError):
        ints.index(1)          # an int is a label here, never a position
    assert Space([(0, 0), (0, 1)]).index((0, 1)) == 1
    with pytest.raises(ValueError, match="duplicate"):
        Space(["a", "a"])
    with pytest.raises(KeyError, match="unknown"):
        s.index("medium")


# -- rewards ------------------------------------------------------------------

def _random_model(seed=0, S=5, A=3):
    rng = np.random.default_rng(seed)
    P = rng.random((A, S, S)) * (rng.random((A, S, S)) < 0.6) + np.eye(S)[None] * 0.05
    P /= P.sum(axis=2, keepdims=True)
    return P, rng.normal(size=(S, A, S))


def test_all_reward_forms_agree():
    P, R3 = _random_model()
    S, A = R3.shape[0], R3.shape[1]
    expected = np.einsum("asj,saj->sa", P, R3)
    m3 = MDP(P, R3, 0.9, validate="warn")
    ml = MDP(P, [sp.csr_array(R3[:, a, :]) for a in range(A)], 0.9, validate="warn")
    m2 = MDP(P, expected, 0.9, validate="warn")
    assert np.allclose(m3.R, expected) and np.allclose(ml.R, expected)
    for a in range(A):
        assert np.allclose(m3.r_next[a], ml.r_next[a])
    assert np.allclose(m3.solve().value, m2.solve().value)
    for s in range(S):
        for a in range(A):
            for j in np.flatnonzero(P[a, s]):
                assert np.isclose(m3.reward(s, a, j), R3[s, a, j])


def test_transposed_rewards_are_rejected_with_a_hint():
    P, _ = _random_model(S=4, A=2)
    with pytest.raises(ValueError, match="pass R.T"):
        MDP(P, np.zeros((2, 4)), 0.9)
    with pytest.raises(ValueError, match="transpose"):
        MDP(P, np.zeros((2, 4, 4)), 0.9)


def test_reward_function_points_to_from_functions():
    m = problems.chain()
    with pytest.raises(TypeError, match="from_functions"):
        MDP(m.P, lambda s, a: 1.0, 0.9)


@pytest.mark.parametrize("kwargs, match", [
    (dict(gamma=1.5), "gamma"), (dict(gamma=-0.1), "gamma"),
    (dict(gamma=0.9, horizon=0), "horizon"), (dict(gamma=0.9, horizon=2.5), "horizon"),
])
def test_invalid_arguments(kwargs, match):
    m = problems.chain()
    with pytest.raises(ValueError, match=match):
        MDP(m.P, m.R, **kwargs)


def test_reward_on_impossible_transition_is_logged_and_refused_on_lookup():
    P = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    R3 = np.zeros((2, 1, 2))
    R3[0, 0, 1] = 5.0                     # the transition 0 -> 1 never happens
    with log.capture() as lines:
        m = MDP(P, R3, 0.9)
    assert any("reward_on_impossible_transition" in line for line in lines)
    with pytest.raises(ValueError, match="probability 0"):
        m.reward("s0", "a0", "s1")


# -- Builder and from_functions -------------------------------------------------

def test_builder_readme_example():
    b = an.Builder(gamma=0.95)
    b.transition("has_stock", "sell", "low_stock", prob=0.7, reward=10)
    b.transition("has_stock", "sell", "has_stock", prob=0.3, reward=10)
    b.transition("low_stock", "sell", "low_stock", prob=1.0, reward=2)
    with log.capture() as lines:
        m = b.build()
    assert any("rewarding_absorbing_state" in line for line in lines)
    assert list(m.states) == ["has_stock", "low_stock"]
    assert m.transitions("has_stock", "sell") == {"has_stock": 0.3, "low_stock": 0.7}
    assert m.reward("has_stock", "sell") == 10.0


def test_builder_names_undeclared_pairs():
    b = an.Builder(gamma=0.9)
    b.transition("a", "x", "b")
    b.transition("b", "y", "a")
    with pytest.raises(ValueError, match="'a'/'y'"):
        b.build()
    m = b.build(missing="stay")
    assert m.transitions("a", "y") == {"a": 1.0} and m.reward("a", "y") == 0.0


def test_builder_repeated_triples_accumulate():
    b = an.Builder(gamma=0.9)
    b.transition("a", "x", "a", prob=0.5, reward=0.0)
    b.transition("a", "x", "a", prob=0.5, reward=4.0)
    m = b.build()
    assert m.transitions("a", "x") == {"a": 1.0} and m.reward("a", "x", "a") == 2.0


def test_builder_terminal_states_absorb_and_are_worth_zero():
    b = an.Builder(gamma=0.9)
    b.transition("start", "go", "done", reward=5.0)
    b.terminal("done")
    m = b.build()
    assert m.terminal[m.index("done")]
    v = m.solve().value
    assert v[m.index("done")] == 0.0 and np.isclose(v[m.index("start")], 5.0)


def test_builder_refuses_transitions_out_of_a_terminal_state():
    b = an.Builder(gamma=0.9)
    b.transition("start", "go", "done", reward=1.0)
    b.transition("done", "go", "start")
    b.terminal("done")
    with pytest.raises(ValueError, match="terminal_not_absorbing"):
        b.build()


def corridor(gamma=0.9, p=0.8):
    return an.from_functions(
        range(5), ["left", "right"],
        transition=lambda s, a: ({max(s - 1, 0): 1.0} if a == "left"
                                 else {min(s + 1, 4): p, s: 1 - p}),
        reward=lambda s, a, s2: 10.0 if s2 == 4 else -1.0,
        gamma=gamma, terminal=[4], initial=0)


def test_from_functions_matches_builder():
    f = corridor()
    b = an.Builder(gamma=0.9)
    for s in range(4):
        b.transition(s, "left", max(s - 1, 0), 1.0, -1.0)
        b.transition(s, "right", min(s + 1, 4), 0.8, 10.0 if s == 3 else -1.0)
        b.transition(s, "right", s, 0.2, -1.0)
    b.terminal(4)
    b.initial(0)
    m = b.build()
    assert f.states == m.states and f.actions == m.actions
    for a in range(2):
        assert np.allclose(f.P[a].toarray(), m.P[a].toarray())
    assert np.allclose(f.R, m.R) and np.array_equal(f.terminal, m.terminal)
    assert f.reward(3, "right", 4) == 10.0 and f.reward(3, "right", 3) == -1.0


def test_from_functions_accepts_two_argument_rewards_and_predicates():
    m = an.from_functions(["a", "b"], ["go"], lambda s, a: {"b": 1.0},
                          reward=lambda s, a: 3.0, gamma=0.5, terminal=lambda s: s == "b")
    assert m.reward("a", "go", "b") == 3.0 and m.terminal.tolist() == [False, True]


def test_from_functions_reports_bad_transition_output():
    with pytest.raises(TypeError, match="must return a dict"):
        an.from_functions([0, 1], ["x"], lambda s, a: [1, 0], gamma=0.9)
    with pytest.raises(ValueError, match="not in states"):
        an.from_functions([0, 1], ["x"], lambda s, a: {7: 1.0}, gamma=0.9)


# -- validation ---------------------------------------------------------------

def test_malformed_models_raise_by_default_and_name_the_culprit():
    P = [np.array([[0.5, 0.2], [0.0, 1.0]])]
    with pytest.raises(ValueError, match=r"row_sum[\s\S]*alpha / go: sums to 0.7"):
        MDP(P, np.zeros((2, 1)), 0.9, states=["alpha", "beta"], actions=["go"])


@pytest.mark.parametrize("P, R, gamma, code", [
    ([[[1.2, -0.2], [0.0, 1.0]]], [[0.0], [0.0]], 0.9, "negative_probability"),
    ([[[0.0, 0.0], [0.0, 1.0]]], [[0.0], [0.0]], 0.9, "dead_end"),
    ([[[np.nan, 1.0], [0.0, 1.0]]], [[0.0], [0.0]], 0.9, "non_finite"),
    ([[[0.0, 1.0], [1.0, 0.0]]], [[1.0], [1.0]], 1.0, "unbounded_value"),
])
def test_validation_errors(P, R, gamma, code):
    with pytest.raises(ValueError, match=code):
        MDP(np.array(P), np.array(R), gamma)
    m = MDP(np.array(P), np.array(R), gamma, validate="warn")
    assert code in codes(m.report) and not m.report.ok


def test_validation_warnings():
    m = MDP(np.array([np.eye(2)]), np.array([[0.0], [3.0]]), 0.9)
    assert "rewarding_absorbing_state" in codes(m.report) and m.report.ok
    m = MDP(np.array([np.eye(2)]), np.zeros((2, 1)), 0.9, initial=[1.0, 0.0], terminal=[])
    assert "unreachable_state" in codes(m.report)


def test_bundled_problems_are_clean():
    for fn in (problems.gridworld, problems.chain, problems.river_swim, problems.inventory):
        assert not fn().report.issues, fn.__name__


# -- reporting ----------------------------------------------------------------

def test_summary_shows_the_whole_definition():
    text = corridor().summary()
    for fragment in ("5 states, 2 actions, gamma=0.9", "actions  : left, right",
                     "initial  : 0", "terminal : 4", "right -> 3", "+10", "4  (terminal)"):
        assert fragment in text, fragment


def test_building_a_model_logs_what_it_is():
    with log.capture("info") as lines:
        corridor()
    assert lines == ["aniate | model    | 5 states | 2 actions | gamma 0.9 | horizon inf | "
                     "14 transitions | valid"]      # 4 left + 8 right + 2 terminal self-loops


def test_describe_is_plain_json_for_every_mdp_object():
    import json

    m = problems.gridworld()
    sol = m.solve()
    objects = (m, m.report, sol, an.simulate(m, sol, episodes=50), an.check(m, episodes=20))
    for obj in objects:
        d = obj.describe()
        assert isinstance(d, dict) and d == json.loads(json.dumps(d))
    assert m.describe()["actions"] == ["north", "south", "east", "west"]
    assert len(sol.describe()["policy"]) == m.S - int(m.terminal.sum())
