"""Non-Markovian problems: automata, labels, the product, memory policies.

An off-by-one in *when* the automaton advances still yields a well-formed MDP
and a plausible policy, so the product is checked against its definition,
against a step-by-step walk of world + machine, and end to end against
simulation.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import aniate as an
from aniate import nmdp, problems

SLOW = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.fixture
def world():
    m = problems.gridworld(rows=4, cols=4, goals=((0, 3),), start=(3, 0), gamma=0.9)
    labels = nmdp.Labels(m).at("r3c3", "A").at("r0c3", "B").on("r1c1", "north", "refuel")
    return m, labels


def line(n=4, gamma=0.9):
    """A deterministic corridor 0..n-1 with nothing to gain by itself."""
    return an.from_functions(range(n), ["left", "right"],
                             lambda s, a: {max(s - 1, 0) if a == "left" else min(s + 1, n - 1): 1.0},
                             gamma=gamma, initial=1)


def a_then_b():
    """Reward 1, once, for reaching B having already visited A."""
    def tr(q, sigma):
        if q == "start" and "A" in sigma:
            return "has_A"
        if q == "has_A" and "B" in sigma:
            return "done"
        return q
    return nmdp.Mealy("A then B", ["start", "has_A", "done"], "start", ["A", "B"], tr,
                      output=lambda q, sigma: 1.0 if q == "has_A" and "B" in sigma else 0.0,
                      meaning={"start": "A not yet visited", "has_A": "A visited; B pays",
                               "done": "reward collected"})


# -- automata -------------------------------------------------------------------

def test_builders_have_the_documented_states():
    assert nmdp.Ordering(["A", "B"]).states == ["before_A", "A_visited_B_pending", "violated"]
    b = nmdp.Budget("refuel", max=3)
    assert b.states == [f"refuels_used={i}" for i in range(4)]
    assert len(nmdp.Deadline(steps=50).states) == 51
    for machine in (nmdp.Ordering(["A", "B"]), b):
        assert all(machine.meaning.get(q) for q in machine.states)


def test_ordering_and_budget_semantics():
    o, fs = nmdp.Ordering(["A", "B"]), frozenset
    assert o.run([fs({"A"}), fs({"B"})])[0] == "A_visited_B_pending"
    assert o.run([fs({"B"}), fs({"A"})]) == ("violated", [-1.0, 0.0])
    b = nmdp.Budget("refuel", max=2, violation_penalty=-7.0)
    assert b.run([fs({"refuel"})] * 4) == ("refuels_used=2", [0.0, 0.0, -7.0, -7.0])


def test_machine_rejects_undeclared_targets():
    with pytest.raises(ValueError, match="not a declared state"):
        nmdp.Mealy("bad", ["x"], "x", ["e"], lambda q, s: "y")


def test_compose_accepts_when_every_component_with_an_accepting_set_does():
    free = nmdp.Mealy("free", ["x"], "x", ["e"], lambda q, s: "x")
    both = nmdp.compose([nmdp.Ordering(["A", "B"]), free])
    assert "before_A + x" in both.accepting and "violated + x" not in both.accepting


# -- labels -----------------------------------------------------------------------

def test_labels_are_read_on_arrival_and_on_action(world):
    m, labels = world
    s, n = m.index("r1c3"), m.action_index("north")
    assert labels.step_events(s, n, m.index("r0c3")) == {"B"}
    assert labels.step_events(m.index("r0c3"), n, s) == frozenset()
    assert labels.step_events(m.index("r1c1"), n, m.index("r0c1")) == {"refuel"}


def test_labels_warn_about_what_can_never_be_read(world):
    from aniate import log

    m, _ = world
    with log.capture() as lines:
        nmdp.Labels(m).where(lambda s: s == "nowhere", "X")
        start_labelled = nmdp.Labels(m).at("r3c0", "A").at("r0c3", "B")
        nmdp.NMDP(m, start_labelled, nmdp.Ordering(["A", "B"]))
    assert any("matched no state" in line for line in lines)
    assert any("not read at time 0" in line for line in lines)
    with pytest.raises(ValueError, match="never make true"):
        nmdp.NMDP(m, nmdp.Labels(m).at("r3c3", "A"), nmdp.Ordering(["A", "B"]))


# -- the product against its definition -------------------------------------------

@pytest.mark.parametrize("spec", [nmdp.Ordering(["A", "B"]), nmdp.Budget("refuel", max=2)])
def test_product_rows_and_rewards_match_the_definition(world, spec):
    m, labels = world
    p = nmdp.NMDP(m, labels, spec).product
    for q in spec.states:
        qi = spec.index(q)
        for s in range(m.S):
            i = qi * m.S + s
            for a in range(m.A):
                row = p.P[a][[i], :].toarray().ravel()
                expected = np.zeros(p.S)
                if m.terminal[s]:
                    expected[i] = 1.0
                    assert np.allclose(row, expected) and p.R[i, a] == 0.0
                    continue
                idx, prob, rew = m._row(s, a)
                for j, pr, r in zip(idx, prob, rew):
                    q2, out = spec.step(q, labels.step_events(s, a, j))
                    k = spec.index(q2) * m.S + j
                    expected[k] += pr
                    assert np.isclose(p.reward(p.states[i], a, p.states[k]), r + out)
                assert np.allclose(row, expected), (q, m.states[s], m.actions[a])


def test_product_is_well_formed_and_terminal_rows_freeze_memory(world):
    m, labels = world
    p = nmdp.NMDP(m, labels, [nmdp.Ordering(["A", "B"]), nmdp.Budget("refuel", max=2)]).product
    assert p.report.ok and not p.report.issues
    goal = m.index("r0c3")
    for qi in range(p.nQ):
        assert p.terminal[qi * m.S + goal]


@given(st.integers(0, 2**31), st.integers(2, 40))
@SLOW
def test_automaton_advances_at_exactly_the_right_moment(seed, steps):
    """Walk world + machine by hand and the product side by side."""
    m = problems.gridworld(rows=4, cols=4, goals=((0, 3),), start=(3, 0), gamma=0.9)
    labels = nmdp.Labels(m).at("r3c3", "A").at("r0c3", "B").at("r2c1", "B")
    spec = nmdp.Ordering(["A", "B"])
    p = nmdp.NMDP(m, labels, spec).product
    rng = np.random.default_rng(seed)
    s, q = m.index("r3c0"), spec.initial
    for _ in range(steps):
        if m.terminal[s]:
            break
        a = int(rng.integers(m.A))
        idx, prob, _ = m._row(s, a)
        s2 = int(rng.choice(idx, p=prob / prob.sum()))
        q2, out = spec.step(q, labels.step_events(s, a, s2))
        i, k = p.flat(m.states[s], q), p.flat(m.states[s2], q2)
        assert p.P[a][i, k] == m.P[a][s, s2]
        assert np.isclose(p.reward(p.states[i], a, p.states[k]), m.reward(s, a, s2) + out)
        s, q = s2, q2


# -- end to end ------------------------------------------------------------------------

def test_goal_labelled_events_are_valued_the_way_they_are_simulated(world):
    """Regression: an event on a terminal goal must fire on arrival, and nothing
    may accrue after the episode ends -- otherwise solver and simulator disagree."""
    m, labels = world
    task = nmdp.NMDP(m, labels, nmdp.Ordering(["A", "B"], violation_penalty=-5.0))
    for policy in (task.solve(), None):
        report = task.check(policy=policy, episodes=600)
        assert report.ok, report


def test_memory_is_necessary_and_the_memory_policy_uses_it():
    w = line()
    labels = nmdp.Labels(w).at(0, "A").at(3, "B")
    task = nmdp.NMDP(w, labels, a_then_b(), combine="replace")
    pi = task.solve()
    best_memoryless = max(float(task.product.initial @ task.evaluate(np.array(t)))
                          for t in itertools.product(range(2), repeat=w.S))
    assert best_memoryless == 0.0                      # no table over world states ever pays
    assert pi.start_value == pytest.approx(0.9**3)     # 1 -> 0 (A), then 1, 2, 3 (B) pays at t=3
    _, n = task.product.policy_disagreements(pi.solution)
    assert n >= 1


def test_memory_policy_tracks_the_same_memory_as_the_simulator(world):
    m, labels = world
    task = nmdp.NMDP(m, labels, [nmdp.Ordering(["A", "B"]), nmdp.Budget("refuel", max=1)])
    pi = task.solve()
    env = task.env(seed=4)
    for _ in range(30):
        obs, info = env.reset()
        pi.reset()
        memory = info["memory"]
        for _ in range(60):
            action = pi.step(obs)
            assert pi.memory == memory
            obs, _, terminated, truncated, info = env.step(action)
            memory = info["memory"]
            if terminated or truncated:
                break


def test_a_user_simulator_with_a_markovian_reward_is_caught():
    """The simulator pays for B regardless of history: steps look right, rewards do not."""
    w = line()
    task = nmdp.NMDP(w, nmdp.Labels(w).at(0, "A").at(3, "B"), a_then_b(), combine="replace")

    class Sim:
        def __init__(self, remembers):
            self.remembers = remembers

        def reset(self, seed=None):
            self.s, self.has_a, self.paid = 1, False, False
            return self.s, {}

        def step(self, a):
            self.s = max(self.s - 1, 0) if a == 0 else min(self.s + 1, 3)
            r = 0.0
            if self.s == 3 and (self.has_a or not self.remembers) and not self.paid:
                r, self.paid = 1.0, self.remembers
            self.has_a |= self.s == 0
            return self.s, r, False, False, {}

    assert task.check(env=Sim(remembers=True), episodes=400).ok
    bad = task.check(env=Sim(remembers=False), episodes=400)
    assert "reward_mismatch" in {i.code for i in bad.issues}


def test_violation_terminal_ends_episodes(world):
    m, labels = world
    task = nmdp.NMDP(m, labels, nmdp.Ordering(["A", "B"]), violation_terminal=True)
    p = task.product
    assert p.terminal[p.flat("r2c2", "violated")]
    assert task.check(episodes=300).ok


def test_summary_and_logs(world):
    from aniate import log

    m, labels = world
    with log.capture("info") as lines:
        task = nmdp.NMDP(m, labels, nmdp.Budget("refuel", max=2))
        pi = task.solve()
    assert lines[0] == ("aniate | nmdp     | 16 world states x 3 memory states = 48 | events refuel | "
                        "machine Budget(refuel <= 2)")
    assert lines[1].startswith("aniate | solve    | policy_iteration")
    assert "memory changes the action in" in lines[2]
    text = task.summary()
    assert "refuels_used=0" in text and "refuel: doing r1c1/north" in text
    assert isinstance(pi.memory, str) and "0 of 2" in pi.explain()


def test_describe_nonmarkovian_objects(world):
    import json

    m, labels = world
    task = nmdp.NMDP(m, labels, nmdp.Ordering(["A", "B"]))
    pi = task.solve()
    for obj in (task, task.machine, labels, task.product, pi):
        d = obj.describe()
        assert d == json.loads(json.dumps(d))
    assert task.describe()["machine"]["states"] == ["before_A", "A_visited_B_pending", "violated"]
    assert pi.describe()["memory"] == "before_A"
    assert task.product.describe()["memory_states"] == 3
