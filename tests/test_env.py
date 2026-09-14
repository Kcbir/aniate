"""The simulator, episodes, and check()."""

from __future__ import annotations

import numpy as np
import pytest

import aniate as an
from aniate import problems


def corridor(gamma=0.9, p=0.8, start=0):
    return an.from_functions(
        range(5), ["left", "right"],
        transition=lambda s, a: ({max(s - 1, 0): 1.0} if a == "left"
                                 else {min(s + 1, 4): p, s: round(1 - p, 12)}),
        reward=lambda s, a, s2: 10.0 if s2 == 4 else -1.0,
        gamma=gamma, terminal=[4], initial=start)


class Corridor:
    """A hand-written simulator of `corridor()`, with switchable bugs.

    Written without the library on purpose: this is the thing a user brings.
    Observations are tuples, so check() has to be told how to read them.
    """

    def __init__(self, p=0.8, goal_reward=10.0, teleport=False, end_early=False,
                 start=0, old_api=False):
        self.p, self.goal_reward, self.teleport = p, goal_reward, teleport
        self.end_early, self.start, self.old_api = end_early, start, old_api
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = self.start
        return ("pos", self.pos) if self.old_api else (("pos", self.pos), {})

    def step(self, action):
        a = ["left", "right"][action]
        if a == "left":
            nxt = max(self.pos - 1, 0)
        elif self.teleport and self.pos == 1:
            nxt = 3
        else:
            nxt = min(self.pos + 1, 4) if self.rng.random() < self.p else self.pos
        reward = self.goal_reward if nxt == 4 else -1.0
        self.pos = nxt
        done = nxt == 4 or (self.end_early and nxt == 3)
        if self.old_api:
            return ("pos", nxt), reward, done, {}
        return ("pos", nxt), reward, done, False, {}


READ = dict(state=lambda obs: obs[1])


# -- Env ----------------------------------------------------------------------

def test_env_follows_the_gym_contract():
    env = problems.gridworld().env(seed=0)
    obs, info = env.reset()
    assert obs == "r3c0" and info["state"] == "r3c0" and not info["terminal"]
    out = env.step("north")
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs in env.states and isinstance(reward, float)
    assert info["action"] == "north" and info["t"] == 1
    assert "north" in env.render()


def test_env_integer_observations_and_actions():
    env = problems.gridworld().env(seed=0, labels=False)
    obs, _ = env.reset()
    assert isinstance(obs, int)
    obs, *_ = env.step(0)
    assert isinstance(obs, int)


def test_env_refuses_to_step_outside_an_episode():
    m = problems.gridworld()
    env = m.env(seed=0)
    with pytest.raises(RuntimeError, match="reset"):
        env.step("north")
    env.reset(state="r1c3")
    while True:
        *_, terminated, truncated, _ = env.step("north")
        if terminated or truncated:
            break
    with pytest.raises(RuntimeError, match="over"):
        env.step("north")
    _, info = env.reset(state="r0c3")          # starting in a terminal state
    assert info["terminal"]


def test_env_is_reproducible_under_a_seed():
    def run(seed):
        env = problems.gridworld().env(seed=seed)
        env.reset()
        return [env.step("east")[0] for _ in range(3)]
    assert run(7) == run(7)


def test_env_truncates_at_the_horizon():
    w = problems.chain()
    m = an.MDP(w.P, w.R, w.gamma, actions=w.actions, horizon=3)
    env = m.env(seed=0)
    env.reset()
    flags = [env.step("forward")[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_env_samples_the_models_probabilities():
    m = problems.gridworld()
    env = m.env(seed=1)
    counts = {}
    n = 20_000
    for _ in range(n):
        env.reset(state="r1c1")
        s2 = env.step("north")[0]
        counts[s2] = counts.get(s2, 0) + 1
    for s2, p in m.transitions("r1c1", "north").items():
        assert abs(counts.get(s2, 0) / n - p) < 5 * np.sqrt(p * (1 - p) / n)


def test_env_emits_transition_rewards_not_their_expectation():
    m = problems.gridworld(slip=0.2)
    env = m.env(seed=3)
    seen = set()
    for _ in range(200):
        env.reset(state="r0c2")
        s2, r, *_ = env.step("east")
        seen.add((s2, round(r, 10)))
    assert ("r0c3", 0.96) in seen and ("r0c2", -0.04) in seen
    assert all(r in (0.96, -0.04) for _, r in seen)


# -- simulate -----------------------------------------------------------------

@pytest.mark.parametrize("form", ["solution", "array", "dict", "function", "stochastic"])
def test_simulated_returns_match_exact_values(form):
    m = problems.gridworld(gamma=0.9)
    sol = m.solve()
    rng = np.random.default_rng(0)
    M = rng.random((m.S, m.A))
    M /= M.sum(axis=1, keepdims=True)
    d = sol.as_dict()
    policy = {"solution": sol, "array": sol.policy, "dict": d,
              "function": lambda s: d[s], "stochastic": M}[form]
    eps = an.simulate(m, policy, episodes=2000, seed=1)
    v0 = float(m.initial @ an.evaluate(m, policy))
    assert abs(eps.mean_return - v0) < 5 * eps.stderr + eps.truncation_bias + 1e-9


def test_simulate_runs_time_indexed_policies():
    w = problems.gridworld(gamma=0.9)
    m = an.MDP(w.P, w.R, w.gamma, states=w.states, actions=w.actions, initial=w.initial,
               terminal=w.terminal, horizon=5)
    sol = m.solve()
    eps = an.simulate(m, sol, episodes=3000, seed=2)
    assert eps.lengths.max() <= 5 and eps.truncation_bias == 0.0
    assert abs(eps.mean_return - sol.start_value) < 5 * eps.stderr + 1e-9


def test_episode_views():
    m = corridor()
    ep = an.simulate(m, {s: "right" for s in range(4)}, episodes=1, seed=0)[0]
    assert ep.terminated and ep.states[-1] == 4 and set(ep.actions) == {"right"}
    assert ep.rewards[-1] == 10.0 and ep.length == len(ep.actions)
    assert "(terminated)" in ep.table()


# -- check --------------------------------------------------------------------

@pytest.mark.parametrize("fn", [problems.gridworld, problems.chain, problems.river_swim])
def test_models_pass_their_own_check(fn):
    m = fn()
    report = an.check(m, episodes=200)
    assert report.ok, report
    assert an.check(m, policy=m.solve(), episodes=200).ok


def test_a_correct_user_simulator_passes_under_many_seeds():
    m = corridor()
    for seed in range(8):
        report = an.check(m, env=Corridor(), seed=seed, **READ)
        assert report.ok, report
        # the four 'right' rows are stochastic; deterministic 'left' rows are
        # covered by the impossible-transition check instead
        assert report.stats["pairs_tested"] == 4


def test_old_gym_step_api_is_understood():
    assert an.check(corridor(), env=Corridor(old_api=True), **READ).ok


@pytest.mark.parametrize("bug, code", [
    (dict(p=0.65), "transition_frequency"),
    (dict(teleport=True), "impossible_transition"),
    (dict(goal_reward=9.0), "reward_mismatch"),
    (dict(end_early=True), "termination_mismatch"),
    (dict(start=1), "impossible_start"),
])
def test_check_catches_each_kind_of_simulator_bug(bug, code):
    report = an.check(corridor(), env=Corridor(**bug), **READ)
    assert not report.ok
    assert code in {i.code for i in report.issues}, report


def test_check_catches_a_wrong_return_even_when_steps_look_fine():
    """Same dynamics, but the model's discount differs from how returns are scored."""
    model = corridor(gamma=0.5)
    report = an.check(model, env=Corridor(), policy={s: "right" for s in range(4)}, **READ)
    assert report.ok                     # gamma is not observable in single steps...
    lying = corridor(gamma=0.9)
    lying.gamma = 0.5                    # ...but values computed under another gamma are
    lying.R = lying.R * 2
    bad = an.check(lying, env=Corridor(), policy={s: "right" for s in range(4)}, **READ)
    assert "return_mismatch" in {i.code for i in bad.issues}


def test_check_reports_unreadable_observations():
    report = an.check(corridor(), env=Corridor())          # tuples, no state= given
    assert "unknown_observation" in {i.code for i in report.issues}


def test_check_logs_a_verdict_and_every_issue():
    from aniate import log

    model = corridor()
    with log.capture("info") as lines:
        an.check(model, env=Corridor(teleport=True), **READ)
    assert lines[0].startswith("aniate | check    | FAILED with")
    assert any("[impossible_transition]" in line for line in lines[1:])


# -- batched simulation, and what it is for ------------------------------------------

class TableFollower:
    """A policy object with its own step(): forces the one-episode-at-a-time path."""

    def __init__(self, table):
        self.table = table

    def step(self, s):
        return int(self.table[s])


def test_batched_and_sequential_simulation_agree():
    m = problems.gridworld(gamma=0.9)
    sol = m.solve()
    fast = an.simulate(m, sol, episodes=3000, seed=0)
    slow = an.simulate(m, TableFollower(sol.policy), episodes=3000, seed=1)
    se = np.hypot(fast.stderr, slow.stderr)
    assert abs(fast.mean_return - slow.mean_return) < 5 * se
    assert abs(fast.lengths.mean() - slow.lengths.mean()) < 5 * np.hypot(
        fast.lengths.std() / 55, slow.lengths.std() / 55)


def test_ten_thousand_episodes_are_fast_and_right():
    import time

    m = problems.gridworld(rows=6, cols=6, goals=((0, 5),), start=(5, 0), gamma=0.95)
    sol = m.solve()
    t0 = time.perf_counter()
    runs = an.simulate(m, sol, episodes=10_000, seed=0)
    assert time.perf_counter() - t0 < 5.0
    assert abs(runs.mean_return - sol.start_value) < 5 * runs.stderr
    ep = runs[17]
    assert np.isclose(ep.discounted_return, runs.returns[17]) and ep.length == runs.lengths[17]
    assert np.isclose(runs.occupancy().sum(), 1.0)


def test_the_value_martingale_is_flat_when_the_model_is_right():
    m = problems.gridworld(gamma=0.9)
    rng = np.random.default_rng(0)
    M_policy = rng.random((m.S, m.A))
    M_policy /= M_policy.sum(axis=1, keepdims=True)
    runs = an.simulate(m, M_policy, episodes=4000, seed=3)
    M = runs.martingale()
    v0 = runs.model_value
    for t in (0, 1, 3, 10, M.shape[1] - 1):
        se = M[:, t].std(ddof=1) / np.sqrt(len(runs))
        assert abs(M[:, t].mean() - v0) < 5 * se + 1e-9, t
    done = runs.terminated
    assert done.any() and np.allclose(M[done, -1], runs.returns[done])   # V = 0 once terminal


def test_the_martingale_needs_an_evaluable_policy():
    runs = an.simulate(problems.chain(), TableFollower(np.zeros(5, int)), episodes=5)
    with pytest.raises(ValueError, match="evaluated exactly"):
        runs.martingale()
    assert runs.model_value is None


def test_env_reports_return_and_discount_and_logs_on_a_thinning_schedule():
    from aniate import log

    m = corridor()
    env = m.env(seed=0)
    with log.capture("info") as lines:
        for _ in range(25):
            env.reset()
            while True:
                _, r, term, trunc, info = env.step("right")
                if term or trunc:
                    break
    assert info["episode"] == 25 and np.isclose(info["discount"], 0.9 ** info["t"])
    done = [int(line.split("episode ")[1].split(" ")[0]) for line in lines]
    assert done == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]


def test_debug_logs_every_step():
    from aniate import log

    env = corridor().env(seed=0)
    with log.capture("debug") as lines:
        env.reset()
        env.step("right")
    assert lines and "t=1" in lines[0] and "-> right ->" in lines[0]
