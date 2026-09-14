"""aniate.gym: models as Gymnasium environments."""

from __future__ import annotations

import pytest

gymnasium = pytest.importorskip("gymnasium")

import aniate as an  # noqa: E402
from aniate import gym, nmdp, problems  # noqa: E402


def corridor():
    return an.from_functions(
        states=range(5), actions=["left", "right"],
        transition=lambda s, a: {max(s - 1, 0): 1.0} if a == "left" else {min(s + 1, 4): 0.8, s: 0.2},
        reward=lambda s, a, s2: 10.0 if s2 == 4 else -1.0,
        gamma=0.9, terminal=[4], initial=0)


def test_spaces_and_an_episode_follow_the_model():
    m = corridor()
    env = gym.make(m)
    assert isinstance(env, gymnasium.Env)
    assert env.observation_space == gymnasium.spaces.Discrete(5)
    assert env.action_space == gymnasium.spaces.Discrete(2)

    obs, info = env.reset(seed=0)
    assert obs == 0 and info["state"] == 0
    right = m.action_index("right")
    assert env.action_label(right) == "right"
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(right)
        assert env.observation_space.contains(obs) and not truncated
        assert reward == (10.0 if obs == 4 else -1.0)
        assert info["action"] == "right"
    assert env.observation_label(obs) == 4
    with pytest.raises(RuntimeError):
        env.step(right)
    with pytest.raises(ValueError):
        env.step(7)


def test_seeds_reproduce_episodes_and_the_gymnasium_checker_passes():
    from gymnasium.utils.env_checker import check_env

    env = gym.make(problems.gridworld(), render_mode="ansi")
    check_env(env, skip_render_check=True)     # render() is tested below; the render-mode sweep needs a spec

    def trajectory(seed):
        obs, _ = env.reset(seed=seed)
        seen = [obs]
        for _ in range(30):
            obs, _, terminated, truncated, _ = env.step(0)
            seen.append(obs)
            if terminated or truncated:
                break
        return seen

    assert trajectory(3) == trajectory(3)
    assert "reward" in env.render()


def test_the_adapter_passes_check_against_its_own_model():
    m = problems.gridworld(gamma=0.9)
    report = an.check(m, env=gym.make(m), episodes=200, seed=0)
    assert report.ok, report


def test_memory_is_hidden_and_registration_reaches_gymnasium_make():
    world = problems.gridworld(rows=3, cols=3, goals=((0, 2),), start=(2, 0))
    labels = nmdp.Labels(world).at("r2c2", "A").at("r0c2", "B")
    task = an.NMDP(world, labels, nmdp.Ordering(["A", "B"], violation_penalty=-1))

    env = gym.make(task)
    assert env.observation_space.n == world.S
    obs, info = env.reset(seed=1, options={"state": "r2c0"})
    assert env.observation_label(obs) == "r2c0" and info["memory"] == task.machine.initial

    gym.register("aniate/TestOrdering-v0", lambda: task)
    made = gymnasium.make("aniate/TestOrdering-v0")
    obs, _ = made.reset(seed=0)
    assert made.observation_space.contains(obs)
    made.close()
