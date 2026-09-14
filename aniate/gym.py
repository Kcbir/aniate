"""Gymnasium adapter: any aniate model as a ``gymnasium.Env``.

    from aniate import gym
    env = gym.make(model)                            # a gymnasium.Env
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    gym.register("aniate/Corridor-v0", build_corridor)
    env = gymnasium.make("aniate/Corridor-v0")

Observations and actions are integer positions in ``Discrete`` spaces, which
is what Gymnasium agents expect; ``observation_label`` and ``action_label``
translate them back to the model's labels, and ``info`` carries the labels
as well (the keys of aniate's ``Env``, without its episode counter).
Transitions are sampled from the model itself with Gymnasium's
seeded generator, so ``reset(seed=...)`` makes episodes reproducible.

Needs gymnasium (``pip install 'aniate[gym]'``).
"""

from __future__ import annotations

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("aniate.gym needs gymnasium:  pip install 'aniate[gym]'") from exc

from aniate.mdp.env import Env, _is_product, _unwrap, default_max_steps

__all__ = ["AniateEnv", "make", "register"]


class AniateEnv(gymnasium.Env):
    """A Gymnasium environment that samples from an aniate model.

    Parameters
    ----------
    model : MDP, or an NMDP / non-Markovian model
    max_steps : int, optional
        Truncate episodes after this many steps.  Defaults to the horizon, or
        to the step at which ``gamma**t`` falls below 1e-6, as in ``simulate``.
    hide_memory : bool
        Non-Markovian models only: observe the world state (default), and
        report the memory state in ``info["memory"]``.
    render_mode : None or "ansi"
        ``"ansi"`` makes ``render()`` return a line describing the last step.
    log : bool
        Write aniate log lines for finished episodes.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, model, *, max_steps=None, hide_memory=True, render_mode=None, log=False):
        m = _unwrap(model)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'ansi'; got {render_mode!r}")
        self.model = m
        self.render_mode = render_mode
        self._env = Env(m, max_steps=default_max_steps(m) if max_steps is None else max_steps,
                        labels=False, hide_memory=hide_memory and _is_product(m), log=log)
        self.observation_space = spaces.Discrete(len(self._env.states))
        self.action_space = spaces.Discrete(m.A)

    def observation_label(self, observation):
        """The state label of an integer observation."""
        return self._env.states[int(observation)]

    def action_label(self, action):
        """The action label of an integer action."""
        return self.model.actions[int(action)]

    def reset(self, *, seed=None, options=None):
        """Start an episode.  ``options={"state": label}`` fixes the start state."""
        super().reset(seed=seed)
        self._env._rng = self.np_random          # sample the model with Gymnasium's seeded generator
        obs, info = self._env.reset(state=(options or {}).get("state"))
        return obs, _gym_info(info)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.model.A}); got {action!r}")
        env = self._env
        if env._s is None:
            raise RuntimeError("call reset() before step()")
        if env._done:
            raise RuntimeError("this episode is over; call reset()")
        # By position, even when the action labels are themselves integers.
        obs, reward, terminated, truncated, info = env._step(int(action))
        return obs, reward, terminated, truncated, _gym_info(info)

    def render(self):
        if self.render_mode == "ansi":
            return self._env.render()
        return None


def _gym_info(info):
    """aniate's ``info`` without the episode counter, so that equal seeds give equal ``info``."""
    info.pop("episode", None)
    return info


def make(model, **kwargs):
    """``AniateEnv(model, **kwargs)``: the model as a Gymnasium environment."""
    return AniateEnv(model, **kwargs)


def _entry_point(model, **kwargs):
    return AniateEnv(model() if callable(model) else model, **kwargs)


def register(id, model, **kwargs):
    """Register a model with Gymnasium, so that ``gymnasium.make(id)`` builds it.

    ``model`` is a model, or a function without arguments that returns one;
    the function is called on every ``gymnasium.make``.  ``kwargs`` are passed
    to :class:`AniateEnv`.
    """
    gymnasium.register(id=id, entry_point=_entry_point, kwargs={"model": model, **kwargs})
