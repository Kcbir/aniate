"""Two ways to write a model without touching a transition matrix.

``Builder`` -- declare transitions one at a time:

    b = Builder(gamma=0.95)
    b.transition("has_stock", "sell", "low_stock", prob=0.7, reward=10)
    b.transition("has_stock", "sell", "has_stock", prob=0.3, reward=10)
    b.transition("low_stock", "sell", "low_stock", prob=1.0, reward=2)
    m = b.build()

``from_functions`` -- describe the dynamics as code:

    m = from_functions(states, actions,
                       transition=lambda s, a: {next_state: prob, ...},
                       reward=lambda s, a, s_next: ...,
                       gamma=0.95)

Labels can be any hashable: strings, ints, tuples like ``(row, col)``.
Rewards are per transition, so "reward 1 on entering the goal" means exactly
that, in the simulator as well as in the solver.
"""

from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp

from aniate.mdp.core import MDP

__all__ = ["Builder", "from_functions"]


class Builder:
    """Incremental model construction.  States and actions register on first use."""

    def __init__(self, gamma=0.95, horizon=None):
        self.gamma = gamma
        self.horizon = horizon
        self._states = {}       # label -> position, first-seen order
        self._actions = {}
        self._entries = {}      # (s, a, s_next) -> [prob, reward]
        self._terminal = set()
        self._initial = {}

    def state(self, label):
        """Register a state (if new) and return its position."""
        if label not in self._states:
            self._states[label] = len(self._states)
        return self._states[label]

    def action(self, label):
        """Register an action (if new) and return its position."""
        if label not in self._actions:
            self._actions[label] = len(self._actions)
        return self._actions[label]

    def transition(self, state, action, next_state, prob=1.0, reward=0.0):
        """Declare ``Pr(next_state | state, action) = prob`` with reward ``reward``.

        Declaring the same triple twice adds the probabilities and keeps the
        probability-weighted reward.
        """
        prob, reward = float(prob), float(reward)
        if not np.isfinite(prob) or prob < 0:
            raise ValueError(f"prob must be a non-negative number; got {prob} for "
                             f"{(state, action, next_state)}")
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite; got {reward} for {(state, action, next_state)}")
        key = (self.state(state), self.action(action), self.state(next_state))
        if key in self._entries:
            p0, r0 = self._entries[key]
            p1 = p0 + prob
            self._entries[key] = [p1, (p0 * r0 + prob * reward) / p1 if p1 else 0.0]
        else:
            self._entries[key] = [prob, reward]
        return self

    def terminal(self, *states):
        """Mark states as ending the episode.  They absorb with zero reward."""
        for s in states:
            self._terminal.add(self.state(s))
        return self

    def initial(self, dist):
        """Start distribution: ``{state: weight}``, or a single state label."""
        if not isinstance(dist, dict):
            dist = {dist: 1.0}
        self._initial = {self.state(k): float(v) for k, v in dist.items()}
        return self

    def build(self, missing="error", validate=True):
        """Assemble the MDP.

        ``missing`` decides what happens to a state-action pair that was never
        declared: ``"error"`` (default) names them; ``"stay"`` makes the
        action leave the state unchanged with zero reward.
        """
        if not self._entries:
            raise ValueError("nothing declared: call transition() at least once")
        if missing not in ("error", "stay"):
            raise ValueError(f"missing must be 'error' or 'stay'; got {missing!r}")
        S, A = len(self._states), len(self._actions)
        states, actions = list(self._states), list(self._actions)
        entries = dict(self._entries)

        declared = {(s, a) for s, a, _ in entries}
        undeclared = [(s, a) for s in range(S) for a in range(A)
                      if (s, a) not in declared and s not in self._terminal]
        if undeclared and missing == "error":
            shown = ", ".join(f"{states[s]!r}/{actions[a]!r}" for s, a in undeclared[:6])
            more = f" and {len(undeclared) - 6} more" if len(undeclared) > 6 else ""
            raise ValueError(
                f"{len(undeclared)} state-action pairs were never declared: {shown}{more}. "
                "Every action must be defined in every state; declare them, or "
                "build(missing='stay') to make undefined actions do nothing."
            )
        for s, a in undeclared:
            entries[(s, a, s)] = [1.0, 0.0]
        for t in self._terminal:
            for a in range(A):
                if (t, a) not in declared:
                    entries[(t, a, t)] = [1.0, 0.0]

        rows = [[] for _ in range(A)]
        cols = [[] for _ in range(A)]
        probs = [[] for _ in range(A)]
        rews = [[] for _ in range(A)]
        for (s, a, s2), (p, r) in entries.items():
            rows[a].append(s)
            cols[a].append(s2)
            probs[a].append(p)
            rews[a].append(r)

        def mat(a, vals):
            return sp.csr_array((np.array(vals, dtype=float),
                                 (np.array(rows[a], dtype=int), np.array(cols[a], dtype=int))),
                                shape=(S, S))

        P = [mat(a, probs[a]) for a in range(A)]
        R = [mat(a, rews[a]) for a in range(A)]

        initial = None
        if self._initial:
            initial = np.zeros(S)
            for s, w in self._initial.items():
                initial[s] = w
        terminal = sorted(self._terminal)
        return MDP(P, R, self.gamma, states=states, actions=actions, initial=initial,
                   terminal=[states[t] for t in terminal] if terminal else None,
                   horizon=self.horizon, validate=validate)

    def __repr__(self):
        return (f"Builder(gamma={self.gamma}, {len(self._states)} states, "
                f"{len(self._actions)} actions, {len(self._entries)} transitions)")


def _arity(fn):
    params = inspect.signature(fn).parameters.values()
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return 3
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
               for p in params)


def from_functions(states, actions, transition, reward=None, gamma=0.95, *,
                   initial=None, terminal=None, horizon=None, validate=True):
    """Build an MDP from code.

    Parameters
    ----------
    states, actions : iterables of hashable labels
    transition : ``f(state, action) -> {next_state: probability}``
    reward : ``f(state, action, next_state) -> float`` or ``f(state, action) -> float``
    terminal : iterable of states, or predicate ``f(state) -> bool``
        Terminal states are not queried; they absorb with zero reward.
    initial : ``{state: weight}`` or a single state label
    """
    states, actions = list(states), list(actions)
    known = set(states)
    if reward is None:
        n_args = 0
    else:
        n_args = _arity(reward)
        if n_args not in (2, 3):
            raise TypeError("reward must take (state, action) or (state, action, next_state)")
    if callable(terminal):
        term = {s for s in states if terminal(s)}
    else:
        term = set(terminal or ())

    b = Builder(gamma=gamma, horizon=horizon)
    for s in states:
        b.state(s)
    for a in actions:
        b.action(a)
    for s in states:
        if s in term:
            continue
        for a in actions:
            dist = transition(s, a)
            if not isinstance(dist, dict):
                raise TypeError(f"transition({s!r}, {a!r}) must return a dict "
                                f"{{next_state: probability}}; got {type(dist).__name__}")
            for s2, p in dist.items():
                if s2 not in known:
                    raise ValueError(f"transition({s!r}, {a!r}) returned {s2!r}, "
                                     "which is not in states")
                r = 0.0 if n_args == 0 else reward(s, a, s2) if n_args == 3 else reward(s, a)
                b.transition(s, a, s2, prob=p, reward=r)
    if term:
        b.terminal(*term)
    if initial is not None:
        b.initial(initial)
    return b.build(validate=validate)
