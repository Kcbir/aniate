"""Every way a user might hand over a policy, reduced to two internal forms.

Accepted anywhere a policy is expected:

* ``None``                -- uniformly random actions
* a :class:`Solution`     -- its (possibly time-indexed) policy
* ``(S,)`` ints           -- a deterministic stationary policy
* ``(T, S)`` ints         -- a deterministic time-indexed policy
* ``(S, A)`` floats       -- a stochastic policy, rows summing to 1
* ``(T, S, A)`` floats    -- a stochastic time-indexed policy
* ``{state: action}``     -- by label
* ``f(state) -> action``  -- by label
* a ``MemoryPolicy``      -- from ``nmdp``
* any object with ``step(state_index) -> action_index`` (simulation only)

On a non-Markovian (product) model, arrays, dicts and functions over the
*world* states are memoryless policies and are lifted to every memory state.

``policy_matrix`` turns the first group into exact probabilities, which is
what evaluation needs; ``as_policy`` turns anything into an object with
``reset()`` and ``step(s)``, which is what simulation needs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["policy_matrix", "as_policy", "one_hot"]


def one_hot(table, A):
    table = np.asarray(table, dtype=int)
    if table.size and (table.min() < 0 or table.max() >= A):
        raise ValueError(f"policy contains action indices outside [0, {A})")
    M = np.zeros(table.shape + (A,), dtype=float)
    np.put_along_axis(M, table[..., None], 1.0, axis=-1)
    return M


def _is_solution(p):
    return hasattr(p, "policy") and hasattr(p, "value") and hasattr(p, "solver")


def _world(m):
    return m.base if hasattr(m, "machine") and hasattr(m, "base") else m


def _table_from_mapping(m, mapping):
    world = _world(m)
    table = np.empty(world.S, dtype=int)
    for i, label in enumerate(world.states):
        if isinstance(mapping, dict):
            try:
                act = mapping[label]
            except KeyError:
                if world.terminal[i]:
                    table[i] = 0
                    continue
                raise KeyError(f"policy dict has no action for state {label!r}") from None
        else:
            act = mapping(label)
        table[i] = m.action_index(act)
    return table


def _lift(m, M):
    """Tile a world-sized policy across memory states of a product model."""
    world = _world(m)
    state_axis = M.ndim - 2
    n = M.shape[state_axis]
    if n == m.S:
        return M
    if world is not m and n == world.S:
        reps = [1] * M.ndim
        reps[state_axis] = m.S // world.S
        return np.tile(M, reps)
    raise ValueError(f"policy covers {n} states but the model has {m.S}")


def policy_matrix(m, policy):
    """Exact action probabilities: ``(S, A)``, or ``(T, S, A)`` if time-indexed."""
    if policy is None:
        return np.full((m.S, m.A), 1.0 / m.A)
    if hasattr(policy, "product_table"):
        arr = policy.product_table(m)
    elif _is_solution(policy):
        arr = policy.policy
    elif isinstance(policy, dict) or (callable(policy) and not hasattr(policy, "step")):
        arr = _table_from_mapping(m, policy)
    elif hasattr(policy, "step"):
        raise TypeError(
            "a policy object with step() can be simulated but not evaluated exactly; "
            "pass an array, a dict, a function, or a Solution instead"
        )
    else:
        arr = np.asarray(policy)

    if arr.dtype.kind in "iub" or (arr.ndim == 1 and arr.dtype.kind == "f"):
        if arr.dtype.kind == "f":
            if not np.all(arr == np.round(arr)):
                raise ValueError("a 1-D policy must hold integer action indices")
            arr = arr.astype(int)
        if arr.ndim not in (1, 2):
            raise ValueError(f"an integer policy must be (S,) or (T, S); got shape {arr.shape}")
        M = one_hot(arr, m.A)
    elif arr.dtype.kind == "f":
        if arr.ndim not in (2, 3) or arr.shape[-1] != m.A:
            raise ValueError(
                f"a stochastic policy must be (S, A) or (T, S, A) with A={m.A}; got shape {arr.shape}"
            )
        if (arr < 0).any() or not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-9):
            raise ValueError("stochastic policy rows must be non-negative and sum to 1")
        M = arr.astype(float)
    else:
        raise TypeError(f"cannot interpret {type(policy).__name__} as a policy")
    return _lift(m, M)


class TablePolicy:
    """Deterministic, stationary ``(S,)`` or time-indexed ``(T, S)``."""

    def __init__(self, table):
        self.table = np.asarray(table, dtype=int)
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, s):
        if self.table.ndim == 1:
            a = self.table[s]
        else:
            if self.t >= self.table.shape[0]:
                raise RuntimeError(f"time-indexed policy only covers {self.table.shape[0]} steps")
            a = self.table[self.t, s]
        self.t += 1
        return int(a)

    def batch(self, s, t):
        if self.table.ndim == 1:
            return self.table[s]
        if t >= self.table.shape[0]:
            raise RuntimeError(f"time-indexed policy only covers {self.table.shape[0]} steps")
        return self.table[t, s]


class StochasticPolicy:
    def __init__(self, M, rng):
        self.cum = np.cumsum(M, axis=-1)
        self.rng = rng
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, s):
        c = self.cum[s] if self.cum.ndim == 2 else self.cum[self.t, s]
        self.t += 1
        return int(min(np.searchsorted(c, self.rng.random() * c[-1], side="right"), c.size - 1))

    def batch(self, s, t):
        c = self.cum[s] if self.cum.ndim == 2 else self.cum[t, s]
        u = self.rng.random(len(s)) * c[:, -1]
        return np.minimum((c <= u[:, None]).sum(axis=1), c.shape[1] - 1)


class RandomPolicy:
    def __init__(self, A, rng):
        self.A, self.rng = A, rng

    def reset(self):
        pass

    def step(self, s):
        return int(self.rng.integers(self.A))

    def batch(self, s, t):
        return self.rng.integers(self.A, size=len(s))


class _Custom:
    def __init__(self, obj):
        self.obj = obj

    def reset(self):
        if hasattr(self.obj, "reset"):
            self.obj.reset()

    def step(self, s):
        return int(self.obj.step(s))


def as_policy(m, policy, rng):
    """An object with ``reset()`` and ``step(state_index) -> action_index``."""
    if policy is None:
        return RandomPolicy(m.A, rng)
    if hasattr(policy, "step") and not hasattr(policy, "product_table") and not _is_solution(policy):
        return _Custom(policy)
    M = policy_matrix(m, policy)
    if np.all((M == 0.0) | (M == 1.0)):
        return TablePolicy(M.argmax(axis=-1))
    return StochasticPolicy(M, rng)
