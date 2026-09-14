"""The MDP itself.

SPARSE ORIENTATION -- decided once, never flipped:

    P[a][s, s']  ==  Pr(next state = s' | current state = s, action = a)

Row ``s`` of ``P[a]`` is a distribution over successors, and the Bellman
backup for action ``a`` is the single matvec ``P[a] @ v``.

REWARDS are stored per transition, ``r(s, a, s')``, aligned entry-for-entry
with ``P[a].data``.  The expected reward ``R[s, a] = sum_s' P r`` is derived
from it.  Solvers only ever need ``R``; a simulator needs ``r(s, a, s')`` --
"+1 on entering the goal" is a statement about the transition, and a
simulator that emits its expectation instead is not simulating the model.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from aniate import log as _log
from aniate.mdp.space import Space

__all__ = ["MDP"]


def _as_csr_list(P):
    """Normalise ``P`` into A canonical CSR arrays, each S x S."""
    if isinstance(P, np.ndarray):
        if P.ndim != 3:
            raise ValueError(f"dense P must have shape (A, S, S); got shape {P.shape}")
        blocks = [P[a] for a in range(P.shape[0])]
    elif isinstance(P, (list, tuple)):
        blocks = list(P)
    else:
        raise TypeError(
            "P must be a dense (A, S, S) array or a list of S x S matrices, one "
            f"per action; got {type(P).__name__}"
        )
    if not blocks:
        raise ValueError("P is empty: an MDP needs at least one action")

    mats = []
    for a, block in enumerate(blocks):
        if not sp.issparse(block):
            block = np.asarray(block, dtype=float)
            if block.ndim != 2:
                raise ValueError(f"P[{a}] must be 2-D; got shape {block.shape}")
        m = sp.csr_array(block, dtype=float)
        m.sum_duplicates()          # also sorts indices: entries are canonical
        mats.append(m)

    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"every P[a] must have the same shape; got {sorted(shapes)}")
    ((n, k),) = shapes
    if n != k:
        raise ValueError(f"each P[a] must be square (S x S); got {n} x {k}")
    return mats


def _entries(m):
    """(rows, cols) of every stored entry of a canonical CSR array, in data order."""
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    return rows, m.indices


def _gather(mat, rows, cols):
    """``mat[rows[i], cols[i]]`` for every i, zero where nothing is stored."""
    mat = sp.csr_array(mat, dtype=float)
    mat.sum_duplicates()
    S = mat.shape[1]
    mr, mc = _entries(mat)
    keys = mr.astype(np.int64) * S + mc
    want = rows.astype(np.int64) * S + cols
    pos = np.searchsorted(keys, want)
    pos_clipped = np.minimum(pos, max(keys.size - 1, 0))
    hit = (pos < keys.size) & (keys[pos_clipped] == want) if keys.size else np.zeros(want.size, bool)
    out = np.zeros(want.size, dtype=float)
    out[hit] = mat.data[pos_clipped[hit]]
    stray = int(np.count_nonzero(mat.data)) - int(np.count_nonzero(out))
    return out, stray


def _rewards(R, P, S, A):
    """Return ``(R_expected (S, A), r_next per action, stray)``.

    Accepted forms, and only these -- no silent transposes:

    * ``(S, A)`` array: the reward depends on the state and action only.
    * ``(S, A, S)`` array: ``R[s, a, s']`` per transition.
    * a list of A sparse or dense ``S x S`` matrices: ``R[a][s, s']``, laid out
      exactly like ``P``.
    """
    if isinstance(R, (list, tuple)) and len(R) and (
        sp.issparse(R[0]) or np.ndim(R[0]) == 2
    ):
        if len(R) != A:
            raise ValueError(f"R has {len(R)} matrices but P has {A} actions")
        per_action = []
        for a, block in enumerate(R):
            shape = block.shape if sp.issparse(block) else np.shape(block)
            if tuple(shape) != (S, S):
                raise ValueError(f"R[{a}] must have shape ({S}, {S}) like P[{a}]; got {shape}")
            per_action.append(block)
        form = "list"
    elif callable(R):
        raise TypeError(
            "R must be an array, not a function.  To define a model with "
            "functions use MDP.from_functions(states, actions, transition, reward, gamma)."
        )
    else:
        R = np.asarray(R, dtype=float)
        if R.shape == (S, A):
            form = "sa"
        elif R.shape == (S, A, S):
            form = "sas"
        else:
            hint = ""
            if R.shape == (A, S):
                hint = "  It looks transposed: pass R.T."
            elif R.shape == (A, S, S):
                hint = "  It looks like (A, S, S): pass R.transpose(1, 0, 2)."
            raise ValueError(
                f"R must have shape (S, A) = ({S}, {A}) or (S, A, S) = ({S}, {A}, {S}); "
                f"got {R.shape}.{hint}"
            )

    r_next, stray = [], 0
    expected = np.zeros((S, A), dtype=float)
    for a in range(A):
        rows, cols = _entries(P[a])
        if form == "sa":
            r = R[rows, a].astype(float)
            n_stray = 0
        elif form == "sas":
            r = R[rows, a, cols].astype(float)
            full = int(np.count_nonzero(R[:, a, :]))
            n_stray = full - int(np.count_nonzero(r))
        else:
            r, n_stray = _gather(per_action[a], rows, cols)
        stray += max(n_stray, 0)
        r_next.append(np.ascontiguousarray(r))
        expected[:, a] = np.bincount(rows, weights=P[a].data * r, minlength=S)
    if form == "sa":
        expected = np.array(R, dtype=float)   # exact even where rows are malformed
    return expected, r_next, stray


class MDP:
    """A finite Markov decision process.

    Parameters
    ----------
    P : list of A (S x S) matrices, or a dense (A, S, S) array
        ``P[a][s, s']`` is the probability of moving to ``s'`` from ``s``
        under action ``a``.  Stored sparse.
    R : (S, A) array, (S, A, S) array, or list of A (S x S) matrices
        Reward.  See :func:`_rewards` for exactly what each form means.
    gamma : float in [0, 1]
        Discount factor.
    states, actions : sequences of hashable labels, optional
        Defaults are ``s0, s1, ...`` and ``a0, a1, ...``.
    initial : (S,) array or dict ``{state: weight}``, optional
        Start distribution; uniform by default.
    terminal : sequence of states, or (S,) bool mask, optional
        States that end an episode.  They must be absorbing with zero reward,
        so that "the episode stopped" and "the value is zero from here" are
        the same statement.  Inferred from zero-reward self-loops if omitted.
    horizon : int, optional
        Finite horizon.  ``None`` means infinite.
    validate : True, "warn", or False
        ``True`` (default) raises on a malformed model; ``"warn"`` builds it
        anyway and logs the problems (see ``m.report``); ``False`` skips the checks.
    """

    layout = None   # optional geometry, e.g. set by problems.gridworld for vis.grid

    def __init__(self, P, R, gamma, *, states=None, actions=None, initial=None,
                 terminal=None, horizon=None, validate=True):
        self.P = _as_csr_list(P)
        self.A = len(self.P)
        self.S = self.P[0].shape[0]
        self.states = (
            Space.default(self.S, "s", "state") if states is None
            else states if isinstance(states, Space) else Space(states, kind="state")
        )
        self.actions = (
            Space.default(self.A, "a", "action") if actions is None
            else actions if isinstance(actions, Space) else Space(actions, kind="action")
        )
        if len(self.states) != self.S:
            raise ValueError(f"P has {self.S} states but {len(self.states)} state labels were given")
        if len(self.actions) != self.A:
            raise ValueError(f"P has {self.A} actions but {len(self.actions)} action labels were given")

        self.R, self.r_next, self._stray_rewards = _rewards(R, self.P, self.S, self.A)

        gamma = float(gamma)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1]; got {gamma}")
        self.gamma = gamma

        if horizon is not None:
            if int(horizon) != horizon or horizon <= 0:
                raise ValueError(f"horizon must be a positive integer; got {horizon}")
            horizon = int(horizon)
        self.horizon = horizon

        self.initial = self._make_initial(initial)
        self._terminal_given = terminal is not None
        self.terminal = self._infer_terminal() if terminal is None else self._make_mask(terminal)

        self.report = None
        if validate:
            from aniate.mdp.validate import validate as _validate

            self.report = _validate(self, raise_on_error=(validate is True))
            if not getattr(self, "_quiet_build", False):
                nnz = sum(int(np.count_nonzero(p.data)) for p in self.P)
                _log.info("model", f"{self.S:,} states", f"{self.A} actions", f"gamma {self.gamma:g}",
                          "horizon inf" if self.horizon is None else f"horizon {self.horizon}",
                          f"{nnz:,} transitions",
                          "valid" if self.report.ok else f"{len(self.report.errors)} errors")

    @classmethod
    def from_functions(cls, states, actions, transition, reward=None, gamma=0.95, **kwargs):
        """Build from ``transition(s, a) -> {s': p}`` and ``reward(s, a[, s'])``.

        See :func:`aniate.mdp.builder.from_functions`.
        """
        from aniate.mdp.builder import from_functions

        return from_functions(states, actions, transition, reward, gamma, **kwargs)

    # -- construction helpers ------------------------------------------------

    def _make_initial(self, initial):
        if initial is None:
            return np.full(self.S, 1.0 / self.S)
        if isinstance(initial, dict):
            d = np.zeros(self.S)
            for k, w in initial.items():
                d[self.index(k)] += float(w)
        else:
            d = np.asarray(initial, dtype=float).ravel()
            if d.shape != (self.S,):
                raise ValueError(f"initial must have shape ({self.S},); got {d.shape}")
        if not np.all(np.isfinite(d)) or (d < 0).any():
            raise ValueError("initial distribution must be finite and non-negative")
        if d.sum() <= 0:
            raise ValueError("initial distribution has no mass")
        return d / d.sum()

    def _make_mask(self, terminal):
        # A bool mask only if it is one; never np.asarray a list of labels, which
        # breaks (or silently reshapes) when labels are tuples of uneven length.
        if isinstance(terminal, np.ndarray) and terminal.dtype == bool:
            if terminal.shape != (self.S,):
                raise ValueError(f"terminal mask must have shape ({self.S},); got {terminal.shape}")
            return terminal.copy()
        if isinstance(terminal, list) and len(terminal) == self.S and terminal and \
                all(isinstance(t, (bool, np.bool_)) for t in terminal):
            return np.array(terminal, dtype=bool)
        mask = np.zeros(self.S, dtype=bool)
        if isinstance(terminal, tuple) and terminal in self.states:
            terminal = [terminal]                 # a single tuple label, not a list of labels
        items = terminal if isinstance(terminal, (list, tuple, set, frozenset)) else [terminal]
        for t in items:
            mask[self.index(t)] = True
        return mask

    def _infer_terminal(self):
        """Terminal = every action self-loops with probability 1 and pays nothing."""
        mask = np.ones(self.S, dtype=bool)
        for a in range(self.A):
            mask &= np.isclose(self.P[a].diagonal(), 1.0, atol=1e-12)
            mask &= self.R[:, a] == 0.0
        return mask

    # -- lookup --------------------------------------------------------------

    def index(self, state):
        """Position of a state label."""
        return self.states.index(state)

    def action_index(self, action):
        """Position of an action label."""
        return self.actions.index(action)

    def _row(self, s, a):
        """Successor positions, probabilities and rewards for ``(s, a)``, by position."""
        m = self.P[a]
        lo, hi = m.indptr[s], m.indptr[s + 1]
        return m.indices[lo:hi], m.data[lo:hi], self.r_next[a][lo:hi]

    def transitions(self, state, action):
        """``{next_state: probability}`` for one state and action."""
        idx, prob, _ = self._row(self.index(state), self.action_index(action))
        return {self.states[int(j)]: float(p) for j, p in zip(idx, prob) if p != 0.0}

    def reward(self, state, action, next_state=None):
        """``r(s, a, s')``, or the expected ``R(s, a)`` when ``next_state`` is omitted."""
        s, a = self.index(state), self.action_index(action)
        if next_state is None:
            return float(self.R[s, a])
        j = self.index(next_state)
        idx, prob, rew = self._row(s, a)
        hit = np.flatnonzero((idx == j) & (prob != 0.0))
        if hit.size == 0:
            raise ValueError(
                f"{next_state!r} cannot follow {state!r} under {action!r}: "
                "that transition has probability 0"
            )
        return float(rew[hit[0]])

    # -- Bellman machinery ---------------------------------------------------

    def q_from_v(self, v):
        """The (S, A) array ``R + gamma * P @ v``."""
        v = np.asarray(v, dtype=float).ravel()
        q = np.empty((self.S, self.A), dtype=float)
        for a in range(self.A):
            q[:, a] = self.R[:, a] + self.gamma * (self.P[a] @ v)
        return q

    def greedy(self, v):
        """Greedy deterministic policy with respect to ``v``."""
        return self.q_from_v(v).argmax(axis=1)

    # -- the things people do with a model -----------------------------------

    def solve(self, method="auto", **kwargs):
        """Optimal policy and value.  See :mod:`aniate.mdp.solve`."""
        from aniate.mdp.solve import solve

        return solve(self, method=method, **kwargs)

    def evaluate(self, policy):
        """Exact value of any policy.  See :func:`aniate.mdp.solve.evaluate`."""
        from aniate.mdp.solve import evaluate

        return evaluate(self, policy)

    def env(self, seed=None, max_steps=None, labels=True):
        """A Gym-style simulator of this model.  See :class:`aniate.mdp.env.Env`."""
        from aniate.mdp.env import Env

        return Env(self, seed=seed, max_steps=max_steps, labels=labels)

    # -- reporting -----------------------------------------------------------

    def summary(self, max_states=12):
        """The whole definition as readable text: spaces, dynamics, rewards."""
        from aniate.mdp.text import mdp_summary

        return mdp_summary(self, max_states=max_states)

    def describe(self):
        """The model as a JSON-serialisable dict."""
        from aniate.mdp.text import state_str

        nnz = sum(int(np.count_nonzero(p.data)) for p in self.P)
        rewards = np.concatenate([r for r in self.r_next if r.size] or [np.zeros(1)])
        return {
            "kind": type(self).__name__,
            "states": self.S,
            "actions": [str(a) for a in self.actions],
            "gamma": self.gamma,
            "horizon": self.horizon,
            "transitions": nnz,
            "reward_range": [float(rewards.min()), float(rewards.max())],
            "initial": {state_str(self, i): float(self.initial[i]) for i in np.flatnonzero(self.initial)[:20]},
            "terminal": [state_str(self, i) for i in np.flatnonzero(self.terminal)[:20]],
            "valid": None if self.report is None else self.report.ok,
            "issues": [] if self.report is None else [i.describe() for i in self.report.issues],
        }

    def __repr__(self):
        h = "inf" if self.horizon is None else self.horizon
        return f"{type(self).__name__}(S={self.S}, A={self.A}, gamma={self.gamma}, horizon={h})"
