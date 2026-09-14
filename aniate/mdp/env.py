"""Simulation: a Gym-style environment, and fast batched episodes.

    env = model.env(seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step("east")

    runs = aniate.simulate(model, policy, episodes=10_000)

Both sample straight from the model's own transition table, so what they do
*is* the definition -- there is no second copy of the dynamics to drift.
"""

from __future__ import annotations

import math
import time

import numpy as np

from aniate import log as _log

__all__ = ["Env", "simulate", "Episode", "Episodes", "default_max_steps"]


def _is_product(m):
    return hasattr(m, "machine") and hasattr(m, "base")


def _unwrap(model):
    """Accept an NMDP definition wherever a model is expected."""
    if not hasattr(model, "P") and hasattr(model, "product"):
        return model.product
    return model


def default_max_steps(m):
    """Episode cap: the horizon, or where gamma**t has decayed below 1e-6."""
    if m.horizon is not None:
        return m.horizon
    if m.gamma == 0.0:
        return 1
    if m.gamma >= 1.0:
        return 10_000
    return int(min(100_000, math.ceil(math.log(1e-6) / math.log(m.gamma))))


def _milestone(n):
    """Log episodes 1-10, then 20, 30 ... 100, then 200 ... 1000, and so on."""
    return n <= 10 or n % (10 ** int(math.log10(n))) == 0


class Env:
    """Gym-style simulator.

    Parameters
    ----------
    model : MDP, or a non-Markovian model / NMDP
    seed : int, optional
    max_steps : int, optional
        Truncate episodes after this many steps.  Defaults to the horizon.
    labels : bool
        Observations are state labels (default), or integer positions with
        ``labels=False`` -- what a table-based agent indexes arrays with.
    hide_memory : bool
        Non-Markovian models only: observe the *world* state and not the
        memory -- the view an agent actually has.  ``info["memory"]`` still
        reports it.
    log : bool
        Log finished episodes (on a thinning schedule) and, at debug level,
        every step.

    ``info`` carries ``episode``, ``t``, ``state``, ``action``, ``return``
    (discounted, so far) and ``discount`` (gamma**t), plus ``memory`` and
    ``events`` for non-Markovian models.
    """

    def __init__(self, model, *, seed=None, max_steps=None, labels=True, hide_memory=False, log=True):
        self.model = _unwrap(model)
        self.product = _is_product(self.model)
        if hide_memory and not self.product:
            raise ValueError("hide_memory only applies to non-Markovian models")
        self.hide_memory = hide_memory
        self.labels = labels
        self.logging = log
        self.max_steps = self.model.horizon if max_steps is None else int(max_steps)
        self._rng = np.random.default_rng(seed)
        self._s = None
        self._t = 0
        self._done = True
        self._last = None
        self.episode = 0
        self.episode_return = 0.0
        self._discount = 1.0
        self._finished = 0
        self._return_sum = 0.0

    @property
    def states(self):
        """The observation space."""
        return self.model.base.states if self.hide_memory else self.model.states

    @property
    def actions(self):
        return self.model.actions

    @property
    def state(self):
        """Current state label (the full model state, memory included)."""
        return None if self._s is None else self.model.states[self._s]

    def _obs(self, s):
        if self.hide_memory:
            s = s % self.model.nS
        return self.states[s] if self.labels else int(s)

    def _info(self, s, a=None, prev=None):
        info = {"episode": self.episode, "t": self._t, "state": self.model.states[s],
                "return": self.episode_return, "discount": self._discount}
        if a is not None:
            info["action"] = self.model.actions[a]
        if self.product:
            world, memory = self.model.unflat(s)
            info["world_state"], info["memory"] = world, memory
            info["events"] = sorted(self.model.step_events(prev, a, s)) if prev is not None else []
        return info

    def reset(self, *, seed=None, state=None):
        """Start an episode.  Returns ``(observation, info)``.

        ``state`` fixes the start (a world state if memory is hidden).  If the
        start state is terminal the episode is already over.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        m = self.model
        if state is None:
            s = int(min(np.searchsorted(np.cumsum(m.initial), self._rng.random(), side="right"), m.S - 1))
        elif self.hide_memory:
            s = m.flat(state, m.machine.initial)
        else:
            s = m.index(state)
        self._s, self._t, self._last = s, 0, None
        self._done = bool(m.terminal[s])
        self.episode += 1
        self.episode_return, self._discount = 0.0, 1.0
        info = self._info(s)
        info["terminal"] = self._done
        return self._obs(s), info

    def step(self, action):
        """Take an action (label, or position).  Returns the Gym 5-tuple."""
        if self._s is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("this episode is over; call reset()")
        return self._step(self.model.action_index(action))

    def _step(self, a):
        m, s = self.model, self._s
        idx, prob, rew = m._row(s, a)
        if idx.size == 0 or prob.sum() <= 0:
            raise RuntimeError(f"{m.states[s]!r} / {m.actions[a]!r} has no successors")
        c = np.cumsum(prob)
        k = int(min(np.searchsorted(c, self._rng.random() * c[-1], side="right"), idx.size - 1))
        s2, r = int(idx[k]), float(rew[k])
        self._t += 1
        self.episode_return += self._discount * r
        self._discount *= m.gamma
        terminated = bool(m.terminal[s2])
        truncated = (not terminated) and self.max_steps is not None and self._t >= self.max_steps
        self._done = terminated or truncated
        self._s = s2
        self._last = (s, a, s2, r)
        info = self._info(s2, a, s)
        if self.logging:
            self._log_step(info, r, terminated)
        return self._obs(s2), r, terminated, truncated, info

    def _log_step(self, info, r, terminated):
        from aniate.mdp.text import state_str

        if _log.enabled("debug"):
            s, a, s2, _ = self._last
            _log.debug("env", f"episode {self.episode}", f"t={self._t}",
                       f"{state_str(self.model, s)} -> {self.model.actions[a]} -> {state_str(self.model, s2)}",
                       f"reward {r:+.4g}", f"return {self.episode_return:.4g}",
                       f"memory {info['memory']}" if self.product else None)
        if self._done:
            self._finished += 1
            self._return_sum += self.episode_return
            if _milestone(self._finished):
                _log.info("env", f"episode {self._finished:,} done", f"{self._t} steps",
                          f"return {self.episode_return:.4g}", "terminated" if terminated else "truncated",
                          f"mean return {self._return_sum / self._finished:.4g}")

    def render(self):
        """One line describing the last transition."""
        from aniate.mdp.text import state_str

        if self._s is None:
            return "(not started)"
        m = self.model
        if self._last is None:
            return f"t=0  start at {state_str(m, self._s)}"
        s, a, s2, r = self._last
        line = (f"t={self._t}  {state_str(m, s)} --{m.actions[a]}--> {state_str(m, s2)}  "
                f"reward {r:+.4g}  return {self.episode_return:.4g}")
        return line + ("  [done]" if self._done else "")

    def __repr__(self):
        return f"Env({self.model!r}, episode={self.episode}, t={self._t}, state={self.state!r})"


class Episode:
    """One trajectory, read from an :class:`Episodes` batch."""

    def __init__(self, model, states, actions, rewards, terminated):
        self.model = model
        self.state_index = np.asarray(states, dtype=int)
        self.action_index = np.asarray(actions, dtype=int)
        self.rewards = np.asarray(rewards, dtype=float)
        self.terminated = bool(terminated)
        self.truncated = not self.terminated
        self.discounted_return = float(np.sum(model.gamma ** np.arange(self.rewards.size) * self.rewards))

    @property
    def length(self):
        return int(self.rewards.size)

    @property
    def total_reward(self):
        return float(self.rewards.sum())

    @property
    def states(self):
        return [self.model.states[i] for i in self.state_index]

    @property
    def actions(self):
        return [self.model.actions[i] for i in self.action_index]

    @property
    def memory(self):
        """Memory state at each step (non-Markovian models only)."""
        if not _is_product(self.model):
            raise AttributeError("only non-Markovian models have memory")
        return [self.model.unflat(i)[1] for i in self.state_index]

    def table(self, max_rows=30):
        """The trajectory, one step per line."""
        from aniate.mdp.text import state_str

        m = self.model
        lines = [f"{'t':>4}  {'state':<24} {'action':<10} {'reward':>9}"]
        for t in range(min(self.length, max_rows)):
            lines.append(f"{t:>4}  {state_str(m, self.state_index[t]):<24} "
                         f"{str(m.actions[self.action_index[t]]):<10} {self.rewards[t]:>+9.4g}")
        if self.length > max_rows:
            lines.append(f"  ... {self.length - max_rows} more steps")
        end = "terminated" if self.terminated else "truncated"
        lines.append(f"{self.length:>4}  {state_str(m, self.state_index[-1]):<24} ({end})")
        lines.append(f"return {self.discounted_return:.6g} (discounted), {self.total_reward:.6g} (total)")
        return "\n".join(lines)

    def __repr__(self):
        end = "terminated" if self.terminated else "truncated"
        return f"Episode({self.length} steps, return {self.discounted_return:.4g}, {end})"


class Episodes:
    """A batch of simulated episodes, stored as padded arrays.

    ``states[i, t]`` is the state position at step ``t`` of episode ``i``
    (``-1`` after the episode ends); ``actions`` and ``rewards`` likewise,
    with rewards padded by 0.
    """

    def __init__(self, model, states, actions, rewards, lengths, terminated, max_steps, policy=None):
        self.model = model
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.lengths = np.asarray(lengths, dtype=int)
        self.terminated = np.asarray(terminated, dtype=bool)
        self.truncated = ~self.terminated
        self.max_steps = max_steps
        self.policy = policy
        self.returns = rewards @ (model.gamma ** np.arange(rewards.shape[1]))
        self._value = False

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        L = int(self.lengths[i])
        return Episode(self.model, self.states[i, :L + 1], self.actions[i, :L], self.rewards[i, :L],
                       self.terminated[i])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def mean_return(self):
        return float(self.returns.mean())

    @property
    def stderr(self):
        n = len(self.returns)
        return float(self.returns.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")

    @property
    def truncation_bias(self):
        """Upper bound on how much truncated episodes shift the mean return."""
        return truncation_bias(self.model, self.lengths, self.truncated)

    @property
    def model_value(self):
        """Exact expected return of the simulated policy, or None if it cannot be evaluated."""
        if self._value is False:
            from aniate.mdp.solve import evaluate

            try:
                self._value = float(self.model.initial @ evaluate(self.model, self.policy))
            except (TypeError, ValueError):
                self._value = None
        return self._value

    def partial_returns(self):
        """``(N, L + 1)``: discounted return collected before step ``t``, held after the end."""
        disc = self.model.gamma ** np.arange(self.rewards.shape[1])
        out = np.zeros((len(self), self.rewards.shape[1] + 1))
        np.cumsum(self.rewards * disc, axis=1, out=out[:, 1:])
        return out

    def martingale(self):
        """``(N, L + 1)``: ``M_t = G_<t + gamma^t V(s_t)`` under the simulated policy.

        If the model is right about the simulator, every ``M_t`` has the same
        expectation, ``V(s_0)``.  Held constant after an episode ends.
        """
        from aniate.mdp.solve import values_by_time

        L = self.rewards.shape[1]
        try:
            V = values_by_time(self.model, self.policy, L)
        except TypeError as exc:
            raise ValueError("the martingale needs a policy that can be evaluated exactly "
                             "(an array, dict, function, Solution or MemoryPolicy)") from exc
        t = np.arange(L + 1)
        s = np.where(self.states >= 0, self.states, 0)
        M = self.partial_returns() + (self.model.gamma ** t) * V[t[None, :], s]
        last = M[np.arange(len(self)), self.lengths]
        return np.where(t[None, :] > self.lengths[:, None], last[:, None], M)

    def occupancy(self):
        """Fraction of all visited time steps spent in each state, ``(S,)``."""
        visits = np.bincount(self.states[self.states >= 0], minlength=self.model.S)
        return visits / max(visits.sum(), 1)

    def describe(self):
        """The batch as a JSON-serialisable dict."""
        p5, p50, p95 = (float(x) for x in np.percentile(self.returns, [5, 50, 95]))
        return {
            "kind": "Episodes",
            "episodes": len(self),
            "steps": int(self.lengths.sum()),
            "mean_return": self.mean_return,
            "stderr": self.stderr,
            "return_quantiles": {"p5": p5, "p50": p50, "p95": p95},
            "mean_length": float(self.lengths.mean()),
            "terminated": int(self.terminated.sum()),
            "truncated": int(self.truncated.sum()),
            "model_value": self.model_value,
            "truncation_bias": self.truncation_bias,
        }

    def _repr_html_(self):
        from aniate.notebook import to_html

        return to_html(self)

    def __repr__(self):
        return (f"Episodes({len(self):,} episodes: mean return {self.mean_return:.5g} "
                f"+/- {self.stderr:.2g} s.e., mean length {self.lengths.mean():.1f}, "
                f"{int(self.truncated.sum()):,} truncated)")


def truncation_bias(m, lengths, truncated):
    """Mean over episodes of the most reward a truncated tail could have held."""
    lengths, truncated = np.asarray(lengths), np.asarray(truncated, dtype=bool)
    if not truncated.any():
        return 0.0
    rmax = max((float(np.abs(r).max()) for r in m.r_next if r.size), default=0.0)
    tails = np.zeros(lengths.size)
    for i in np.flatnonzero(truncated):
        L = int(lengths[i])
        if m.horizon is not None:
            k = max(m.horizon - L, 0)
            tails[i] = rmax * (k if m.gamma == 1.0 else m.gamma**L * (1 - m.gamma**k) / (1 - m.gamma))
        elif m.gamma < 1.0:
            tails[i] = rmax * m.gamma**L / (1.0 - m.gamma)
        else:
            return float("inf")
    return float(tails.mean())


class _Buffer:
    """(N, capacity) arrays that double in width as episodes get longer."""

    def __init__(self, n, fill, dtype):
        self.a = np.full((n, 64), fill, dtype=dtype)
        self.fill = fill

    def put(self, rows, t, values):
        if t >= self.a.shape[1]:
            wider = np.full((self.a.shape[0], self.a.shape[1] * 2), self.fill, dtype=self.a.dtype)
            wider[:, :self.a.shape[1]] = self.a
            self.a = wider
        self.a[rows, t] = values


def _run_batched(m, pol, N, T, rng, start):
    """All episodes advance together: one vectorised draw per action per step."""
    keys, dead = [], []
    for a in range(m.A):
        P = m.P[a]
        counts = np.diff(P.indptr).astype(np.intp)
        totals = np.asarray(P.sum(axis=1)).ravel()
        cum = np.cumsum(P.data)
        row_start = np.concatenate([[0.0], cum])[P.indptr[:-1]]
        within = cum - np.repeat(row_start, counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.minimum(within / np.repeat(totals, counts), 1.0)
        # row + cumulative fraction is sorted, so one searchsorted samples every row at once
        keys.append(np.repeat(np.arange(m.S), counts) + frac)
        dead.append(totals <= 0)

    if start is None:
        s = np.minimum(np.searchsorted(np.cumsum(m.initial), rng.random(N), side="right"), m.S - 1)
    else:
        s = np.full(N, m.index(start))
    states, actions, rewards = _Buffer(N, -1, np.int32), _Buffer(N, -1, np.int32), _Buffer(N, 0.0, float)
    states.put(slice(None), 0, s)
    terminated = m.terminal[s].copy()
    alive = ~terminated
    lengths = np.zeros(N, dtype=int)
    marks = [0.25, 0.5, 0.75] if N >= 1000 else []

    for t in range(T):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        st = s[idx]
        act = np.asarray(pol.batch(st, t), dtype=np.int64)
        u = rng.random(idx.size)
        nxt = np.empty(idx.size, dtype=np.int64)
        rew = np.empty(idx.size)
        for a in np.unique(act):
            sel = act == a
            ss = st[sel]
            if dead[a][ss].any():
                bad = ss[dead[a][ss]][0]
                raise RuntimeError(f"{m.states[bad]!r} / {m.actions[a]!r} has no successors")
            k = np.minimum(np.searchsorted(keys[a], ss + u[sel], side="right"), m.P[a].indptr[ss + 1] - 1)
            nxt[sel] = m.P[a].indices[k]
            rew[sel] = m.r_next[a][k]
        actions.put(idx, t, act)
        rewards.put(idx, t, rew)
        states.put(idx, t + 1, nxt)
        s[idx] = nxt
        lengths[idx] += 1
        ended = m.terminal[nxt]
        terminated[idx] = ended
        alive[idx] = ~ended
        while marks and 1.0 - alive.mean() >= marks[0]:
            _log.info("simulate", f"{int(marks.pop(0) * 100)}% of {N:,} episodes finished by step {t + 1}")

    L = int(lengths.max())
    return states.a[:, :L + 1], actions.a[:, :L], rewards.a[:, :L], lengths, terminated


def _run_sequential(m, pol, N, T, seed, start):
    """For policy objects that keep their own state: one episode at a time."""
    env = Env(m, seed=seed, max_steps=T, labels=False, log=False)
    runs = []
    for _ in range(N):
        s, info = env.reset(state=start)
        pol.reset()
        S_, A_, R_ = [s], [], []
        while not env._done:
            a = pol.step(s)
            s, r, _, _, _ = env._step(a)
            S_.append(s)
            A_.append(a)
            R_.append(r)
        runs.append((S_, A_, R_, bool(m.terminal[s])))
    L = max(len(r[1]) for r in runs)
    states = np.full((N, L + 1), -1, dtype=np.int32)
    actions = np.full((N, L), -1, dtype=np.int32)
    rewards = np.zeros((N, L))
    for i, (S_, A_, R_, _) in enumerate(runs):
        states[i, :len(S_)] = S_
        actions[i, :len(A_)] = A_
        rewards[i, :len(R_)] = R_
    return (states, actions, rewards, np.array([len(r[1]) for r in runs]),
            np.array([r[3] for r in runs]))


def simulate(model, policy=None, episodes=100, *, max_steps=None, seed=0, start=None):
    """Run ``episodes`` episodes of ``policy`` (uniformly random if omitted).

    Returns :class:`Episodes`.  Accepts every policy form in
    :mod:`aniate.mdp.policies`, including time-indexed ones.  Tables and
    stochastic matrices run all episodes in parallel; policy objects with
    their own ``step()`` run one episode at a time.
    """
    from aniate.mdp.policies import as_policy

    m = _unwrap(model)
    N = int(episodes)
    if N < 1:
        raise ValueError("episodes must be at least 1")
    T = default_max_steps(m) if max_steps is None else int(max_steps)
    env_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    pol = as_policy(m, policy, np.random.default_rng(policy_seed))
    t0 = time.perf_counter()
    if hasattr(pol, "batch"):
        arrays = _run_batched(m, pol, N, T, np.random.default_rng(env_seed), start)
    else:
        arrays = _run_sequential(m, pol, N, T, env_seed, start)
    runs = Episodes(m, *arrays, max_steps=T, policy=policy)
    dt = time.perf_counter() - t0
    _log.info("simulate", f"{N:,} episodes", f"{int(runs.lengths.sum()):,} steps",
              f"{dt * 1e3:.0f} ms" if dt < 1 else f"{dt:.2f} s",
              f"mean return {runs.mean_return:.6g} +/- {runs.stderr:.2g}",
              f"{int(runs.terminated.sum()):,} terminated", f"{int(runs.truncated.sum()):,} truncated")
    return runs
