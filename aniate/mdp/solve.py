"""Exact policy evaluation and three solvers.

``evaluate``            value of any policy (deterministic, stochastic, time-indexed)
``policy_iteration``    default for discounted problems; exact
``value_iteration``     for gamma = 1 with terminal states
``backward_induction``  finite horizon; exact, time-indexed policy

Value iteration always finishes by evaluating the policy it returns.  The
usual span stopping rule certifies the greedy *policy*, not the value
vector: on a long thin reward gradient (``problems.chain``) it stops with
values tens of units wrong while every number looks healthy.  Here
``Solution.value`` is that policy's exact value, and ``bound`` is about it.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from aniate import log as _log
from aniate.mdp.policies import one_hot, policy_matrix
from aniate.mdp.solution import Solution

__all__ = ["evaluate", "values_by_time", "policy_iteration", "value_iteration",
           "backward_induction", "solve"]


def _chain(m, M):
    """Markov chain ``(P_pi, r_pi)`` induced by an ``(S, A)`` probability matrix."""
    P = None
    for a in range(m.A):
        w = M[:, a]
        if not w.any():
            continue
        term = sp.diags_array(w) @ m.P[a]
        P = term if P is None else P + term
    return sp.csr_array(P), (M * m.R).sum(axis=1)


def _solve_linear(m, P_pi, r_pi):
    """Solve ``v = r_pi + gamma P_pi v`` with ``v = 0`` on terminal states."""
    v = np.zeros(m.S)
    live = np.flatnonzero(~m.terminal)
    if live.size == 0:
        return v
    sub = P_pi[live][:, live]
    lhs = (sp.eye_array(live.size, format="csr") - m.gamma * sub).tocsc()
    rhs = r_pi[live]
    x = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            x = np.atleast_1d(np.asarray(spsolve(lhs, rhs), dtype=float)).ravel()
        except Exception:
            x = None
    if x is None or not np.all(np.isfinite(x)) or \
            np.abs(lhs @ x - rhs).max() > 1e-6 * (1.0 + np.abs(rhs).max()):
        raise ValueError(
            "this policy's value is undefined: with gamma=1 some states never reach a "
            "terminal state, so returns do not converge.  Use gamma < 1 or a horizon."
        )
    v[live] = x
    return v


def evaluate(m, policy):
    """The exact value ``v_pi`` of any policy, as an ``(S,)`` array.

    On a finite-horizon model this is the value over the whole horizon from
    time 0.  See :mod:`aniate.mdp.policies` for accepted policies.
    """
    return values_by_time(m, policy)[0]


def values_by_time(m, policy, T=None):
    """``(T + 1, S)``: the value of ``policy`` with ``t`` steps already taken.

    Constant in ``t`` for a stationary policy on an infinite horizon; zero
    once a finite horizon has run out.  ``T`` defaults to the horizon (or 0).
    """
    M = policy_matrix(m, policy)
    if M.ndim == 2 and m.horizon is None:
        v = _solve_linear(m, *_chain(m, M))
        return np.tile(v, ((T or 0) + 1, 1))
    H = M.shape[0] if M.ndim == 3 else m.horizon
    if m.horizon is not None and M.ndim == 3 and H != m.horizon:
        raise ValueError(f"policy covers {H} steps but the model's horizon is {m.horizon}")
    V = np.zeros((max(H, T or 0) + 1, m.S))
    for t in reversed(range(H)):
        P_pi, r_pi = _chain(m, M[t] if M.ndim == 3 else M)
        V[t] = r_pi + m.gamma * (P_pi @ V[t + 1])
    return V[: (H if T is None else T) + 1]


def _no_horizon(m, name):
    if m.horizon is not None:
        raise ValueError(
            f"this model has a finite horizon, so its optimal policy depends on time; "
            f"{name} does not apply.  Use backward_induction (what solve() picks)."
        )


def _logged(sol):
    _log.info("solve", sol.solver, f"{sol.iterations} iterations", f"{sol.wall_time * 1e3:.1f} ms",
              f"start value {sol.start_value:.6g}", f"bound {sol.bound:.2g}")
    return sol


def policy_iteration(m, max_iter=1_000):
    """Exact evaluation plus greedy improvement.  Needs gamma < 1."""
    _no_horizon(m, "policy_iteration")
    if m.gamma >= 1.0:
        raise ValueError("policy_iteration needs gamma < 1; use value_iteration")
    t0 = time.perf_counter()
    policy = m.greedy(np.zeros(m.S))
    rows = np.arange(m.S)
    iterations, reason = 0, "max_iterations"
    for _ in range(max_iter):
        iterations += 1
        v = _solve_linear(m, *_chain(m, one_hot(policy, m.A)))
        q = m.q_from_v(v)
        best = q.argmax(axis=1)
        current = q[rows, policy]
        # Switch only on a real improvement; exact ties otherwise cycle forever.
        improves = q[rows, best] > current + 1e-12 * np.maximum(1.0, np.abs(current))
        new_policy = np.where(improves, best, policy)
        if np.array_equal(new_policy, policy):
            reason = "policy_stable"
            break
        policy = new_policy
    return _logged(Solution(m, policy, v, solver="policy_iteration", iterations=iterations,
                            reason=reason, wall_time=time.perf_counter() - t0, polished=True))


def value_iteration(m, tol=1e-8, max_iter=100_000):
    """Repeated Bellman backups, then exact evaluation of the greedy policy.

    Stops when ``span(v_new - v) < tol (1 - gamma) / gamma`` (gamma < 1) or
    ``max |v_new - v| < tol`` (gamma = 1).
    """
    _no_horizon(m, "value_iteration")
    t0 = time.perf_counter()
    g = m.gamma
    thresh = np.inf if g == 0.0 else tol if g >= 1.0 else tol * (1.0 - g) / g
    v = np.zeros(m.S)
    iterations, reason = 0, "max_iterations"
    for _ in range(max_iter):
        iterations += 1
        v_new = m.q_from_v(v).max(axis=1)
        d = v_new - v
        size = float(np.abs(d).max()) if g >= 1.0 else float(d.max() - d.min())
        v = v_new
        if size < thresh:
            reason = "converged"
            break
    if reason != "converged" and tol > 0:
        _log.warning("solve", f"value_iteration stopped at max_iter={max_iter} before converging; "
                              "the bound still holds for the returned policy")

    policy = m.greedy(v)
    try:
        v = _solve_linear(m, *_chain(m, one_hot(policy, m.A)))
        polished = True
    except ValueError:
        polished = False     # gamma = 1 and the greedy policy is improper: keep the iterate
    return _logged(Solution(m, policy, v, solver="value_iteration", iterations=iterations,
                            reason=reason, wall_time=time.perf_counter() - t0, polished=polished))


def backward_induction(m, horizon=None):
    """Finite horizon: ``T`` exact backward passes and a ``(T, S)`` policy."""
    t0 = time.perf_counter()
    T = m.horizon if horizon is None else int(horizon)
    if T is None or T <= 0:
        raise ValueError("backward_induction needs a positive horizon: pass horizon= or set it on the MDP")
    v = np.zeros(m.S)
    policy = np.zeros((T, m.S), dtype=int)
    for t in reversed(range(T)):
        q = m.q_from_v(v)
        policy[t] = q.argmax(axis=1)
        v = q.max(axis=1)
    return _logged(Solution(m, policy, v, solver="backward_induction", iterations=T,
                            reason="horizon", wall_time=time.perf_counter() - t0, exact=True))


SOLVERS = {
    "policy_iteration": policy_iteration,
    "value_iteration": value_iteration,
    "backward_induction": backward_induction,
}


def solve(m, method="auto", **kwargs):
    """Solve a model.

    ``"auto"``: backward induction for a finite horizon, policy iteration for
    gamma < 1, value iteration for gamma = 1.
    """
    if not hasattr(m, "P") and hasattr(m, "product"):
        m = m.product
    if method == "auto":
        method = ("backward_induction" if m.horizon is not None
                  else "policy_iteration" if m.gamma < 1.0 else "value_iteration")
    if method not in SOLVERS:
        raise ValueError(f"unknown solver {method!r}; choose from {sorted(SOLVERS)}")
    return SOLVERS[method](m, **kwargs)
