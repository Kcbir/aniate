"""The object a solver returns."""

from __future__ import annotations

import numpy as np

__all__ = ["Solution"]


class Solution:
    """An optimal (or near-optimal) policy, its value, and how sure we are.

    Attributes
    ----------
    policy : (S,) ints, or (T, S) for a finite horizon
    value : (S,) floats
        The exact value of ``policy`` -- never an unconverged iterate.
    bound : float
        Guaranteed upper bound on ``max_s v*(s) - value(s)``: the most reward
        ``policy`` can lose against the optimum, from any state.  0 for exact
        methods; ``inf`` where no finite guarantee exists (gamma = 1).
    residual : float
        ``max_s |(T value)(s) - value(s)|``, the Bellman residual.
    """

    def __init__(self, mdp, policy, value, *, solver, iterations, reason, wall_time,
                 polished=False, exact=False):
        self.mdp = mdp
        self.policy = np.asarray(policy, dtype=int)
        self.value = np.asarray(value, dtype=float)
        self.solver = solver
        self.iterations = int(iterations)
        self.reason = reason
        self.wall_time = float(wall_time)
        if exact:
            self.residual, self.bound = 0.0, 0.0
        else:
            self.residual = float(np.abs(mdp.q_from_v(self.value).max(axis=1) - self.value).max())
            # With value == v_pi exactly, v* - v_pi <= ||T v_pi - v_pi|| / (1 - gamma).
            self.bound = (self.residual / (1.0 - mdp.gamma)
                          if polished and mdp.gamma < 1.0 else float("inf"))

    @property
    def time_indexed(self):
        return self.policy.ndim == 2

    @property
    def start_value(self):
        """Expected return from the model's initial distribution."""
        return float(self.mdp.initial @ self.value)

    def action(self, state, t=0):
        """The action label chosen in ``state`` (at time ``t`` if time-indexed)."""
        s = self.mdp.index(state)
        a = self.policy[t, s] if self.time_indexed else self.policy[s]
        return self.mdp.actions[int(a)]

    def as_dict(self, t=0):
        """``{state: action}`` by label."""
        row = self.policy[t] if self.time_indexed else self.policy
        return {self.mdp.states[s]: self.mdp.actions[int(a)] for s, a in enumerate(row)}

    def describe(self):
        """The solution as a JSON-serialisable dict."""
        from aniate.mdp.text import state_str

        d = {
            "kind": "Solution",
            "solver": self.solver,
            "iterations": self.iterations,
            "wall_time_s": round(self.wall_time, 6),
            "stopped": self.reason,
            "start_value": self.start_value,
            "value_range": [float(self.value.min()), float(self.value.max())],
            "bound": self.bound,
            "residual": self.residual,
            "time_indexed": self.time_indexed,
        }
        if not self.time_indexed:
            m = self.mdp
            if m.S <= 50:
                d["policy"] = {state_str(m, s): str(m.actions[int(a)])
                               for s, a in enumerate(self.policy) if not m.terminal[s]}
            else:
                counts = np.bincount(self.policy, minlength=m.A)
                d["action_counts"] = {str(m.actions[a]): int(counts[a]) for a in range(m.A)}
        return d

    def __repr__(self):
        from aniate.mdp.text import state_str

        lines = [
            f"Solution({self.solver}, {self.iterations} iterations, {self.wall_time * 1e3:.1f} ms)",
            f"  start value : {self.start_value:.6g}",
            f"  value range : [{self.value.min():.6g}, {self.value.max():.6g}]",
            f"  bound       : {self.bound:.3g}  (no state loses more than this vs. optimal)",
        ]
        if self.time_indexed:
            lines.append(f"  policy      : time-indexed over {self.policy.shape[0]} steps; "
                         "use .action(state, t)")
        elif self.mdp.S <= 12:
            lines.append("  policy      : " + ", ".join(
                f"{state_str(self.mdp, s)} -> {self.mdp.actions[int(a)]}"
                for s, a in enumerate(self.policy) if not self.mdp.terminal[s]))
        return "\n".join(lines)
