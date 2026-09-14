"""Static checks on a model: the specification mistakes people actually make.

Every issue names the offending states and actions by label.  Errors mean
the object is not an MDP (rows that do not sum to one, a terminal state that
leaks); warnings mean it is an MDP that probably is not the one you meant.
"""

from __future__ import annotations

import numpy as np

from aniate import log as _log

__all__ = ["validate", "ValidationReport", "Issue", "reachable"]

ROW_SUM_TOL = 1e-9
MAX_EXAMPLES = 5


class Issue:
    """One problem, with a few named examples."""

    def __init__(self, code, severity, message, examples=(), count=1):
        self.code = code
        self.severity = severity            # "error" | "warning"
        self.message = message
        self.examples = list(examples)[:MAX_EXAMPLES]
        self.count = int(count)

    def line(self):
        ex = f"; e.g. {', '.join(self.examples[:3])}" if self.examples else ""
        return f"[{self.code}] {self.message}{ex}"

    def describe(self):
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "count": self.count, "examples": list(self.examples)}

    def __str__(self):
        lines = [f"[{self.code}] {self.message}"]
        lines += [f"    - {e}" for e in self.examples]
        if self.count > len(self.examples) and self.examples:
            lines.append(f"    ... {self.count - len(self.examples)} more")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Issue {self.severity}:{self.code} x{self.count}>"


class ValidationReport:
    def __init__(self, issues):
        self.issues = list(issues)

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self):
        return not self.errors

    def describe(self):
        return {"kind": "ValidationReport", "ok": self.ok, "errors": len(self.errors),
                "warnings": len(self.warnings), "issues": [i.describe() for i in self.issues]}

    def __repr__(self):
        if not self.issues:
            return "ValidationReport: clean"
        head = f"ValidationReport: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([head] + [str(i) for i in self.issues])


def _sa(m, s, a):
    from aniate.mdp.text import state_str

    return f"{state_str(m, s)} / {m.actions[a]}"


def reachable(m):
    """Bool mask of states reachable from the initial distribution under some action."""
    adj = None
    for a in range(m.A):
        mat = m.P[a].copy()
        mat.data = (mat.data != 0).astype(float)
        adj = mat if adj is None else adj + mat
    seen = m.initial > 0
    back = adj.T.tocsr()
    while True:
        nxt = seen | ((back @ seen.astype(float)) > 0)
        if (nxt == seen).all():
            return seen
        seen = nxt


def validate(m, raise_on_error=False):
    """Check a model.  Returns a :class:`ValidationReport`.

    With ``raise_on_error`` any error raises ``ValueError`` listing every
    issue; otherwise every issue is logged.
    """
    issues = []

    bad = []
    for a in range(m.A):
        rows = np.repeat(np.arange(m.S), np.diff(m.P[a].indptr))
        nf = ~np.isfinite(m.P[a].data) | ~np.isfinite(m.r_next[a])
        bad += [_sa(m, int(rows[k]), a) for k in np.flatnonzero(nf)]
    if not np.all(np.isfinite(m.R)) and not bad:
        s, a = np.argwhere(~np.isfinite(m.R))[0]
        bad.append(_sa(m, int(s), int(a)))
    if bad:
        issues.append(Issue("non_finite", "error",
                            "probabilities or rewards contain NaN or infinity", bad, len(bad)))
        return _finish(ValidationReport(issues), raise_on_error)

    neg, dead, off = [], [], []
    for a in range(m.A):
        rows = np.repeat(np.arange(m.S), np.diff(m.P[a].indptr))
        for k in np.flatnonzero(m.P[a].data < 0):
            neg.append(f"{_sa(m, int(rows[k]), a)}: {m.P[a].data[k]:.4g}")
        sums = np.asarray(m.P[a].sum(axis=1)).ravel()
        for s in np.flatnonzero(np.abs(sums) <= ROW_SUM_TOL):
            dead.append(_sa(m, int(s), a))
        for s in np.flatnonzero((np.abs(sums - 1.0) > ROW_SUM_TOL) & (np.abs(sums) > ROW_SUM_TOL)):
            off.append(f"{_sa(m, int(s), a)}: sums to {sums[s]:.6g}")
    if neg:
        issues.append(Issue("negative_probability", "error",
                            f"{len(neg)} transition probabilities are negative", neg, len(neg)))
    if dead:
        issues.append(Issue("dead_end", "error",
                            f"{len(dead)} state-action pairs have no successor at all "
                            "(every action must be defined in every state)", dead, len(dead)))
    if off:
        issues.append(Issue("row_sum", "error",
                            f"{len(off)} state-action pairs have probabilities that do not "
                            f"sum to 1 (tolerance {ROW_SUM_TOL:g})", off, len(off)))

    if m._stray_rewards:
        issues.append(Issue("reward_on_impossible_transition", "warning",
                            f"{m._stray_rewards} reward entries sit on transitions with "
                            "probability 0 and will never be collected", count=m._stray_rewards))

    if m._terminal_given:
        leaks = []
        for t in np.flatnonzero(m.terminal):
            for a in range(m.A):
                idx, prob, rew = m._row(int(t), a)
                live = prob != 0
                if not (np.all(idx[live] == t) and np.isclose(prob[live].sum(), 1.0)
                        and np.all(rew[live] == 0)):
                    leaks.append(_sa(m, int(t), a))
        if leaks:
            issues.append(Issue(
                "terminal_not_absorbing", "error",
                "terminal states must loop to themselves with probability 1 and reward 0, "
                "otherwise the simulator stops where the solver keeps counting", leaks, len(leaks)))

    absorbing = np.ones(m.S, dtype=bool)
    for a in range(m.A):
        absorbing &= np.isclose(m.P[a].diagonal(), 1.0, atol=1e-12)
    paying = absorbing & ~m.terminal & (np.abs(m.R).max(axis=1) > 0)
    if paying.any():
        ex = [f"{m.states[int(s)]}: reward {m.R[s].max():.4g} forever" for s in np.flatnonzero(paying)]
        issues.append(Issue("rewarding_absorbing_state", "warning",
                            "some states can never be left and pay a reward on every step; "
                            "if they are meant to end the episode, mark them terminal",
                            ex, int(paying.sum())))

    seen = reachable(m)
    if not getattr(m, "_skip_reachability", False) and not seen.all():
        idx = np.flatnonzero(~seen)
        issues.append(Issue("unreachable_state", "warning",
                            f"{idx.size} state(s) cannot be reached from the initial distribution",
                            [str(m.states[int(s)]) for s in idx], int(idx.size)))

    if m.gamma == 1.0 and m.horizon is None and not (m.terminal & seen).any():
        issues.append(Issue("unbounded_value", "error",
                            "gamma=1 with no reachable terminal state: returns are unbounded. "
                            "Set gamma < 1, a horizon, or a terminal state."))

    return _finish(ValidationReport(issues), raise_on_error)


def _finish(report, raise_on_error):
    if raise_on_error and report.errors:
        raise ValueError(
            "the model is malformed:\n"
            + "\n".join(str(i) for i in report.errors)
            + "\n(pass validate='warn' to build it anyway and inspect m.report)"
        )
    for issue in report.issues:
        _log.warning("model", issue.line())
    return report
