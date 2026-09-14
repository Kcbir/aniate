"""Rich display in notebooks: result objects render as compact HTML tables.

Every public result object defines ``_repr_html_``, so evaluating one as the
last expression of a Jupyter cell shows a table instead of plain text:

    m = aniate.problems.gridworld()
    m                     # states, actions, initial and terminal states, transitions
    m.solve()             # solver, start value, bound, and the policy by state

The markup uses inline styles only and inherits the notebook's text colour,
so it reads in light and dark themes alike.  Long tables stop after
``MAX_ROWS`` rows and state how many were left out.
"""

from __future__ import annotations

import html as _html
import math

import numpy as np

__all__ = ["to_html", "MAX_ROWS"]

MAX_ROWS = 20

_RULE = "1px solid rgba(127, 127, 127, 0.3)"
_CELL = f"padding: 3px 14px 3px 0; border-bottom: {_RULE}; text-align: left; vertical-align: top;"
_HEAD = _CELL + " font-weight: 600; opacity: 0.75;"
_GOOD, _BAD = "#1b7837", "#b35806"
_FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


class _Raw(str):
    """Markup inserted without escaping."""


def _e(x):
    return x if isinstance(x, _Raw) else _html.escape(str(x))


def _num(x, digits=6):
    if x is None:
        return "-"
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}g}"


def _count(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _more(n, word):
    return f"{_count(n, word)} not shown" if n > 0 else None


def _dot(*parts):
    return " · ".join(str(p) for p in parts if p not in (None, ""))


def _listing(items, k=6):
    items = [str(x) for x in items]
    return ", ".join(items[:k]) + (f", and {len(items) - k:,} more" if len(items) > k else "")


def _code(x):
    return _Raw(f"<code>{_e(x)}</code>")


def _badge(ok, yes, no):
    colour = _GOOD if ok else _BAD
    return _Raw(f'<span style="display: inline-block; padding: 0 6px; border: 1px solid {colour}; '
                f'border-radius: 3px; color: {colour}; font-size: 0.85em;">{_e(yes if ok else no)}</span>')


def _table(head, rows, note=None):
    parts = ['<table style="border-collapse: collapse; margin: 8px 0 4px;">']
    if head:
        parts.append("<thead><tr>" + "".join(f'<th style="{_HEAD}">{_e(h)}</th>' for h in head)
                     + "</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        parts.append("<tr>" + "".join(f'<td style="{_CELL}">{_e(c)}</td>' for c in row) + "</tr>")
    parts.append("</tbody></table>")
    if note:
        parts.append(f'<div style="opacity: 0.7; font-size: 0.9em;">{_e(note)}</div>')
    return "".join(parts)


def _fields(pairs):
    rows = [(_Raw(f'<span style="opacity: 0.75;">{_e(k)}</span>'), v) for k, v in pairs if v is not None]
    return _table(None, rows)


def _box(kind, subtitle, *sections, badge=None):
    head = f"<strong>{_e(kind)}</strong>"
    if badge:
        head += f" {badge}"
    if subtitle:
        head += f' <span style="opacity: 0.75;">{_e(subtitle)}</span>'
    return (f'<div style="font-family: {_FONT}; font-size: 13px; line-height: 1.45; max-width: 860px;">'
            f"<div>{head}</div>" + "".join(s for s in sections if s) + "</div>")


def _issues(issues):
    rows = []
    for issue in issues:
        examples = f" (for example {'; '.join(issue.examples[:3])})" if issue.examples else ""
        rows.append((_code(issue.code), issue.severity, issue.message + examples, f"{issue.count:,}"))
    return _table(("code", "severity", "message", "count"), rows)


# -- models ----------------------------------------------------------------------


def _mdp(m):
    from aniate.mdp.text import state_str

    horizon = "infinite" if m.horizon is None else f"{m.horizon} steps"
    stored = sum(int(np.count_nonzero(p.data)) for p in m.P)
    subtitle = _dot(_count(m.S, "state"), _count(m.A, "action"), f"gamma {m.gamma:g}", f"horizon {horizon}",
                    _count(stored, "transition"))
    badge = None if m.report is None else _badge(m.report.ok, "valid", _count(len(m.report.errors), "error"))

    rewards = np.concatenate([r for r in m.r_next if r.size] or [np.zeros(1)])
    initial, terminal = np.flatnonzero(m.initial), np.flatnonzero(m.terminal)
    fields = _fields([
        ("actions", _listing(m.actions)),
        ("initial", _listing(state_str(m, i) if initial.size == 1 else f"{state_str(m, i)} ({m.initial[i]:.3g})"
                             for i in initial)),
        ("terminal", _listing(state_str(m, i) for i in terminal) if terminal.size else "none"),
        ("reward range", f"{_num(rewards.min())} to {_num(rewards.max())}"),
    ])

    rows, shown = [], 0
    for s in range(m.S):
        if len(rows) >= MAX_ROWS:
            break
        if m.terminal[s]:
            continue
        for a in range(m.A):
            idx, prob, rew = m._row(s, a)
            for j, p, r in zip(idx, prob, rew):
                if p != 0 and len(rows) < MAX_ROWS:
                    rows.append((state_str(m, s), m.actions[a], state_str(m, j), _num(p, 4), _num(r, 4)))
    for P in m.P:
        rows_of = np.repeat(np.arange(m.S), np.diff(P.indptr).astype(np.intp))
        shown += int(np.count_nonzero((P.data != 0) & ~m.terminal[rows_of]))
    table = _table(("state", "action", "next state", "probability", "reward"), rows,
                   _more(shown - len(rows), "transition"))

    issues = _issues(m.report.issues) if m.report is not None and m.report.issues else ""
    return _box(type(m).__name__, subtitle, fields, table, issues, badge=badge)


def _validation(report):
    subtitle = _dot(_count(len(report.errors), "error"), _count(len(report.warnings), "warning"))
    body = _issues(report.issues) if report.issues else _table(None, [("No issues were found.",)])
    return _box("ValidationReport", subtitle, body, badge=_badge(report.ok, "valid", "invalid"))


# -- results ---------------------------------------------------------------------


def _solution(sol):
    from aniate.mdp.text import state_str

    m = sol.mdp
    subtitle = _dot(sol.solver, _count(sol.iterations, "iteration"), f"{sol.wall_time * 1e3:.1f} ms")
    fields = _fields([
        ("start value", _num(sol.start_value)),
        ("value range", f"{_num(sol.value.min())} to {_num(sol.value.max())}"),
        ("bound", f"{_num(sol.bound, 3)} (no state loses more than this against the optimal policy)"),
        ("residual", _num(sol.residual, 3)),
    ])
    if sol.time_indexed:
        body = _fields([("policy", f"time-indexed over {sol.policy.shape[0]} steps; use action(state, t)")])
    else:
        live = [s for s in range(m.S) if not m.terminal[s]]
        rows = [(state_str(m, s), m.actions[int(sol.policy[s])], _num(sol.value[s])) for s in live[:MAX_ROWS]]
        body = _table(("state", "action", "value"), rows, _more(len(live) - len(rows), "state"))
    return _box("Solution", subtitle, fields, body)


def _episodes(runs):
    d = runs.describe()
    q = d["return_quantiles"]
    model_value = d["model_value"]
    fields = _fields([
        ("mean return", f"{_num(d['mean_return'])} ± {_num(d['stderr'], 2)} (standard error)"),
        ("model value", _num(model_value) if model_value is not None else "not available for this policy"),
        ("return quantiles", f"5%: {_num(q['p5'])}, 50%: {_num(q['p50'])}, 95%: {_num(q['p95'])}"),
        ("mean length", f"{d['mean_length']:.1f} steps"),
        ("episodes ended", f"{d['terminated']:,} terminated, {d['truncated']:,} truncated"),
        ("truncation bias", _num(d["truncation_bias"], 3)),
    ])
    return _box("Episodes", _dot(_count(d["episodes"], "episode"), _count(d["steps"], "step")), fields)


def _check(report):
    st = report.stats
    n = len(report.issues)
    if st.get("model_return") is not None:
        returns = (f"sampled {_num(st['sampled_return'])} ± {_num(st['sampled_stderr'], 2)}, "
                   f"model {_num(st['model_return'])}")
    else:
        returns = f"not compared ({st.get('return_note', 'not available')})"
    fields = _fields([
        ("coverage", f"{st['pairs_visited']:,} of {st['pairs_reachable']:,} reachable state-action pairs visited"),
        ("transitions", f"{st['pairs_tested']:,} pairs tested at alpha {st['alpha']:g}; "
                        f"{st['pairs_untested']:,} deterministic or too rarely visited to test"),
        ("rewards", f"{st['rewards_compared']:,} distinct transitions compared"),
        ("returns", returns),
    ])
    issues = _issues(report.issues) if report.issues else ""
    return _box("CheckReport", _dot(_count(st["episodes"], "episode"), _count(st["steps"], "step")),
                fields, issues, badge=_badge(report.ok, "passed", f"failed with {_count(n, 'issue')}"))


# -- non-Markovian problems ----------------------------------------------------------


def _memory_table(machine):
    rows = []
    for q in machine.states:
        role = [r for r, yes in (("initial", q == machine.initial), ("accepting", q in machine.accepting)) if yes]
        rows.append((_code(q), ", ".join(role), machine.meaning.get(q, "")))
    return _table(("memory state", "role", "meaning"), rows[:MAX_ROWS], _more(len(rows) - MAX_ROWS, "memory state"))


def _mealy(machine):
    edges = machine.edges()
    subtitle = _dot(_count(len(machine.states), "memory state"),
                    f"events {', '.join(machine.events)}" if machine.events else "no events")
    table = _table(("from", "on", "to"), [(_code(src), label, _code(dst)) for src, dst, label in edges[:MAX_ROWS]],
                   _more(len(edges) - MAX_ROWS, "transition"))
    return _box(f"Mealy {machine.name}", subtitle, _memory_table(machine), table)


def _nmdp(task):
    p = task.product
    subtitle = _dot(f"{task.world.S:,} world states x {len(task.machine)} memory states = {p.S:,} product states",
                    _count(task.world.A, "action"), f"gamma {task.gamma:g}")
    fields = _fields([
        ("machine", task.machine.name),
        ("events", ", ".join(task.machine.events) or "none"),
        ("rewards", "world reward + automaton output" if task.combine == "add" else "automaton output only"),
    ])
    return _box("NMDP", subtitle, fields, _memory_table(task.machine))


def _memory_policy(pi):
    rows = [
        ("machine", pi.machine.name),
        ("memory", _Raw(f"{_code(pi.memory)} {_e(pi.explain()) if pi.explain() != pi.memory else ''}")),
        ("start value", _num(pi.start_value) if pi.start_value is not None else None),
    ]
    sections = []
    if pi.table.ndim == 2:
        examples, n = pi.product.policy_disagreements(pi._flat)
        rows.append(("memory matters", f"the action depends on memory in {n:,} of {pi.world.S:,} world states"))
        if examples:
            sections.append(_table(("world state", "action by memory state"), [
                (e["state"], "; ".join(f"{q}: {a}" for q, a in e["action_by_memory"].items())) for e in examples
            ]))
    return _box("MemoryPolicy", None, _fields(rows), *sections)


def to_html(obj):
    """HTML for an aniate object; the ``_repr_html_`` methods call this."""
    from aniate.mdp.check import CheckReport
    from aniate.mdp.core import MDP
    from aniate.mdp.env import Episodes
    from aniate.mdp.solution import Solution
    from aniate.mdp.validate import ValidationReport
    from aniate.nmdp.machine import Mealy
    from aniate.nmdp.policy import MemoryPolicy
    from aniate.nmdp.task import NMDP

    renderers = ((MDP, _mdp), (Solution, _solution), (Episodes, _episodes), (CheckReport, _check),
                 (ValidationReport, _validation), (NMDP, _nmdp), (Mealy, _mealy), (MemoryPolicy, _memory_policy))
    for kind, render in renderers:
        if isinstance(obj, kind):
            return render(obj)
    raise TypeError(f"no HTML display is defined for {type(obj).__name__}")
