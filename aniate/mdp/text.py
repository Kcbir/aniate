"""Plain-text rendering of models: the terminal is a visualisation too."""

from __future__ import annotations

import numpy as np

__all__ = ["mdp_summary", "state_str"]


def state_str(m, i):
    """Human label of state position ``i``; product states read ``world @ memory``."""
    label = m.states[int(i)]
    if hasattr(m, "machine") and isinstance(label, tuple) and len(label) == 2:
        return f"{label[0]} @ {label[1]}"
    return str(label)


def _listing(items, k=8):
    items = [str(x) for x in items]
    shown = ", ".join(items[:k])
    return shown + (f", ... ({len(items)} total)" if len(items) > k else "")


def mdp_summary(m, max_states=12):
    h = "infinite" if m.horizon is None else f"{m.horizon} steps"
    lines = [
        f"{type(m).__name__}: {m.S} states, {m.A} actions, gamma={m.gamma:g}, horizon={h}",
        f"  states   : {_listing(state_str(m, i) for i in range(m.S))}",
        f"  actions  : {_listing(m.actions)}",
    ]
    init = np.flatnonzero(m.initial)
    lines.append("  initial  : " + _listing(
        (f"{state_str(m, i)} ({m.initial[i]:.3g})" if len(init) > 1 else state_str(m, i))
        for i in init))
    term = np.flatnonzero(m.terminal)
    lines.append("  terminal : " + (_listing(state_str(m, i) for i in term) if term.size else "none"))
    lines.append("")
    lines.append("  state / action -> next state   prob   reward")

    shown = range(min(m.S, max_states))
    sw = max([len(state_str(m, j)) for j in range(min(m.S, 200))] + [4])
    aw = max(len(str(x)) for x in m.actions)
    for s in shown:
        if m.terminal[s]:
            lines.append(f"  {state_str(m, s)}  (terminal)")
            continue
        lines.append(f"  {state_str(m, s)}")
        for a in range(m.A):
            idx, prob, rew = m._row(s, a)
            first = True
            for j, p, r in zip(idx, prob, rew):
                if p == 0:
                    continue
                head = f"    {str(m.actions[a]):<{aw}} -> " if first else " " * (aw + 8)
                lines.append(f"{head}{state_str(m, j):<{sw}}  {p:<5.3g}  {r:+.4g}")
                first = False
            if first:
                lines.append(f"    {str(m.actions[a]):<{aw}} -> (no successors)")
    if m.S > max_states:
        lines.append(f"  ... {m.S - max_states} more states; summary(max_states=...) shows them")
    return "\n".join(lines)
