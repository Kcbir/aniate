"""Node-link diagrams drawn with matplotlib alone.

Layout is layered: states are ranked by breadth-first distance from where
episodes start, so a diagram reads left to right in the order things happen.
Past ``MAX_NODES`` a node-link drawing is unreadable, so these refuse.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from aniate.vis.style import GRAY, INK, PALETTE, axes, plain, text

__all__ = ["graph", "automaton", "MAX_NODES"]

MAX_NODES = 30
DX, DY = 2.8, 1.9      # spacing between layers / within a layer
SHADE = "0.93"         # initial states
SINK = "0.80"          # non-accepting sinks


def _layout(n, succ, roots):
    """Positions for nodes 0..n-1: layers by BFS depth, ordered by parent barycentre."""
    depth, queue = {}, deque()
    for r in roots:
        if r not in depth:
            depth[r] = 0
            queue.append(r)
    while queue:
        u = queue.popleft()
        for v in sorted(succ[u]):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    last = max(depth.values(), default=-1) + 1
    for u in range(n):
        depth.setdefault(u, last)

    layers, preds = {}, {u: [] for u in range(n)}
    for u in range(n):
        layers.setdefault(depth[u], []).append(u)
        for v in succ[u]:
            preds[v].append(u)
    pos = {}
    for d in sorted(layers):
        nodes = layers[d]
        if d > 0:
            def bary(u):
                ys = [pos[p][1] for p in preds[u] if p in pos]
                return sum(ys) / len(ys) if ys else 0.0
            nodes.sort(key=bary, reverse=True)
        for k, u in enumerate(nodes):
            pos[u] = (d * DX, ((len(nodes) - 1) / 2 - k) * DY)
    return pos


def _wrap(label, width=10):
    label, parts, token = str(label), [], ""
    if len(label) <= width:
        return label
    for ch in label:
        token += ch
        if ch in "_=| +@,":
            parts.append(token)
            token = ""
    parts.append(token)
    lines, line = [], ""
    for p in parts:
        if line and len(line) + len(p) > width:
            lines.append(line)
            line = p
        else:
            line += p
    lines.append(line)
    return "\n".join(x for x in lines if x)


def _node(ax, xy, label, face="white", double=False):
    """Draw a state; returns its radius, which grows with the label."""
    from matplotlib.patches import Circle

    wrapped = _wrap(label)
    lines = wrapped.split("\n")
    radius = max(0.42, 0.058 * max(len(x) for x in lines) + 0.05, 0.17 * len(lines) + 0.12)
    ax.add_patch(Circle(xy, radius, facecolor=face, edgecolor=INK, lw=0.8, zorder=3))
    if double:
        ax.add_patch(Circle(xy, radius + 0.07, facecolor="none", edgecolor=INK, lw=0.6, zorder=3))
    ax.text(xy[0], xy[1], plain(wrapped), ha="center", va="center", zorder=4, **text(7))
    return radius + (0.07 if double else 0.0)


def _label(ax, xy, label):
    ax.text(xy[0], xy[1], plain(label), ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), **text(6.5))


def _extent(ax):
    """Points drawn outside node positions (arc apexes, loops), so limits include them."""
    if not hasattr(ax, "_aniate_extent"):
        ax._aniate_extent = []
    return ax._aniate_extent


def _edge(ax, p, q, label, color, rad=0.0, r0=0.42, r1=0.42, head=True, lw=0.8):
    from matplotlib.patches import FancyArrowPatch

    p, q = np.asarray(p, float), np.asarray(q, float)
    dist = float(np.hypot(*(q - p)))
    if dist < 1e-9:
        return
    u = (q - p) / dist
    a, b = p + u * r0, q - u * r1
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>" if head else "-", mutation_scale=8,
                                 color=color, lw=lw, zorder=2, shrinkA=0, shrinkB=0))
    d = b - a
    apex = (a + b) / 2 + 0.5 * rad * np.array([d[1], -d[0]])
    _extent(ax).append(apex)
    if label:
        _label(ax, apex, label)


def _loop(ax, xy, radius, angle, label, color, lw=0.8):
    """A self-loop sitting on the node's rim at ``angle``."""
    cen = np.asarray(xy, float)
    direction = np.array([math.cos(angle), math.sin(angle)])
    rl = 0.28
    centre = cen + direction * (radius + rl * 0.55)
    back = angle + math.pi
    t = np.linspace(back + 0.95, back + 2 * math.pi - 0.95, 40)
    pts = centre + rl * np.stack([np.cos(t), np.sin(t)], axis=1)
    ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, zorder=2)
    ax.annotate("", xy=pts[-1], xytext=pts[-4],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=8), zorder=2)
    n_lines = label.count("\n") + 1 if label else 0
    tip = centre + direction * (rl + 0.12 + 0.11 * n_lines)
    _extent(ax).append(tip)
    if label:
        _label(ax, tip, label)


def _finish(ax, pos):
    pts = list(pos.values()) + [tuple(p) for p in getattr(ax, "_aniate_extent", [])]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.9)
    ax._aniate_extent = []
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def _too_big(n, what, hint):
    raise ValueError(f"{n} {what} is past the {MAX_NODES}-node limit where a diagram stays "
                     f"readable. {hint}")


def graph(model, policy=None, ax=None, rewards=True):
    """Transition diagram of a small MDP.

    Circles are states (initial shaded, terminal double-ringed).  A
    deterministic action is one arrow labelled with its name; a stochastic
    action goes through a dot and fans out with probabilities; rewards are in
    parentheses.  With ``policy`` only the chosen actions are drawn.
    """
    from aniate.mdp.env import _unwrap
    from aniate.mdp.policies import policy_matrix
    from aniate.mdp.text import state_str

    m = _unwrap(model)
    if m.S > MAX_NODES:
        _too_big(m.S, "states", "Use vis.grid for gridworlds, or print(model.summary()).")
    ax = axes(ax, figsize=(6, 3.8))
    colors = [PALETTE[a % len(PALETTE)] for a in range(m.A)]

    chosen = np.ones((m.S, m.A), bool)
    if policy is not None:
        M = policy_matrix(m, policy)
        if M.ndim != 2:
            raise ValueError("vis.graph needs a stationary policy")
        chosen = M > 0

    succ = [set() for _ in range(m.S)]
    for s in range(m.S):
        for a in range(m.A):
            if chosen[s, a]:
                idx, prob, _ = m._row(s, a)
                succ[s].update(int(j) for j, p in zip(idx, prob) if p > 0 and j != s)
    pos = _layout(m.S, succ, [int(i) for i in np.flatnonzero(m.initial > 0)])
    radius = {s: _node(ax, pos[s], state_str(m, s), face=SHADE if m.initial[s] > 0 else "white",
                       double=bool(m.terminal[s]))
              for s in range(m.S)}

    def tag(r):
        return f" ({r:+.3g})" if rewards and r != 0 else ""

    pair_count, loop_count = {}, {}
    for s in range(m.S):
        if m.terminal[s]:
            continue
        acts = [a for a in range(m.A) if chosen[s, a]]
        for k, a in enumerate(acts):
            idx, prob, rew = m._row(s, a)
            live = [(int(j), float(p), float(r)) for j, p, r in zip(idx, prob, rew) if p > 0]
            name, col = str(m.actions[a]), colors[a]
            if len(live) == 1:
                j, _, r = live[0]
                if j == s:
                    n = loop_count.get(s, 0)
                    loop_count[s] = n + 1
                    _loop(ax, pos[s], radius[s], math.pi / 2 - n * math.pi / 2.2, name + tag(r), col)
                else:
                    n = pair_count.get((s, j), 0)
                    pair_count[(s, j)] = n + 1
                    _edge(ax, pos[s], pos[j], name + tag(r), col, rad=0.15 + 0.2 * n,
                          r0=radius[s], r1=radius[j])
                continue

            here = np.asarray(pos[s], float)
            others = [(np.asarray(pos[j], float), p) for j, p, _ in live if j != s]
            d = (sum(p * xy for xy, p in others) / sum(p for _, p in others) - here) if others \
                else np.array([0.0, 1.0])
            ang = math.atan2(d[1], d[0]) - 0.45 + (k - (len(acts) - 1) / 2) * 0.8
            dot = here + (radius[s] + 0.85) * np.array([math.cos(ang), math.sin(ang)])
            ax.plot(*dot, "o", color=col, ms=3, zorder=3)
            _edge(ax, here, dot, name, col, r0=radius[s], r1=0.05, head=False)
            for j, p, r in live:
                backwards = pos[j][0] < dot[0] - 1e-9 or j == s
                _edge(ax, dot, pos[j], f"{p:.2g}{tag(r)}", col, rad=0.35 if backwards else 0.1,
                      r0=0.05, r1=radius[j], lw=0.7)
    return _finish(ax, pos)


def automaton(machine, ax=None):
    """Draw a Mealy machine: named memory states, event-labelled edges, outputs after '/'.

    The initial state has an incoming arrow; accepting states are
    double-ringed; non-accepting sinks are shaded.
    """
    if hasattr(machine, "machine"):
        machine = machine.machine
    Q = list(machine.states)
    if len(Q) > MAX_NODES:
        _too_big(len(Q), "memory states", "Print machine.summary(), or draw the components separately.")
    ax = axes(ax, figsize=(6, 3.8))
    idx = {q: i for i, q in enumerate(Q)}
    edges = machine.edges()
    succ = [set() for _ in Q]
    for src, dst, _ in edges:
        if src != dst:
            succ[idx[src]].add(idx[dst])
    pos = _layout(len(Q), succ, [idx[machine.initial]])

    partial = bool(machine.accepting) and machine.accepting != set(Q)
    radius = {}
    for q in Q:
        bad_sink = partial and q not in machine.accepting and machine.is_sink(q)
        face = SINK if bad_sink else SHADE if q == machine.initial else "white"
        radius[q] = _node(ax, pos[idx[q]], q, face=face, double=partial and q in machine.accepting)

    merged = {}
    for src, dst, label in edges:
        merged.setdefault((src, dst), []).append(label)
    for (src, dst), group in merged.items():
        label = "\n".join(group)
        if src == dst:
            _loop(ax, pos[idx[src]], radius[src], math.pi / 2, label, GRAY)
        else:
            _edge(ax, pos[idx[src]], pos[idx[dst]], label, GRAY,
                  rad=0.2 if (dst, src) in merged else 0.0, r0=radius[src], r1=radius[dst])

    x0, y0 = pos[idx[machine.initial]]
    r0 = radius[machine.initial]
    _edge(ax, (x0 - r0 - 0.8, y0), (x0, y0), "", GRAY, r0=0.0, r1=r0)
    return _finish(ax, pos)
