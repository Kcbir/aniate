"""The TTP-D model from the paper: the plan evaluator, the drone scheduler, and the moves the search makes.

A plan is three things. ``truck`` is the visiting order, ``sorties`` are (launch, customer, landing)
triples, and ``z`` marks the items taken. Node 0 is the depot, and node n + 1 is the depot again on return.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from itertools import islice

eps = 1e-9

@dataclass
class Problem:
    n: int
    dist: tuple
    p: tuple
    w: tuple
    Wcap: float
    vmax: float
    vmin: float
    vD: float
    R: float
    ED: float | None = None   # longest flight, launch to customer to landing; None is unlimited

    @property
    def src(self) -> int:
        return 0

    @property
    def snk(self) -> int:
        return self.n + 1

    @property
    def slope(self) -> float:
        return (self.vmax - self.vmin) / self.Wcap


def evaluate(P: Problem, truck, sorties, z):
    route = (0,) + tuple(truck) + (P.n + 1,)
    pos = {}
    for i, nd in enumerate(route):
        pos[nd] = i

    rejoin_at = {}
    launch_at = {}
    if sorties:
        try:
            sord = sorted(sorties, key=lambda s: pos[s[0]])
        except KeyError:
            return None
        last_end = 0
        ED = P.ED
        for idx, (L, D, J) in enumerate(sord):
            pL = pos.get(L)
            pJ = pos.get(J)
            if pL is None or pJ is None or pL >= pJ or pL < last_end:
                return None
            # Drone endurance: round-trip launch->target->rejoin must fit the budget.
            if ED is not None and P.dist[L][D] + P.dist[D][J] > ED + eps:
                return None
            last_end = pJ
            launch_at[pL] = idx
            rejoin_at[pJ] = idx
    else:
        sord = ()

    dist = P.dist
    w = P.w
    p = P.p
    n = P.n
    Wcap = P.Wcap + eps
    vD = P.vD
    vmax = P.vmax
    slope = P.slope

    tau = 0.0
    Wt = 0.0
    prof = 0.0
    lt = [0.0] * len(sord)
    last = len(route) - 1

    for i, nd in enumerate(route):
        si = rejoin_at.get(i)
        if si is not None:
            L, D, _ = sord[si]
            arr = lt[si] + (dist[L][D] + dist[D][nd]) / vD
            if arr > tau:
                tau = arr
            if z[D]:
                Wt += w[D]
                if Wt > Wcap:
                    return None
                prof += p[D]
        if 0 < nd <= n and z[nd]:
            Wt += w[nd]
            if Wt > Wcap:
                return None
            prof += p[nd]
        si = launch_at.get(i)
        if si is not None:
            lt[si] = tau
        if i < last:
            tau += dist[nd][route[i + 1]] / (vmax - Wt * slope)

    return prof - P.R * tau


def check_solution(P: Problem, truck, sorties, z) -> str | None:
    targets = [s[1] for s in sorties]
    if len(set(truck)) != len(truck):
        return "duplicate truck customers"
    if len(set(targets)) != len(targets):
        return "duplicate sortie targets"
    cust = set(range(1, P.n + 1))
    if set(truck) | set(targets) != cust or set(truck) & set(targets):
        return "truck/drone sets do not partition the customers"
    for D in targets:
        if not (1 <= D <= P.n):
            return f"sortie target {D} is not a customer"
    if z[0] != 0 or z[P.n + 1] != 0:
        return "z set at a depot"
    for i in cust:
        if z[i] not in (0, 1):
            return f"z[{i}] not binary"
    if evaluate(P, truck, sorties, z) is None:
        return "evaluate() rejects (anchors/overlap/capacity)"
    return None


dp_cap = 12


def dp_schedule(P: Problem, truck, targets, z):
    targets = tuple(targets)
    k = len(targets)
    if k == 0:
        v = evaluate(P, truck, (), z)
        return (v, ()) if v is not None else None
    if k > dp_cap:
        return None

    route = (0,) + tuple(truck) + (P.n + 1,)
    nr = len(route)
    last = nr - 1
    dist = P.dist
    w = P.w
    vmax = P.vmax
    slope = P.slope
    vD = P.vD
    ED = P.ED

    tot_w = sum(w[i] for i in range(1, P.n + 1) if z[i])
    if tot_w > P.Wcap + eps:
        return None

    prefix = [0.0] * nr
    acc = 0.0
    for i, nd in enumerate(route):
        if 0 < nd <= P.n and z[nd]:
            acc += w[nd]
        prefix[i] = acc

    wD = [w[t] if z[t] else 0.0 for t in targets]
    full = (1 << k) - 1

    dWtab = [0.0] * (full + 1)
    for mask in range(1, full + 1):
        low = mask & -mask
        dWtab[mask] = dWtab[mask ^ low] + wD[low.bit_length() - 1]

    dp = [dict() for _ in range(nr)]
    dp[0][0] = (0.0, None)

    for i in range(nr):
        if not dp[i]:
            continue
        if i == last:
            continue
        for mask, (tau, _) in list(dp[i].items()):
            dW = dWtab[mask]
            tarr = [0.0] * nr
            t_run = tau
            for j in range(i + 1, nr):
                v_t = vmax - (prefix[j - 1] + dW) * slope
                t_run += dist[route[j - 1]][route[j]] / v_t
                tarr[j] = t_run
            old = dp[i + 1].get(mask)
            if old is None or tarr[i + 1] < old[0] - 1e-12:
                dp[i + 1][mask] = (tarr[i + 1], (i, mask, None))
            here_nd = route[i]
            mi = full & ~mask
            while mi:
                bbit = mi & -mi
                bi = bbit.bit_length() - 1
                mi ^= bbit
                D = targets[bi]
                d_out = dist[here_nd][D]
                drow = dist[D]
                nmask = mask | bbit
                for j in range(i + 1, nr):
                    # Drone endurance budget on the round-trip distance (skip if violated).
                    if ED is not None and d_out + drow[route[j]] > ED + eps:
                        continue
                    arr = tau + (d_out + drow[route[j]]) / vD
                    t_j = tarr[j] if tarr[j] > arr else arr
                    dpj = dp[j]
                    old = dpj.get(nmask)
                    if old is None or t_j < old[0] - 1e-12:
                        dpj[nmask] = (t_j, (i, mask, (here_nd, D, route[j])))

    end = dp[last].get(full)
    if end is None:
        return None
    tau_end, _ = end

    sorties = []
    i, mask = last, full
    while True:
        _, parent = dp[i][mask]
        if parent is None:
            break
        pi, pmask, sortie = parent
        if sortie is not None:
            sorties.append(sortie)
        i, mask = pi, pmask
    sorties.reverse()

    prof = sum(P.p[i] for i in range(1, P.n + 1) if z[i])
    return prof - P.R * tau_end, tuple(sorties)


def _anchor_nodes(sorties):
    s = set()
    for L, _, J in sorties:
        s.add(L)
        s.add(J)
    return s


def _repaired(truck, sorties, P):
    pos = {nd: i for i, nd in enumerate((0,) + tuple(truck) + (P.n + 1,))}
    out = []
    changed = False
    for L, D, J in sorties:
        pL, pJ = pos.get(L), pos.get(J)
        if pL is None or pJ is None:
            return None
        if pL > pJ:
            out.append((J, D, L))
            changed = True
        else:
            out.append((L, D, J))
    return tuple(out) if changed else None


def _order(seq, rng):
    """Iterate ``seq`` in random order when ``rng`` is given, else in order.

    Used by the O(n^2)/O(n^3) route neighborhoods so that, combined with the
    ``max_cands`` islice in vnd(), the first slice is a uniform random subset of
    moves rather than always the same low-index prefix. With rng=None the
    neighborhood is exhaustive and deterministic (small-n behaviour unchanged).
    """
    if rng is None:
        return seq
    seq = list(seq)
    rng.shuffle(seq)
    return seq


def nb_zflip(P, truck, sorties, z, rng=None):
    for i in _order(range(1, P.n + 1), rng):
        z2 = list(z)
        z2[i] ^= 1
        yield truck, sorties, tuple(z2)


def nb_swap_truck(P, truck, sorties, z, rng=None):
    m = len(truck)
    for a in _order(range(m - 1), rng):
        for b in range(a + 1, m):
            t2 = list(truck)
            t2[a], t2[b] = t2[b], t2[a]
            t2 = tuple(t2)
            yield t2, sorties, z
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                yield t2, rep, z


def nb_two_opt(P, truck, sorties, z, rng=None):
    m = len(truck)
    for a in _order(range(m - 1), rng):
        for b in range(a + 1, m):
            t2 = truck[:a] + truck[a:b + 1][::-1] + truck[b + 1:]
            yield t2, sorties, z
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                yield t2, rep, z


def nb_oropt(P, truck, sorties, z, rng=None):
    m = len(truck)
    for seg_len in (1, 2, 3):
        if seg_len > m:
            break
        for a in _order(range(m - seg_len + 1), rng):
            seg = truck[a:a + seg_len]
            rest = truck[:a] + truck[a + seg_len:]
            for b in range(len(rest) + 1):
                t2 = rest[:b] + seg + rest[b:]
                if t2 == truck:
                    continue
                yield t2, sorties, z
                rep = _repaired(t2, sorties, P)
                if rep is not None:
                    yield t2, rep, z


def nb_reanchor(P, truck, sorties, z, rng=None):
    route = (0,) + tuple(truck) + (P.n + 1,)
    nr = len(route)
    for si, (L, D, J) in enumerate(sorties):
        z_flip = list(z)
        z_flip[D] ^= 1
        z_flip = tuple(z_flip)
        for ia in _order(range(nr - 1), rng):
            for ib in range(ia + 1, nr):
                L2, J2 = route[ia], route[ib]
                s2 = sorties[:si] + ((L2, D, J2),) + sorties[si + 1:]
                if L2 != L or J2 != J:
                    yield truck, s2, z
                yield truck, s2, z_flip


def nb_truck_to_drone(P, truck, sorties, z, rng=None):
    anchors = _anchor_nodes(sorties)
    for t in _order([x for x in truck if x not in anchors], rng):
        t2 = tuple(x for x in truck if x != t)
        route = (0,) + t2 + (P.n + 1,)
        nr = len(route)
        z_flip = list(z)
        z_flip[t] ^= 1
        z_flip = tuple(z_flip)
        for ia in range(nr - 1):
            for ib in range(ia + 1, nr):
                s2 = sorties + ((route[ia], t, route[ib]),)
                yield t2, s2, z
                yield t2, s2, z_flip


def nb_drone_to_truck(P, truck, sorties, z, rng=None):
    for si, (L, D, J) in enumerate(sorties):
        s2 = sorties[:si] + sorties[si + 1:]
        z_flip = list(z)
        z_flip[D] ^= 1
        z_flip = tuple(z_flip)
        for b in range(len(truck) + 1):
            t2 = truck[:b] + (D,) + truck[b:]
            yield t2, s2, z
            yield t2, s2, z_flip


def nb_swap_td(P, truck, sorties, z, rng=None):
    anchors = _anchor_nodes(sorties)
    for ti, t in enumerate(truck):
        if t in anchors:
            continue
        for si, (L, D, J) in enumerate(sorties):
            t2 = list(truck)
            t2[ti] = D
            t2 = tuple(t2)
            s2 = sorties[:si] + ((L, t, J),) + sorties[si + 1:]
            for zt in (0, 1):
                for zd in (0, 1):
                    z2 = list(z)
                    z2[t], z2[D] = zt, zd
                    yield t2, s2, tuple(z2)


def nb_dp_resched(P, truck, sorties, z, rng=None):
    targets = tuple(s[1] for s in sorties)
    out = dp_schedule(P, truck, targets, z)
    if out is not None and out[1] != sorties:
        yield truck, out[1], z


flip_dp_cap = 10


def nb_flip_dp(P, truck, sorties, z, rng=None):
    targets = tuple(s[1] for s in sorties)
    if len(targets) + 1 > flip_dp_cap:
        return
    for ti, t in enumerate(truck):
        t2 = truck[:ti] + truck[ti + 1:]
        nt = targets + (t,)
        z_flip = list(z)
        z_flip[t] ^= 1
        z_flip = tuple(z_flip)
        for z2 in (z, z_flip):
            out = dp_schedule(P, t2, nt, z2)
            if out is not None:
                yield t2, out[1], z2


neighborhoods = (
    nb_zflip,
    nb_swap_truck,
    nb_two_opt,
    nb_oropt,
    nb_dp_resched,
    nb_reanchor,
    nb_swap_td,
    nb_drone_to_truck,
    nb_truck_to_drone,
    nb_flip_dp,
)


def vnd(P, truck, sorties, z, obj, rng, counter=None,
        deadline=None, max_cands=None):
    """First-improvement Variable Neighborhood Descent.

    Two scalability levers (both optional, off by default so small-n behaviour
    is identical to before):

      * ``deadline`` -- a ``time.perf_counter()`` timestamp. The descent checks
        it *inside* the candidate loop, so it can never overrun the caller's
        time budget (the old VND only checked between neighborhoods, letting a
        single n=100 scan blow ~1000s past the deadline).
      * ``max_cands`` -- cap on candidates *generated and evaluated* per
        neighborhood pass. When set, each neighborhood is asked to emit moves in
        random order (the ``rng`` argument) and we ``islice`` the first
        ``max_cands``, so we never even *build* the full O(n^2)-O(n^3) move set
        (the real bottleneck at n>=50, e.g. nb_oropt's ~3*n^2 candidates).
        ``max_cands=None`` keeps the exhaustive, deterministic descent.
    """
    k = 0
    n_nb = len(neighborhoods)
    order_rng = rng if max_cands is not None else None
    while k < n_nb:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        gen = neighborhoods[k](P, truck, sorties, z, order_rng)
        if max_cands is not None:
            gen = islice(gen, max_cands)
        improved = False
        for t2, s2, z2 in gen:
            v = evaluate(P, t2, s2, z2)
            if counter is not None:
                counter[0] += 1
            if v is not None and v > obj + 1e-9:
                truck, sorties, z, obj = t2, s2, z2, v
                improved = True
                break
            if deadline is not None and time.perf_counter() >= deadline:
                return truck, sorties, z, obj
        k = 0 if improved else k + 1
    return truck, sorties, z, obj


def nn_route(P, customers):
    unvis = list(customers)
    cur = 0
    out = []
    while unvis:
        nxt = min(unvis, key=lambda j: P.dist[cur][j])
        out.append(nxt)
        unvis.remove(nxt)
        cur = nxt
    return tuple(out)


def construct_nn(P):
    truck = nn_route(P, range(1, P.n + 1))
    z = tuple(0 for _ in range(P.n + 2))
    return truck, (), z


def _shake_once(P, truck, sorties, z, rng):
    for _ in range(30):
        op = rng.randint(0, 5)
        t2, s2, z2 = truck, sorties, z

        if op == 0 and P.n > 0:
            i = rng.randint(1, P.n)
            zl = list(z)
            zl[i] ^= 1
            z2 = tuple(zl)

        elif op == 1 and sorties:
            si = rng.randrange(len(sorties))
            L, D, J = sorties[si]
            route = (0,) + truck + (P.n + 1,)
            ia = rng.randrange(len(route) - 1)
            ib = rng.randrange(ia + 1, len(route))
            s2 = sorties[:si] + ((route[ia], D, route[ib]),) + sorties[si + 1:]

        elif op == 2 and truck:
            t = rng.choice(truck)
            if t in _anchor_nodes(sorties):
                continue
            t2 = tuple(x for x in truck if x != t)
            route = (0,) + t2 + (P.n + 1,)
            ia = rng.randrange(len(route) - 1)
            ib = rng.randrange(ia + 1, len(route))
            s2 = sorties + ((route[ia], t, route[ib]),)

        elif op == 3 and sorties:
            si = rng.randrange(len(sorties))
            D = sorties[si][1]
            s2 = sorties[:si] + sorties[si + 1:]
            b = rng.randint(0, len(truck))
            t2 = truck[:b] + (D,) + truck[b:]

        elif op == 4 and len(truck) >= 2:
            a = rng.randrange(len(truck))
            t = truck[a]
            rest = truck[:a] + truck[a + 1:]
            b = rng.randint(0, len(rest))
            t2 = rest[:b] + (t,) + rest[b:]
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                s2 = rep

        elif op == 5 and len(truck) >= 3:
            a = rng.randrange(len(truck) - 1)
            b = rng.randrange(a + 1, len(truck))
            t2 = truck[:a] + truck[a:b + 1][::-1] + truck[b + 1:]
            rep = _repaired(t2, sorties, P)
            if rep is not None:
                s2 = rep
        else:
            continue

        if (t2, s2, z2) == (truck, sorties, z):
            continue
        v = evaluate(P, t2, s2, z2)
        if v is not None:
            return t2, s2, z2, v
    return None
