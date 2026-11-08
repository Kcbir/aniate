"""The paper's exact model: a mixed-integer linear programme, solved with Gurobi.

The truck's travel time is the one nonlinear part, because speed depends on load. The model bounds it from
above with a piecewise-linear curve, so it never flatters a plan. The plan that comes back is then timed by
the same evaluator as every other solver.
"""
from __future__ import annotations

from .core import nn_route

SEGMENTS = 10


def solve(P, flying, seconds):
    """Solve `P` and return (truck, sorties, z, the optimality gap in percent)."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise ImportError('cube.milp needs Gurobi: pip install "aniate[milp]"') from error

    n, src, snk = P.n, 0, P.n + 1
    V, N = range(n + 2), range(1, n + 1)
    arcs = [(i, j) for i in V for j in V if i != j]
    d = {(i, j): P.dist[i][j] for i, j in arcs}
    p, w, slope = P.p, P.w, (P.vmax - P.vmin) / P.Wcap

    heaviest = min(sum(w), P.Wcap)
    loads = [k * heaviest / SEGMENTS for k in range(SEGMENTS + 1)]
    slowness = [1.0 / (P.vmax - load * slope) for load in loads]

    tour = (src, *nn_route(P, N), snk)
    horizon = sum(d[a, b] for a, b in zip(tour, tour[1:])) / P.vmin
    has_items = sum(w) > 0 or sum(p) > 0

    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as m:
        m.Params.TimeLimit = float(seconds)
        m.Params.MIPGap = 1e-4
        m.Params.MIPFocus = 3
        m.Params.Heuristics = 0.2
        m.Params.Cuts = 2
        m.Params.Presolve = 2
        m.Params.IntFeasTol = 1e-7

        leaving = [v for v in V if v != snk]
        xT = m.addVars(arcs, vtype=GRB.BINARY)
        xD = m.addVars(arcs, vtype=GRB.BINARY)
        yT = m.addVars(N, vtype=GRB.BINARY)
        yD = m.addVars(N, vtype=GRB.BINARY, ub=1.0 if flying else 0.0)
        yC = m.addVars(N, vtype=GRB.BINARY, ub=1.0 if flying else 0.0)
        z = m.addVars(N, vtype=GRB.BINARY, ub=1.0 if has_items else 0.0)
        carried = m.addVars([(k, j) for k in N for j in V if j != k], lb=0.0, ub=1.0)
        load = m.addVars(V, lb=0.0, ub=heaviest)
        arrive = m.addVars(V, lb=0.0)
        beta = m.addVars([(i, k) for i in leaving for k in range(SEGMENTS + 1)], lb=0.0, ub=1.0)
        phi = m.addVars(leaving, lb=slowness[0], ub=slowness[-1])

        m.setObjective(gp.quicksum(p[i] * z[i] for i in N) - P.R * arrive[snk], GRB.MAXIMIZE)

        # Every customer is served by the truck, by the drone, or by both meeting there.
        m.addConstrs(yT[i] + yD[i] + yC[i] == 1 for i in N)

        for x in (xT, xD):
            m.addConstr(gp.quicksum(x[src, j] for j in V if j != src) == 1)
            m.addConstr(gp.quicksum(x[i, snk] for i in V if i != snk) == 1)
            m.addConstrs(x[j, src] == 0 for j in V if j != src)
            m.addConstrs(x[snk, i] == 0 for i in V if i != snk)
        m.addConstrs(gp.quicksum(xT[i, j] for j in V if j != i) == gp.quicksum(xT[j, i] for j in V if j != i) for i in N)
        m.addConstrs(gp.quicksum(xT[i, j] for j in V if j != i) == yT[i] + yC[i] for i in N)
        m.addConstrs(gp.quicksum(xD[j, i] for j in V if j != i) == yD[i] + yC[i] for i in N)
        m.addConstrs(gp.quicksum(xD[i, j] for j in V if j != i) == yD[i] + yC[i] for i in N)
        m.addConstrs(xD[i, j] <= yC[i] + yC[j] for i in N for j in N if i != j)

        if flying and P.ED is not None:
            m.addConstrs(xD[i, k] + xD[k, j] + yD[k] <= 2
                         for k in N for i in V for j in V
                         if i != k and j != k and i != j and i != snk and j != src and d[i, k] + d[k, j] > P.ED)

        # What the drone collected rides the truck from the landing onwards.
        m.addConstrs(carried[k, j] >= z[k] + xD[k, j] - 1 for k, j in carried)
        m.addConstrs(carried[k, j] <= z[k] for k, j in carried)
        m.addConstrs(carried[k, j] <= xD[k, j] for k, j in carried)

        m.addConstr(load[src] == 0)
        m.addConstr(arrive[src] == 0)
        for i in leaving:
            for j in V:
                if j == i or j == src:
                    continue
                landed = gp.quicksum(carried[k, j] * w[k] for k in N if k != j)
                picked = w[j] * z[j] if j in N else 0.0
                m.addGenConstrIndicator(xT[i, j], True, load[j] - load[i] - picked - landed, GRB.EQUAL, 0.0)
                m.addGenConstrIndicator(xT[i, j], True, arrive[j] - arrive[i] - d[i, j] * phi[i], GRB.GREATER_EQUAL, 0.0)

        for i in leaving:
            m.addConstr(load[i] == gp.quicksum(beta[i, k] * loads[k] for k in range(SEGMENTS + 1)))
            m.addConstr(phi[i] == gp.quicksum(beta[i, k] * slowness[k] for k in range(SEGMENTS + 1)))
            m.addConstr(gp.quicksum(beta[i, k] for k in range(SEGMENTS + 1)) == 1)
            m.addSOS(GRB.SOS_TYPE2, [beta[i, k] for k in range(SEGMENTS + 1)])

        m.addConstrs(arrive[i] + d[i, j] / P.vD <= arrive[j] + horizon * (1 - xD[i, j] + xT[i, j]) for i, j in arcs)

        # Valid bounds on the finishing time, which keep the search small.
        m.addConstr(arrive[snk] >= slowness[0] * gp.quicksum(d[a] * xT[a] for a in arcs))
        m.addConstr(arrive[snk] >= gp.quicksum(d[a] * xD[a] for a in arcs) / P.vD)
        m.addConstrs(arrive[snk] >= (d[src, i] + d[i, snk]) * ((yT[i] + yC[i]) / P.vmax + yD[i] / P.vD) for i in N)

        # Start from a plain nearest-neighbour tour, packing greedily along it.
        for x in (xT, xD):
            for a in arcs:
                x[a].Start = 0.0
        for a in zip(tour, tour[1:]):
            xT[a].Start = 1.0
        xD[src, snk].Start = 1.0
        room = P.Wcap
        for i in N:
            yT[i].Start, yD[i].Start, yC[i].Start = 1.0, 0.0, 0.0
        for i in tour[1:-1]:
            take = has_items and w[i] <= room
            z[i].Start = 1.0 if take else 0.0
            room -= w[i] if take else 0.0

        try:
            m.optimize()
        except gp.GurobiError as error:
            if error.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
                raise RuntimeError(f"{n} customers is too many for Gurobi's size-limited licence; "
                                   "a full or academic licence removes the limit, or use cube") from error
            raise
        if m.SolCount == 0:
            raise RuntimeError(f"the exact model found no plan within {seconds:g} seconds")

        truck, here = [], src
        while True:
            here = next(j for j in V if j != here and xT[here, j].X > 0.5)
            if here == snk:
                break
            truck.append(here)
        sorties = tuple((next(a for a in V if a != k and xD[a, k].X > 0.5), k,
                         next(c for c in V if c != k and xD[k, c].X > 0.5)) for k in N if yD[k].X > 0.5)
        chosen = tuple(int(i in N and z[i].X > 0.5) for i in V)
        return tuple(truck), sorties, chosen, 100.0 * m.MIPGap
