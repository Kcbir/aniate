"""Cube and NCube: solvers for the Travelling Thief Problem with Drone and the problems inside it.

All of them take the same inputs, and what you leave out decides the problem.

    from aniate import cube, ncube

    nodes = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]   # depot first
    profits, weights = [50, 60, 70, 80, 40], [2, 3, 4, 5, 2]

    ncube(nodes, profits, weights, drone=True)                 # the neural accelerator
    cube(nodes, profits, weights, drone=True)                  # the metaheuristic
    cube.milp(nodes, profits, weights, drone=True)             # the exact model

    cube(nodes)                                                # TSP: a truck
    cube(nodes, drone=True)                                    # TSP-D: a truck and a drone
    cube(nodes, profits, weights)                              # TTP: a truck and a knapsack

    env = cube.env(nodes, profits, weights, drone=True)        # the simulator, one decision at a time
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

from .core import Problem, check_solution, evaluate, nn_route
from .metaheuristic import run

__all__ = ["cube", "ncube", "Plan"]

SPEED, SLOWEST, DRONE_FACTOR = 1.0, 0.1, 2.0
NO_FLIGHT = 1e-9       # an endurance no flight can meet: this is what switches the drone off


@dataclass(frozen=True)
class Plan:
    """A finished plan. Node numbers are positions in `nodes`, so 0 is the depot."""
    problem: str          # "TSP", "TSP-D", "TTP" or "TTP-D"
    route: tuple          # the truck's path, from the depot back to it
    flights: tuple        # drone flights as (launch, customer, landing)
    items: tuple          # customers whose item is taken
    time: float           # when everyone is back
    profit: float         # value of the items taken
    objective: float      # profit minus rent times time
    seconds: float        # how long the solver ran
    solver: str           # "neural", "metaheuristic" or "milp"
    gap: float | None = None   # the exact model's optimality gap in percent; 0 means proved optimal

    def __str__(self):
        parts = [f"time {self.time:,.2f}"]
        if self.problem in ("TTP", "TTP-D"):
            parts = [f"objective {self.objective:,.2f}", f"profit {self.profit:,.2f}", *parts, f"{len(self.items)} items"]
        if self.problem in ("TSP-D", "TTP-D"):
            parts.append(f"{len(self.flights)} flights")
        if self.gap is not None:
            parts.append(f"gap {self.gap:.2f}%")
        return f"{self.problem} plan: " + ", ".join(parts)


def _points(nodes):
    points = [tuple(float(v) for v in node) for node in nodes]
    if len(points) < 2:
        raise ValueError("nodes needs the depot and at least one customer")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("each node must be a finite (x, y) pair")
    return points


def _column(values, n, name):
    column = [float(v) for v in values]
    if len(column) != n:
        raise ValueError(f"{name} has {len(column)} values for {n} customers; the depot takes none")
    if any(v < 0 or not math.isfinite(v) for v in column):
        raise ValueError(f"{name} must be finite and not negative")
    return column


def _problem(nodes, profits, weights, drone, endurance, capacity, rent, drone_speed, dist):
    """Turn what the caller gave into the paper's model, filling in everything they left out."""
    if (profits is None) != (weights is None):
        raise ValueError("give profits and weights together, or neither")
    points = _points(nodes)
    n = len(points) - 1
    order = points + points[:1]
    if dist is None:
        table = tuple(tuple(math.dist(a, b) for b in order) for a in order)
    else:
        rows = [[float(v) for v in row] for row in dist]
        if len(rows) != n + 1 or any(len(row) != n + 1 for row in rows):
            raise ValueError(f"dist must be {n + 1} by {n + 1}, depot first")
        keep = list(range(n + 1)) + [0]
        table = tuple(tuple(rows[i][j] for j in keep) for i in keep)

    items = profits is not None
    p = [0.0] + _column(profits, n, "profits") + [0.0] if items else [0.0] * (n + 2)
    w = [0.0] + _column(weights, n, "weights") + [0.0] if items else [0.0] * (n + 2)
    flying = drone or endurance is not None or drone_speed is not None
    if endurance is not None and endurance <= 0:
        raise ValueError("endurance must be positive; leave the drone out for no drone")
    if capacity is None:
        capacity = sum(w) / 2 if sum(w) > 0 else 1.0
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if rent is None:
        rent = 1.0
        if sum(p) > 0:   # the rate at which driving a plain tour costs half of all the profit
            tour = (0, *nn_route(_Distances(table), range(1, n + 1)), n + 1)
            rent = sum(p) / 2 / max(sum(table[a][b] for a, b in zip(tour, tour[1:])) / SPEED, 1e-9)
    if rent < 0:
        raise ValueError("rent must not be negative")

    P = Problem(n=n, dist=table, p=tuple(p), w=tuple(w), Wcap=float(capacity), vmax=SPEED, vmin=SLOWEST,
                vD=float(drone_speed or SPEED * DRONE_FACTOR), R=float(rent),
                ED=(None if endurance is None else float(endurance)) if flying else NO_FLIGHT)
    name = {(False, False): "TSP", (True, False): "TSP-D", (False, True): "TTP", (True, True): "TTP-D"}[flying, items]
    return P, name, order


class _Distances:
    def __init__(self, table):
        self.dist = table


def _finish(P, name, truck, sorties, z, started, solver, gap=None):
    problem = check_solution(P, truck, sorties, z)
    if problem is not None:
        raise RuntimeError(f"the {solver} solver returned an infeasible plan: {problem}")
    home = lambda v: 0 if v == P.n + 1 else v
    clock = replace(P, p=(0.0,) * len(P.p), R=1.0)
    return Plan(problem=name, route=(0, *truck, 0), flights=tuple((home(a), b, home(c)) for a, b, c in sorties),
                items=tuple(i for i in range(1, P.n + 1) if z[i]), time=-evaluate(clock, truck, sorties, z),
                profit=sum(P.p[i] for i in range(1, P.n + 1) if z[i]), objective=evaluate(P, truck, sorties, z),
                seconds=time.perf_counter() - started, solver=solver, gap=gap)


_OPTIONS = """
    nodes        (x, y) pairs, depot first.
    profits      value of each customer's item, depot left out.
    weights      weight of each customer's item, depot left out.
    drone        True to fly a drone from the truck.
    endurance    longest flight, launch to customer to landing. Unlimited by default; setting it turns the drone on.
    capacity     what the truck can carry. Half the total weight by default.
    rent         cost per unit of time. By default, the rate at which a plain tour costs half of all the profit.
    drone_speed  twice the empty truck's speed by default.
    dist         your own distance matrix, depot first, if straight lines are wrong.
"""


def cube(nodes, profits=None, weights=None, *, drone=False, endurance=None, capacity=None, rent=None,
         drone_speed=None, seconds=None, seed=0, dist=None):
    started = time.perf_counter()
    P, name, _ = _problem(nodes, profits, weights, drone, endurance, capacity, rent, drone_speed, dist)
    budget = max(1.0, 0.5 * P.n) if seconds is None else float(seconds)
    if budget <= 0:
        raise ValueError("seconds must be positive")
    truck, sorties, z, _ = run(P, budget, seed=seed)
    return _finish(P, name, truck, sorties, z, started, "metaheuristic")


def milp(nodes, profits=None, weights=None, *, drone=False, endurance=None, capacity=None, rent=None,
         drone_speed=None, seconds=3600, dist=None):
    started = time.perf_counter()
    P, name, _ = _problem(nodes, profits, weights, drone, endurance, capacity, rent, drone_speed, dist)
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    from .exact import solve
    truck, sorties, z, gap = solve(P, name in ("TSP-D", "TTP-D"), seconds)
    return _finish(P, name, truck, sorties, z, started, "milp", gap)


def env(nodes, profits=None, weights=None, *, drone=False, endurance=None, capacity=None, rent=None,
        drone_speed=None, dist=None):
    P, _, order = _problem(nodes, profits, weights, drone, endurance, capacity, rent, drone_speed, dist)
    from .network import instance_of
    from .network.simulator import TTPDEnv
    return TTPDEnv(n=P.n, instance=instance_of(P, order))


def ncube(nodes, profits=None, weights=None, *, drone=False, endurance=None, capacity=None, rent=None,
          drone_speed=None, network=True, dist=None):
    started = time.perf_counter()
    P, name, order = _problem(nodes, profits, weights, drone, endurance, capacity, rent, drone_speed, dist)
    from .network import accelerate
    truck, sorties, z = accelerate(P, order, network)
    return _finish(P, name, truck, sorties, z, started, "neural")


def train(problems, **options):
    """Train the neural accelerator on your own problems; pass the result to `ncube` as `network`.

    problems   a number of customers for random problems, or a list of (nodes, profits, weights), all one size.
    examples   how many random problems to make.
    seconds    search time for each one. Half a second per customer by default.
    epochs     passes over the plans, stopping early once held-out plans stop improving.
    save       a path to write the trained weights to.
    endurance  the longest drone flight in your problems. Unlimited by default.
    """
    from .network import train as run_training
    return run_training(problems, **options)


cube.__doc__ = ("""Plan with the metaheuristic. Leave out the items for a TSP or TSP-D, and the drone for a TSP or TTP.
""" + _OPTIONS + """    seconds      search time. Half a second per customer by default; the paper gives it 15.
    seed         random seed for the search.
""")
ncube.__doc__ = ("""Plan with the neural accelerator. It takes the same inputs as `cube`.
""" + _OPTIONS + """    network      True for our weights, a path to your own, or what `ncube.train` returned.
""")
milp.__doc__ = ("""Plan with the paper's exact model. It takes the same inputs as `cube`, needs Gurobi, and suits small problems.
""" + _OPTIONS + """    seconds      time limit, one hour by default. The plan's `gap` is 0 when it is proved optimal.
""")
env.__doc__ = ("""The paper's simulator, stepped one decision at a time. It takes the same inputs as `cube`.

`reset()` returns (observation, info), and `step(action)` returns (observation, reward, terminated, truncated, info).
Rewards add up to the plan's objective. Customers are numbered from 1, the start is 0 and the return to the depot
is the last node. An action is a dict. Anything left out is 0, except `k`, which is -1 for no launch, and `j`, which is the final depot:

    rejoin   1 to land the drone at the truck's position
    z_drone  1 to keep the item the landing drone brought
    z_curr   1 to take the item at the truck's position
    k        the customer to launch the drone to, or -1
    j        the node the truck drives to next

`info["action_masks"]` says which choices are allowed now.
""" + _OPTIONS)

cube.milp = milp
cube.env = env
ncube.train = train
