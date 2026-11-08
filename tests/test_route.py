"""Cube and NCube: the four problems, the defaults, the simulator, and what they refuse. The search runs for a fraction of a second."""
import pytest

from aniate import cube, mdp, ncube, nmdp, problems
from aniate.route import _problem
from aniate.route.core import check_solution, evaluate
from aniate.route.network.simulator import actions_of

NODES = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]
PROFITS, WEIGHTS = [50, 60, 70, 80, 40], [2, 3, 4, 5, 2]
QUICK = dict(seconds=0.2)


@pytest.mark.parametrize("args, options, name", [
    ((NODES,), {}, "TSP"),
    ((NODES,), {"drone": True}, "TSP-D"),
    ((NODES, PROFITS, WEIGHTS), {}, "TTP"),
    ((NODES, PROFITS, WEIGHTS), {"drone": True}, "TTP-D"),
    ((NODES, PROFITS, WEIGHTS), {"endurance": 6}, "TTP-D"),
])
def test_what_you_leave_out_decides_the_problem(args, options, name):
    plan = cube(*args, **options, **QUICK)
    assert plan.problem == name
    assert plan.route[0] == plan.route[-1] == 0
    served = set(plan.route[1:-1]) | {flight[1] for flight in plan.flights}
    assert served == {1, 2, 3, 4, 5}
    assert plan.time > 0
    assert plan.objective == pytest.approx(plan.profit - (1 if name in ("TSP", "TSP-D") else _rent()) * plan.time)


def _rent():
    return _problem(NODES, PROFITS, WEIGHTS, True, None, None, None, None, None)[0].R


def test_no_drone_means_no_flights_and_no_items_means_none_taken():
    assert cube(NODES, PROFITS, WEIGHTS, **QUICK).flights == ()
    assert cube(NODES, drone=True, **QUICK).items == ()


def test_a_drone_never_slows_the_truck_down():
    assert cube(NODES, drone=True, **QUICK).time <= cube(NODES, **QUICK).time + 1e-9


def test_flights_keep_to_the_endurance():
    limit = 6
    plan = cube(NODES, drone=True, endurance=limit, **QUICK)
    for launch, customer, landing in plan.flights:
        reach = [NODES[launch], NODES[customer], NODES[landing]]
        total = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(reach, reach[1:]))
        assert total <= limit + 1e-9


def test_the_truck_never_carries_more_than_its_capacity():
    plan = cube(NODES, PROFITS, WEIGHTS, drone=True, capacity=4, **QUICK)
    assert sum(WEIGHTS[i - 1] for i in plan.items) <= 4


def test_the_evaluator_agrees_with_a_hand_computed_tour():
    P = _problem([(0, 0), (3, 0), (3, 4)], None, None, False, None, None, None, None, None)[0]
    assert evaluate(P, (1, 2), (), (0, 0, 0, 0)) == pytest.approx(-12)
    assert check_solution(P, (1, 2), (), (0, 0, 0, 0)) is None


def test_a_distance_matrix_replaces_straight_lines():
    table = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert cube([(0, 0), (9, 9), (5, 1)], dist=table, **QUICK).time == pytest.approx(3)


@pytest.mark.parametrize("call, message", [
    (lambda: cube([(0, 0)]), "at least one customer"),
    (lambda: cube(NODES, PROFITS), "together"),
    (lambda: cube(NODES, [1], [1]), "for 5 customers"),
    (lambda: cube(NODES, [1, 2, 3, 4, -5], WEIGHTS), "not negative"),
    (lambda: cube(NODES, endurance=-1), "endurance"),
    (lambda: cube(NODES, PROFITS, WEIGHTS, capacity=0), "capacity"),
    (lambda: cube(NODES, seconds=0), "seconds"),
    (lambda: cube(NODES, dist=[[0]]), "dist"),
])
def test_bad_input_is_named(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_the_neural_solver_says_what_it_needs():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="aniate\\[neural\\]"):
            ncube(NODES, PROFITS, WEIGHTS, drone=True)


@pytest.mark.parametrize("args, options", [
    ((NODES,), {}),
    ((NODES,), {"drone": True}),
    ((NODES, PROFITS, WEIGHTS), {}),
    ((NODES, PROFITS, WEIGHTS), {"drone": True}),
])
def test_the_simulator_replays_a_plan_to_its_objective(args, options):
    plan = cube(*args, **options, **QUICK)
    n = len(NODES) - 1
    z = [int(i in plan.items) for i in range(n + 2)]
    flights = [(a, b, n + 1 if c == 0 else c) for a, b, c in plan.flights]
    env = cube.env(*args, **options)
    env.reset()
    total, done = 0.0, False
    for action in actions_of(n, plan.route[1:-1], flights, z):
        _, reward, done, _, _ = env.step(action)
        total += reward
    assert done
    assert total == pytest.approx(plan.objective)


def test_the_simulator_refuses_an_illegal_step():
    env = cube.env(NODES)
    env.reset()
    with pytest.raises(ValueError, match="launch"):
        env.step({"k": 2, "j": 1})


def test_the_exact_model_says_what_it_needs():
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="aniate\\[milp\\]"):
            cube.milp(NODES, PROFITS, WEIGHTS, drone=True)


def test_the_exact_model_proves_what_the_search_finds():
    pytest.importorskip("gurobipy")
    plan = cube.milp(NODES, PROFITS, WEIGHTS, drone=True, seconds=60)
    assert plan.solver == "milp"
    assert plan.gap == pytest.approx(0, abs=0.01)
    assert plan.objective == pytest.approx(cube(NODES, PROFITS, WEIGHTS, drone=True, seconds=1).objective, abs=1e-3)


def test_mdp_and_nmdp_are_called_directly():
    model = mdp(states=range(3), actions=["go"], transition=lambda s, a: {min(s + 1, 2): 1.0},
                reward=lambda s, a, t: 1.0 if t == 2 else 0.0, gamma=0.9, terminal=[2], initial=0)
    assert model.solve().start_value == pytest.approx(0.9)
    world = problems.gridworld(rows=5, cols=5, goals=((0, 4),), start=(4, 0))
    labels = nmdp.Labels(world).at("r4c4", "A").at("r0c4", "B")
    assert nmdp(world, labels, nmdp.Ordering(["A", "B"], violation_penalty=-1)).solve() is not None
