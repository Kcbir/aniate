<div align="center">

# aniate

*Open source combinatorics solvers*

[![pypi](https://img.shields.io/pypi/v/aniate?label=pypi&logo=pypi&logoColor=white&color=3775A9)](https://pypi.org/project/aniate/)
[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![tests](https://img.shields.io/github/actions/workflow/status/Kcbir/aniate/test.yml?branch=main&label=tests&logo=github)](https://github.com/Kcbir/aniate/actions/workflows/test.yml)
[![website](https://img.shields.io/badge/website-aniate.com-5A7B55)](https://aniate.com)
[![guide](https://img.shields.io/badge/guide-tree.md-B03A2E)](https://github.com/Kcbir/aniate/blob/main/tree.md)
[![licence](https://img.shields.io/badge/licence-MIT-555555)](https://github.com/Kcbir/aniate/blob/main/LICENSE)

<br>

<img src="https://cdn.jsdelivr.net/gh/Kcbir/aniate@main/assets/fibonacci.svg" width="440" alt="Fibonacci squares 1, 1, 2, 3, 5, 8 and 13 tiling a 21 by 13 rectangle, with the golden spiral drawn through them">

</div>

---

```python
from aniate import cube, ncube, mdp, nmdp
```

| Solver | Solves | Call |
|---|---|---|
| **NCube** | the TTP-D with a trained network, the neural accelerator | `ncube(nodes, profits, weights, drone=True)` |
| **Cube** | the TTP-D with the metaheuristic | `cube(nodes, profits, weights, drone=True)` |
| **cube.milp** | the TTP-D with the exact model | `cube.milp(nodes, profits, weights, drone=True)` |
| **MDP** | decisions where only the present matters | `mdp(states, actions, transition, reward).solve()` |
| **NMDP** | decisions where the past matters | `nmdp(world, labels, rule).solve()` |

## Cube and NCube

Solvers for the Travelling Thief Problem with Drone, from [Drive, Pack, Fly: The Travelling Thief Problem with Drone](https://arxiv.org/abs/2608.16435), by Kabir Murjani and Prof. Abhay Sobhanan. They take the same inputs, and what you leave out decides the problem: with no items it is a TSP-D, with no drone a TTP, and with neither a TSP.

```python
from aniate import cube, ncube

nodes = [(0, 0), (4, 1), (5, 4), (1, 5), (2, 2), (3, 3)]   # depot first
profits, weights = [50, 60, 70, 80, 40], [2, 3, 4, 5, 2]

ncube(nodes, profits, weights, drone=True)                 # the neural accelerator, with our weights
net = ncube.train(30)                                      # or your own, on 30 customers
ncube(nodes, profits, weights, drone=True, network=net)

print(cube(nodes))                                         # TSP: a truck
print(cube(nodes, drone=True))                             # TSP-D: a truck and a drone
print(cube(nodes, profits, weights))                       # TTP: a truck and a knapsack
print(cube(nodes, profits, weights, drone=True))           # TTP-D: all three

print(cube.milp(nodes, profits, weights, drone=True))      # the exact model, for small problems

env = cube.env(nodes, profits, weights, drone=True)        # the simulator, one decision at a time
```

| input | default | meaning |
|---|---|---|
| `nodes` | required | `(x, y)` pairs, depot first |
| `profits`, `weights` | none | one value per customer, depot left out; give both or neither |
| `drone` | `False` | fly a drone from the truck |
| `endurance` | unlimited | longest flight, launch to customer to landing; setting it turns the drone on |
| `capacity` | half the total weight | what the truck can carry |
| `rent` | the rate at which a plain tour costs half of all the profit | cost per unit of time |
| `drone_speed` | twice the empty truck's speed | setting it turns the drone on |
| `dist` | straight lines | your own distance matrix, depot first |
| `seconds` | half a second per customer; one hour for `cube.milp` | time limit for `cube` and `cube.milp`; the paper gives `cube` 15 per customer |
| `seed` | `0` | `cube` only |
| `network` | `True`, our weights | `ncube` only; a path to your own, or what `ncube.train` returned |

The plan comes back with `route`, `flights`, `items`, `time`, `profit`, `objective`, `problem`, `seconds`, `solver` and, for `cube.milp`, `gap`. `ncube` needs `pip install 'aniate[neural]'` and downloads the closest trained size once from Hugging Face. `ncube.train(problems, examples=100, seconds=None, epochs=60, save=None)` solves each problem with the metaheuristic, then teaches the network to imitate those plans; `problems` is a number of customers, or a list of `(nodes, profits, weights)`. `cube.milp` needs `pip install 'aniate[milp]'` and a Gurobi licence for all but small problems.

## MDP and NMDP

The decision solvers work with finite problems: Markov decision processes in `mdp`, and problems whose reward depends on the history of the episode in `nmdp`.

A problem is specified by its states, actions, transition probabilities, rewards and discount factor. The specification is validated at construction, and a model is rejected when a probability row does not sum to one, when a terminal state continues to move or to accrue reward, or when its value is unbounded. The library computes optimal policies together with their exact values, simulates episodes in batches, and compares any external simulator with the model through per-transition chi-square tests and the value martingale `M_t = G_<t + γ^t V(s_t)`. History-dependent objectives are expressed with event labels and a Mealy automaton, and the exact product model is solved. Every operation writes one structured log line, and every figure is saved as a PDF with embedded Computer Modern fonts.

## Installation

```bash
pip install aniate            # numpy and scipy
pip install 'aniate[vis]'     # adds matplotlib, required for plots
pip install 'aniate[gym]'     # adds gymnasium, required for aniate.gym
pip install 'aniate[neural]'  # adds PyTorch, required for the neural accelerator
pip install 'aniate[milp]'    # adds Gurobi, required for the exact model
```

For Julia, `cube` is one call too:

```julia
using Pkg; Pkg.add(url="https://github.com/Kcbir/aniate", subdir="julia")

using Aniate
cube(nodes, profits, weights; drone=true)
```

## Verifying the installation

```python
from aniate import ant
ant()
```

`ant()` prints the Fibonacci drawing, the installed version of Python and of each dependency, and the result of a self-test that builds, solves, simulates and checks a small model. It returns `True` when aniate is working. The same report is printed by `python -m aniate`.

```
aniate 2.2.0   Open source combinatorics solvers

python      3.12.4      ok             required, 3.10 or later
numpy       2.1.0       ok             required, 1.24 or later
scipy       1.14.1      ok             required, 1.12 or later
matplotlib  -           not installed  optional, for plots in aniate.vis
pytest      -           not installed  optional, for running the tests
hypothesis  -           not installed  optional, for property-based tests
gymnasium   -           not installed  optional, for the Gymnasium adapter in aniate.gym

self-test   model valid | solved, value 3.20876 | 2,000 episodes simulated | check passed | 13 ms
            plots unavailable; install them with: pip install 'aniate[vis]'

aniate is working.
```

## Quick start

```python
from aniate import mdp, vis

m = mdp(
    states=range(5), actions=["left", "right"],
    transition=lambda s, a: {max(s - 1, 0): 1.0} if a == "left" else {min(s + 1, 4): 0.8, s: 0.2},
    reward=lambda s, a, s_next: 10.0 if s_next == 4 else -1.0,
    gamma=0.9, terminal=[4], initial=0,
)

sol = m.solve()                                   # optimal policy, its exact value, an error bound
runs = mdp.simulate(m, sol, episodes=10_000)       # 10,000 episodes, run in parallel
report = mdp.check(m, policy=sol)                  # compares simulated behaviour with the model
vis.save(vis.overview(runs), "overview.pdf")      # four standard plots in one PDF
m.describe()                                      # the model as a JSON-compatible dict
```

Each operation writes one line to standard error:

```
aniate | model    | 5 states | 2 actions | gamma 0.9 | horizon inf | 14 transitions | valid
aniate | solve    | policy_iteration | 4 iterations | 2.8 ms | start value 3.20876 | bound 8.9e-15
aniate | simulate | 10,000 episodes | 49,768 steps | 4 ms | mean return 3.23989 +/- 0.014 | 10,000 terminated | 0 truncated
aniate | check    | PASSED | 300 episodes | 1,505 steps | 4 pairs tested | return 3.1863 vs model 3.2088
aniate | vis      | saved overview.pdf
```

## Validation

Construction fails, and the error names each offending state and action, if any of the following hold:

| code | condition |
|---|---|
| `row_sum` | a row of transition probabilities does not sum to 1 |
| `dead_end` | a state-action pair has no successor |
| `terminal_not_absorbing` | a terminal state moves to another state or pays a reward |
| `unbounded_value` | the discount factor is 1 and no terminal state is reachable |

`mdp.check(model, env=simulator)` runs episodes through an external simulator and reports `impossible_transition`, `transition_frequency`, `reward_mismatch`, `termination_mismatch` and `return_mismatch`. These properties are verified by runtime checks and by the test suite; they are not formally proved.

## History-dependent rewards

```python
from aniate import nmdp, problems

world  = problems.gridworld(rows=5, cols=5, goals=((0, 4),), start=(4, 0))
labels = nmdp.Labels(world).at("r4c4", "A").at("r0c4", "B")
task   = nmdp(world, labels, nmdp.Ordering(["A", "B"], violation_penalty=-1))
policy = task.solve()        # a MemoryPolicy: policy.step(state) returns the next action
```

`nmdp.Ordering`, `nmdp.Budget` and `nmdp.Deadline` cover common patterns, and `nmdp.Mealy` expresses any finite-memory rule.

## Notebooks

In Jupyter, a model, solution, set of episodes, check report, automaton or non-Markovian task that ends a cell is displayed as a table. A long table is shortened, and the number of omitted rows is stated.

## Gymnasium

```python
from aniate import gym

env = gym.make(m)                               # a gymnasium.Env with Discrete spaces
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

gym.register("aniate/Corridor-v0", lambda: m)   # gymnasium.make("aniate/Corridor-v0") then builds it
```

Transitions are sampled from the model with the seeded generator of Gymnasium, so equal seeds give equal episodes, and the environment passes the Gymnasium environment checker. The adapter requires `pip install 'aniate[gym]'`.

## Interface

| call | purpose |
|---|---|
| `ant()` | report installed dependencies and run a self-test |
| `mdp(states, actions, transition, reward, gamma, terminal=, initial=)` | define a model with Python functions |
| `mdp.Builder(gamma)` | define a model one transition at a time |
| `mdp.MDP(P, R, gamma)` | define a model from matrices |
| `m.solve()` | optimal policy and exact value |
| `mdp.evaluate(m, policy)` | exact value of any policy |
| `mdp.simulate(m, policy, episodes)` | batched episodes, returned as `Episodes` |
| `m.env()` | a Gym-style environment with `reset` and `step` |
| `mdp.check(m, env=simulator, policy=)` | statistical comparison of a simulator with the model |
| `nmdp(world, labels, automaton)` | a problem whose reward depends on history |
| `gym.make(m)`, `gym.register(id, model)` | the model as a Gymnasium environment |
| `vis.overview(runs)`, `vis.graph(m)`, `vis.grid(m, sol)` | plots; `vis.save(fig, "name.pdf")` writes a PDF |
| `obj.describe()` | a JSON-compatible summary of any model, solution, report or set of episodes |
| `aniate.verbosity("quiet")` | set the log level |

A decision tree for choosing among these calls, the package layout and the conventions for extending the library are given in [tree.md](https://github.com/Kcbir/aniate/blob/main/tree.md).

## Worked examples

| folder | problem | result verified by the tests |
|---|---|---|
| [`tests/stsp`](https://github.com/Kcbir/aniate/tree/main/tests/stsp) | stochastic travelling salesman with randomly blocked roads | the exact solution equals the best of all 24 tours, with expected cost 17.6227 |
| [`tests/gaussian`](https://github.com/Kcbir/aniate/tree/main/tests/gaussian) | Gaussian random walk between two absorbing edges | the hitting probability is 0.5 by symmetry and 0.956 under drift; Monte Carlo estimates agree |
| [`tests/heavy_tail`](https://github.com/Kcbir/aniate/tree/main/tests/heavy_tail) | travel time with power-law delays | the value equals the closed form, the martingale mean is constant, and an incorrect simulator is detected |

```bash
pip install -e '.[dev]'
pytest                               # all tests
python tests/stsp/plot.py            # writes route.pdf and overview.pdf into tests/stsp
```

Importing aniate prints a Fibonacci rectangle to standard error once per process; `ANIATE_BANNER=0` disables it.

## Citation

```bibtex
@software{murjani2026aniate,
  author  = {Murjani, Kabir},
  title   = {aniate: open source combinatorics solvers},
  year    = {2026},
  version = {2.2.0},
  url     = {https://github.com/Kcbir/aniate}
}
```

## Licence

Released under the [MIT Licence](https://github.com/Kcbir/aniate/blob/main/LICENSE).
