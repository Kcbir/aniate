# tree.md

This guide shows which part of aniate to use for a given task, how the package is organised, and where new functionality belongs.

## 0. Verifying the installation

```python
from aniate import ant
ant()                 # or, from a shell: python -m aniate
```

`ant()` lists Python, the required dependencies (numpy, scipy) and the optional ones (matplotlib, pytest, hypothesis) with their installed versions. It then builds, solves, simulates and checks a five-state model, and writes a PDF plot when matplotlib is present. The last line is either `aniate is working.` or a statement that a component failed, and the function returns `True` or `False` accordingly.

| report status | meaning | action |
|---|---|---|
| `ok` | installed at a supported version | none |
| `missing` | a required dependency is absent | `pip install aniate` |
| `too old` | installed below the minimum version | upgrade the named package |
| `not installed` | an optional dependency is absent | `pip install 'aniate[vis]'` for plots, `pip install 'aniate[test]'` for tests |

## 1. Decision tree

```
What is known about the problem?
│
├── The transition probabilities and rewards are known
│   │
│   ├── How is the model written?
│   │   ├── as Python functions s, a -> {s2: p} ............ an.from_functions
│   │   ├── as a list of individual transitions ............ an.Builder
│   │   └── as arrays or sparse matrices ................... an.MDP(P, R, gamma)
│   │
│   ├── Does the reward or a constraint depend on history?
│   │   ├── no .............................................. use the MDP directly
│   │   └── yes ............................................. nmdp.Labels + an automaton + an.NMDP
│   │       ├── "A before B before C" ....................... nmdp.Ordering
│   │       ├── "at most k uses of a resource" .............. nmdp.Budget
│   │       ├── "finish within T steps" ..................... nmdp.Deadline
│   │       ├── any other finite-memory rule ................ nmdp.Mealy
│   │       └── several rules at once ....................... pass a list, or nmdp.compose
│   │
│   └── What is needed?
│       ├── the optimal policy .............................. m.solve()
│       ├── the exact value of a given policy ............... an.evaluate(m, policy)
│       ├── values at each time step ........................ an.mdp.values_by_time(m, policy, T)
│       ├── sampled episodes and their statistics ........... an.simulate(m, policy, episodes)
│       ├── one step at a time, Gym style ................... m.env()
│       └── a picture ....................................... aniate.vis (section 5)
│
└── Only a simulator exists (a program, a Gym environment, a physical system)
    └── Write down the model believed to describe it, then call
        an.check(model, env=simulator, policy=...)
        A failed check names the state-action pairs whose behaviour disagrees with the model.
```

## 2. Defining a model

| situation | call | notes |
|---|---|---|
| Transitions are easiest to state as a function | `an.from_functions(states, actions, transition, reward, gamma, terminal=, initial=)` | `reward` may take `(s, a)` or `(s, a, s2)`; `terminal` may be a list or a predicate |
| Transitions come from data or a table | `an.Builder(gamma).transition(s, a, s2, prob=, reward=)` | finish with `.build(missing="error")` or `.build(missing="stay")` |
| Matrices already exist | `an.MDP(P, R, gamma, states=, actions=)` | `P[a][s, s2]`; `R` may be `(S, A)`, `(S, A, S)` or per-action matrices |
| A standard test problem is sufficient | `an.problems.gridworld()`, `chain()`, `river_swim()`, `inventory()` | |
| A fixed number of steps | pass `horizon=T` | solved by backward induction |

States and actions may be any hashable labels, for example `"ok"`, `3` or `(2, 1)`. Every public function accepts labels.

## 3. Solving

| model | method chosen by `solve()` | what is returned |
|---|---|---|
| `horizon` set | `backward_induction` | time-indexed policy |
| `gamma < 1` | `policy_iteration` | stationary policy |
| `gamma = 1` with terminal states | `value_iteration`, then exact policy evaluation | stationary policy |

A method can be forced with `m.solve(method="value_iteration")`. `Solution.value` is always the exact value of `Solution.policy`, and `Solution.bound` limits the loss of any state relative to the optimum.

## 4. Running and checking

| goal | call |
|---|---|
| many episodes, fast | `an.simulate(m, policy, episodes=10_000, seed=0)` |
| mean return with its standard error | `runs.mean_return`, `runs.stderr` |
| exact value for comparison | `runs.model_value` |
| does the simulation agree with the model's values? | `runs.martingale()`: its column means should stay at `model_value` |
| fraction of time spent in each state | `runs.occupancy()` |
| a single episode | `runs[i]`, or `m.env()` with `reset` and `step` |
| test an external simulator | `an.check(m, env=sim, policy=pi)`, then `report.ok` and `report.issues` |
| an observation that is not a model label | `an.check(..., state=lambda obs: label)` |

Accepted policy forms: `None` (uniform random), a `Solution`, an array of action indices `(S,)` or `(T, S)`, a probability array `(S, A)`, a dict `{state: action}`, a function `f(state) -> action`, a `MemoryPolicy`, or any object with a `step(state)` method. The last form can be simulated but not evaluated exactly.

## 5. Plotting

All plots require `pip install 'aniate[vis]'`. Each accepts `ax=` and returns the Axes. `vis.save(figure_or_ax, "name.pdf")` writes the file.

| question | plot |
|---|---|
| What does a small model look like? (up to 30 states) | `vis.graph(m, policy=None)` |
| What does an automaton look like? | `vis.automaton(machine)` |
| What does a gridworld policy or value look like? | `vis.grid(m, solution)` |
| How are returns distributed? | `vis.returns(runs)` |
| How does the return accumulate over time? | `vis.paths(runs)` |
| Does the model's value agree with the simulation? | `vis.martingale(runs)` |
| How many episodes are enough? | `vis.convergence(runs)` |
| All four episode plots at once | `vis.overview(runs)` |

## 6. Reading errors and check results

| code | meaning | usual correction |
|---|---|---|
| `row_sum` | probabilities for a state-action pair do not sum to 1 | fix the transition function for the named pair |
| `dead_end` | a state-action pair has no successor | add a transition, or mark the state terminal |
| `terminal_not_absorbing` | a terminal state moves or pays reward | remove its outgoing transitions and rewards |
| `unbounded_value` | `gamma = 1` and no terminal state is reachable | add a terminal state or set `gamma < 1` |
| `impossible_transition` | the simulator reached a successor with model probability 0 | the model or the simulator is missing a transition |
| `transition_frequency` | observed frequencies differ significantly from the model | a probability in the model or the simulator is wrong |
| `reward_mismatch` | a reward differs from the model | compare the reward functions |
| `return_mismatch` | the mean return lies outside the confidence band | usually a consequence of one of the codes above |

## 7. Package layout

```
aniate/
├── __init__.py        public names: ant, MDP, NMDP, Builder, from_functions, Env, simulate, check, solve, evaluate
├── selfcheck.py       ant: installation report and self-test
├── __main__.py        python -m aniate
├── art.py             import banner
├── log.py             log lines, verbosity, capture
├── mdp/
│   ├── space.py       label <-> index mapping
│   ├── core.py        MDP: storage, lookups, describe
│   ├── builder.py     Builder, from_functions
│   ├── validate.py    construction-time checks and their codes
│   ├── solve.py       evaluate, policy iteration, value iteration, backward induction
│   ├── solution.py    Solution
│   ├── policies.py    conversion of every accepted policy form
│   ├── env.py         Env, simulate, Episodes, martingale
│   ├── check.py       check, CheckReport
│   └── text.py        text summaries
├── nmdp/
│   ├── labels.py      events attached to states and actions
│   ├── machine.py     Mealy automata, compose
│   ├── builders.py    Ordering, Budget, Deadline
│   ├── product.py     product of a world model and an automaton
│   ├── policy.py      MemoryPolicy
│   └── task.py        NMDP
├── problems/          gridworld, chain, river_swim, inventory
└── vis/
    ├── style.py       fonts, colours, save
    ├── stats.py       returns, paths, martingale, convergence, overview
    ├── diagrams.py    graph, automaton
    └── grid.py        grid
tests/
├── test_*.py          unit and property tests for each module
├── stsp/              worked example: stochastic travelling salesman
├── gaussian/          worked example: Gaussian random walk
└── heavy_tail/        worked example: heavy-tailed durations and the value martingale
```

## 8. Extending aniate

The public interface is the set of names exported by `aniate`, `aniate.nmdp` and `aniate.vis`. Other modules are internal.

| addition | location | requirements |
|---|---|---|
| a new solver | `aniate/mdp/solve.py`, registered in `SOLVERS` | return a `Solution` whose `value` comes from exact evaluation of its policy |
| a new automaton pattern | `aniate/nmdp/builders.py` | build and return a `Mealy`; give memory states readable names |
| a new standard problem | `aniate/problems/` | return an `MDP` that passes validation |
| a new plot | `aniate/vis/` | accept `ax=`, return the Axes, use the helpers in `style.py`, never call `show()` |
| a new check | `aniate/mdp/check.py` | report an `Issue` with a short code and the states involved |
| a new worked example | `tests/<name>/` | a model module, a `test_<name>.py`, and optionally a `plot.py` |

Conventions that every addition follows:

1. Public functions accept state and action labels as well as integer positions.
2. Every operation writes one line through `aniate.log`.
3. Every result object has a `describe()` method that returns a JSON-compatible dict.
4. A new feature ships with a test that fails when the feature is broken.

## 9. Scope

aniate works with finite models whose states can be listed in memory. Transition matrices are stored in sparse form, and the worked examples include a model with 1,011 states. Continuous state spaces must be discretised before use. Training agents with function approximation is outside the scope of the current version.
