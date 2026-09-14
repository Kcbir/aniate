"""Notebook display: every result object renders as an HTML table."""

from __future__ import annotations

import re

import aniate as an
from aniate import nmdp, problems


def balanced(markup):
    for tag in ("div", "table", "tr", "td", "th", "span", "code"):
        assert len(re.findall(rf"<{tag}[\s>]", markup)) == markup.count(f"</{tag}>"), tag
    return markup


def test_model_solution_episodes_and_check_show_their_numbers():
    m = problems.gridworld(gamma=0.9)
    sol = m.solve()
    runs = an.simulate(m, sol, episodes=500, seed=0)
    report = an.check(m, policy=sol, episodes=100, seed=0)

    html = balanced(m._repr_html_())
    assert f"{m.S} states" in html and "valid" in html and "r0c0" in html and "transitions not shown" in html
    html = balanced(sol._repr_html_())
    assert sol.solver in html and f"{sol.start_value:.6g}" in html and "r3c0" in html
    html = balanced(runs._repr_html_())
    assert "500 episodes" in html and f"{runs.mean_return:.6g}" in html
    html = balanced(report._repr_html_())
    assert "passed" in html and "reachable state-action pairs" in html
    assert "No issues were found." in balanced(m.report._repr_html_())


def test_labels_are_escaped_and_long_tables_are_cut():
    m = an.from_functions(states=["<b>", "done"], actions=["go"], transition=lambda s, a: {"done": 1.0},
                          reward=lambda s, a, s2: 1.0 if s == "<b>" else 0.0, gamma=0.9,
                          terminal=["done"], initial="<b>")
    html = balanced(m._repr_html_())
    assert "&lt;b&gt;" in html and "<b>" not in html

    big = problems.gridworld(rows=8, cols=8)
    assert "transitions not shown" in big._repr_html_()
    assert "states not shown" in big.solve()._repr_html_()


def test_a_failed_check_lists_its_issues():
    model = problems.chain()
    wrong = an.Env(problems.chain(slip=0.5), seed=1, log=False)
    report = an.check(model, env=wrong, episodes=100, max_steps=100)
    assert not report.ok
    html = balanced(report._repr_html_())
    assert "failed with" in html and "transition_frequency" in html


def test_non_markovian_objects_show_their_memory():
    world = problems.gridworld(rows=5, cols=5, goals=((0, 4),), start=(4, 0))
    labels = nmdp.Labels(world).at("r4c4", "A").at("r0c4", "B")
    task = an.NMDP(world, labels, nmdp.Ordering(["A", "B"], violation_penalty=-1))
    policy = task.solve()
    for obj in (task, task.machine, policy):
        html = balanced(obj._repr_html_())
        assert "before_A" in html, type(obj).__name__
    assert "product states" in task._repr_html_()
    assert "the action depends on memory" in policy._repr_html_()
