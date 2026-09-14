"""Plots: black Computer Modern text, nothing shown, PDF only."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import aniate as an  # noqa: E402
from aniate import nmdp, problems, vis  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ax():
    return plt.subplots()[1]


@pytest.fixture(scope="module")
def runs():
    m = problems.gridworld(gamma=0.9)
    return an.simulate(m, m.solve(), episodes=2000, seed=0)


def texts(fig):
    fig.canvas.draw()                     # tick labels only exist after a draw
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def test_all_text_is_plain_black_computer_modern(runs):
    figures = [vis.overview(runs), vis.graph(problems.chain()).figure,
               vis.automaton(nmdp.Ordering(["A", "B"])).figure,
               vis.grid(problems.gridworld(), problems.gridworld().solve()).figure]
    for fig in figures:
        found = texts(fig)
        assert found
        for t in found:
            assert to_rgba(t.get_color()) == (0.0, 0.0, 0.0, 1.0), t.get_text()
            assert t.get_fontweight() in ("normal", 400) and t.get_fontstyle() == "normal", t.get_text()
            assert t.get_fontfamily() == ["cmr10"], (t.get_text(), t.get_fontfamily())


def test_diagrams(ax):
    m = problems.river_swim()
    assert vis.graph(m, ax=ax) is ax
    with_policy = vis.graph(m, m.solve(), ax=plt.subplots()[1])
    assert len(with_policy.patches) < len(ax.patches)
    assert vis.automaton(nmdp.compose([nmdp.Ordering(["A", "B"]), nmdp.Budget("r", max=1)])).get_title() == ""
    # cmr10 has no underscore glyph: labels containing one go through mathtext, escaped
    node_labels = [t.get_text() for t in vis.automaton(nmdp.Ordering(["A", "B"])).texts]
    assert r"$\mathrm{before\_A}$" in node_labels and "violated" in node_labels
    with pytest.raises(ValueError, match="limit"):
        vis.graph(problems.gridworld(rows=8, cols=8))


def test_grid(ax, runs):
    m = runs.model
    assert vis.grid(m, m.solve(), episode=runs[0], ax=ax) is ax
    vis.grid(m, values=runs.occupancy())
    task = nmdp.NMDP(m, nmdp.Labels(m).at("r3c3", "A").at("r0c3", "B"), nmdp.Ordering(["A", "B"]))
    slice_ax = vis.grid(task, task.solve(), memory="A_visited_B_pending")
    assert slice_ax.get_xlabel() == r"$\mathrm{memory:\ A\_visited\_B\_pending}$"
    with pytest.raises(ValueError, match="layout"):
        vis.grid(problems.chain())


@pytest.mark.parametrize("plot", [vis.returns, vis.paths, vis.martingale, vis.convergence])
def test_episode_plots(runs, plot, ax):
    assert plot(runs, ax=ax) is ax
    ax.figure.canvas.draw()
    assert ax.get_title() == ""


def test_overview_and_its_fallback(runs):
    assert len(vis.overview(runs).axes) == 4

    class Stepper:
        def step(self, s):
            return 0

    fig = vis.overview(an.simulate(problems.chain(), Stepper(), episodes=20))
    assert any("evaluated exactly" in t.get_text() for t in texts(fig))
    with pytest.raises(TypeError, match="simulate"):
        vis.returns(np.zeros(10))


def test_save_writes_pdf_with_embedded_computer_modern(tmp_path, runs):
    path = vis.save(vis.overview(runs), tmp_path / "overview")
    assert path.suffix == ".pdf"
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and b"CMR10" in data.upper()
    with pytest.raises(ValueError, match="PDF"):
        vis.save(vis.returns(runs), tmp_path / "overview.png")


def test_nothing_is_shown(monkeypatch, runs):
    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))
    vis.overview(runs)
    vis.graph(problems.chain())
    assert not called
