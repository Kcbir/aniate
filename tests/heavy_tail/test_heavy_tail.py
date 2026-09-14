"""Heavy-tailed robot: exact value, fat tails, a flat martingale, a refused leak, a caught simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest
from warehouse import RealRobot, closed_form, warehouse_robot

import aniate as an


def assert_flat(M, level, z=5.0):
    """Every column mean of M sits at ``level`` within ``z`` standard errors (exactly where se = 0)."""
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / math.sqrt(M.shape[0])
    live = se > 1e-9
    assert np.all(np.abs(mean[~live] - level) < 1e-9)
    assert np.all(np.abs(mean[live] - level) < z * se[live])


def test_matches_closed_form_and_its_martingale_is_flat():
    m, _, _ = warehouse_robot()
    exact = closed_form()
    assert m.initial @ m.evaluate(None) == pytest.approx(exact, rel=1e-10)

    runs = an.simulate(m, None, episodes=20_000, seed=2)
    x = runs.returns
    z = (x - x.mean()) / x.std()
    assert (z ** 4).mean() - 3 > 10            # excess kurtosis ~25: nothing like a Gaussian
    assert np.median(x) > x.mean()             # most runs are clean; rare long jams drag the mean
    assert_flat(runs.martingale(), exact)


def test_leaky_jam_distribution_is_refused_and_a_wrong_robot_is_caught():
    with pytest.raises(ValueError, match=r"row_sum[\s\S]*\(0, 0\) / drive"):
        warehouse_robot(leak=0.03)             # 3% of probability mass silently lost

    m, _, _ = warehouse_robot()
    report = an.check(m, env=RealRobot(p=0.10), episodes=2000)   # jams twice as often
    assert "transition_frequency" in {i.code for i in report.issues}
