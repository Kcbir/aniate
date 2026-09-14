"""Gaussian plane: calm air is a fair coin; wind tilts the odds and the martingale stays flat."""

from __future__ import annotations

import math

import numpy as np
import pytest
from plane import gaussian_plane

import aniate as an


def assert_flat(M, level, z=5.0):
    """Every column mean of M sits at ``level`` within ``z`` standard errors (exactly where se = 0)."""
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1) / math.sqrt(M.shape[0])
    live = se > 1e-9
    assert np.all(np.abs(mean[~live] - level) < 1e-9)
    assert np.all(np.abs(mean[live] - level) < z * se[live])


def test_symmetric_gaussian_walk_is_a_fair_coin():
    plane = gaussian_plane()
    assert plane.initial @ plane.evaluate(None) == pytest.approx(0.5, abs=1e-9)
    runs = an.simulate(plane, None, episodes=20_000, seed=1)
    assert set(np.unique(runs.returns)) <= {0.0, 1.0}
    assert abs(runs.mean_return - 0.5) < 5 * runs.stderr


def test_wind_tilts_the_odds_and_the_martingale_stays_flat():
    windy = gaussian_plane(drift=0.3)
    p = windy.initial @ windy.evaluate(None)
    assert 0.9 < p < 1.0
    runs = an.simulate(windy, None, episodes=20_000, seed=1)
    assert_flat(runs.martingale(), p)
    assert an.check(windy, episodes=1000).ok
