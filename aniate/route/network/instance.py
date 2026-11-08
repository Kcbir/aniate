"""The problem as the network sees it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TTPDInstance:
    n: int
    coords: np.ndarray
    profits: np.ndarray
    weights: np.ndarray
    dist: np.ndarray
    W: float
    v_max: float
    v_min: float
    v_D: float
    R: float
    node_ids: list
    ED: float | None = None   # longest flight; None is unlimited

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return self.n + 1

    @property
    def d_max(self) -> float:
        return float(self.dist.max())


def tour_time_ub(d_max: float, n: int) -> float:
    return d_max * (n + 1)
