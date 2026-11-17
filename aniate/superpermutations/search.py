"""The search over orderings: a route, the exact chains working on it, and kicks when it stalls."""
from __future__ import annotations

import time

import numpy as np

from . import engine
from .orderings import Orderings
from .words import check, route_from_word, word_from_route

# how often a neighbour reached in this many letters is tried when a chain grows
NEIGHBOUR_WEIGHT = {1: 24, 2: 12, 3: 4, 4: 1}


class Search:
    """One route over every ordering of n symbols. The word has n + cost letters."""

    def __init__(self, n: int, seed: int = 0, tables: Orderings | None = None):
        self.tables = tables or Orderings(n)
        self.n, self.N = self.tables.n, self.tables.N
        self.rng = np.random.default_rng(seed)
        engine.seed_random(seed)
        N, M, Q = self.N, engine.MMAX, engine.QMAX
        self.route = np.empty(N, dtype=np.int32)
        self.pos = np.empty(N, dtype=np.int32)
        self.wout = np.zeros(N, dtype=np.int32)
        self.cost = 0
        self.colcum = np.cumsum([NEIGHBOUR_WEIGHT[w] for w in self.tables.out_w]).astype(np.int64)
        self._f = np.empty((1 << M, Q), dtype=np.int32)
        self._par = np.empty((1 << M, Q), dtype=np.int16)
        self._C = np.zeros((Q, Q), dtype=np.int32)
        self._head, self._tail, self._item = (np.empty(Q, dtype=np.int32) for _ in range(3))
        self._base, self._order, self._turn, self._opt = (np.empty(M + 2, dtype=np.int32) for _ in range(4))
        self._cuts = np.empty(M + 4, dtype=np.int32)
        self._buf = np.empty(N, dtype=np.int32)
        self.stats = np.zeros(5, dtype=np.int64)   # chains solved, improving, level, letters saved, worsening

    def set_route(self, route) -> None:
        route = np.asarray(route, dtype=np.int32)
        if len(route) != self.N or np.bincount(route, minlength=self.N).max() != 1:
            raise ValueError("a route must hold every ordering once")
        self.route[:] = route
        t = self.tables
        self.cost = int(engine.load_route(t.perms, t.inv, self.route, self.pos, self.wout))

    def start_random(self) -> None:
        self.set_route(self.rng.permutation(self.N))

    def start_word(self, word) -> None:
        self.set_route(route_from_word(word, self.tables)[0])

    @property
    def length(self) -> int:
        return self.n + self.cost

    def run(self, iters: int, kmin: int = 2, kmax: int = 8, temp: float = 0.0, p_tie: float = 1.0,
            p_ring: float = 0.0, p_chain: float = 0.8, draws: int = 2) -> int:
        """`iters` chains of kmin to kmax cuts. Returns the word length afterwards."""
        t = self.tables
        self.cost = int(engine.descend(
            iters, kmin, min(kmax, engine.MMAX + 1), temp, p_tie, p_ring, p_chain, draws, self.cost, self.route,
            self.pos, self.wout, t.perms, t.inv, t.out_nbr, t.in_nbr, self.colcum, self._f, self._par, self._C,
            self._head, self._tail, self._item, self._base, self._order, self._turn, self._opt, self._cuts,
            self._buf, self.stats))
        return self.length

    def kick(self, count: int = 1, span: int | None = None) -> int:
        t = self.tables
        self.cost = int(engine.kick(count, span or 2 * self.n, self.cost, self.route, self.pos, self.wout, t.perms,
                                    t.inv, self._cuts, self._order, self._turn, self._buf))
        return self.length

    def solve(self, seconds: float, chunk: int = 500, hot: float = 0.0, hot_chunks: int = 20, cold_chunks: int = 40,
              back: float = 0.5, kicks: int = 2, target: int | None = None, on_better=None) -> int:
        """Run chains for `seconds` in cycles and end on the best route found.

        A cycle is `hot_chunks` chunks in which the temperature falls from `hot` to zero, then `cold_chunks`
        chunks at zero. After a cycle the search returns to the best route with probability `back`. With
        hot = 0 a cycle that found nothing ends with `kicks` kicks. Stops once the word is no longer than
        `target`. `on_better(search)` is called whenever the best length falls."""
        best, best_route = self.cost, self.route.copy()
        started = time.perf_counter()

        def note():
            nonlocal best
            if self.cost < best:
                best = self.cost
                best_route[:] = self.route
                if on_better:
                    on_better(self)
                return True
            return False

        def done():
            return time.perf_counter() - started >= seconds or (target is not None and self.n + best <= target)

        while not done():
            improved = False
            for i in range(hot_chunks if hot > 0 else 0):
                self.run(chunk, temp=hot * (1 - i / hot_chunks))
                improved |= note()
                if done():
                    break
            for _ in range(cold_chunks):
                if done():
                    break
                self.run(chunk)
                improved |= note()
            if done():
                break
            if self.cost > best and self.rng.random() < back:
                self.set_route(best_route)
            if hot <= 0 and kicks and not improved:
                self.kick(kicks)
        self.set_route(best_route)
        return self.length

    def word(self) -> str:
        return word_from_route(self.tables, self.route)

    def verify(self) -> str:
        """Recount the route from scratch and scan the word it gives. Returns the word; raises on a mismatch."""
        t = self.tables
        pos, wout = np.empty(self.N, dtype=np.int32), np.zeros(self.N, dtype=np.int32)
        cost = int(engine.load_route(t.perms, t.inv, self.route.copy(), pos, wout))
        if cost != self.cost:
            raise AssertionError(f"the search's count is wrong: tracked {self.cost}, recounted {cost}")
        word = self.word()
        report = check(word)
        if not report.complete or report.length != self.length:
            raise AssertionError(f"the word does not check: {report}, expected {self.length} letters")
        return word
