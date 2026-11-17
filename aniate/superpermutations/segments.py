"""The order of the segments of a Detour word, and the search that improves it."""

from __future__ import annotations

import time

import numpy as np

from ._jit import jit

KMAX = 13


@jit
def W(E, S, i, j, n):
    """Letters added going from the end of segment i to the start of segment j (0 into segment 0)."""
    if j == 0:
        return 0
    e = E[i]
    s = S[j]
    for o in range(n - 1, 0, -1):
        if (e & ((np.int64(1) << (4 * o)) - 1)) == (s >> (4 * (n - o))):
            return n - o
    return n


@jit
def path_cost(E, S, order, n):
    t = 0
    for p in range(len(order) - 1):
        t += W(E, S, order[p], order[p + 1], n)
    return t


def arcs(E, S, n, wmax):
    """(end segment, start segment, weight) for every pair that joins in at most wmax letters."""
    out = []
    for o in range(n - 1, n - wmax - 1, -1):
        pre = S >> (4 * (n - o))
        suf = E & ((1 << (4 * o)) - 1)
        order = np.argsort(pre, kind="stable")
        ps = pre[order]
        lo = np.searchsorted(ps, suf, "left")
        hi = np.searchsorted(ps, suf, "right")
        cnt = hi - lo
        src = np.repeat(np.arange(len(E)), cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        dst = order[np.repeat(lo, cnt) + off]
        keep = (src != dst) & (dst != 0)
        out.append((src[keep], dst[keep], np.full(int(keep.sum()), n - o, dtype=np.int64)))
    a = np.concatenate([x[0] for x in out])
    b = np.concatenate([x[1] for x in out])
    w = np.concatenate([x[2] for x in out])
    key = a * len(E) + b
    order = np.lexsort((w, key))
    key = key[order]
    firsts = np.concatenate(([True], key[1:] != key[:-1]))
    idx = order[firsts]
    return a[idx], b[idx], w[idx]


def csr(key, other, w, N):
    order = np.lexsort((w, key))
    ptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(ptr, key + 1, 1)
    return np.cumsum(ptr), other[order].copy(), w[order].copy()


@jit
def _grow(order, pos, sp, si, sw, pp, pi, pw, p0, K, wl, cuts, fe, fs):
    N = len(order)
    nc = 1
    cuts[0] = p0
    ne = 1
    fe[0] = order[p0]
    ns = 0
    if p0 + 1 < N:
        fs[0] = order[p0 + 1]
        ns = 1
    tries = 0
    while nc < K and tries < 6 * K:
        tries += 1
        newcut = -1
        if ns == 0 or np.random.random() < 0.5:
            e = fe[np.random.randint(ne)]
            lo = sp[e]
            hi = sp[e + 1]
            m = 0
            while lo + m < hi and sw[lo + m] <= wl:
                m += 1
            if m == 0:
                continue
            y1 = si[lo + np.random.randint(m)]
            newcut = pos[y1] - 1
        else:
            s = fs[np.random.randint(ns)]
            if s == 0:
                continue
            lo = pp[s]
            hi = pp[s + 1]
            m = 0
            while lo + m < hi and pw[lo + m] <= wl:
                m += 1
            if m == 0:
                continue
            newcut = pos[pi[lo + np.random.randint(m)]]
        dup = False
        for t in range(nc):
            if cuts[t] == newcut:
                dup = True
        if dup or newcut < 0:
            continue
        cuts[nc] = newcut
        nc += 1
        fe[ne] = order[newcut]
        ne += 1
        if newcut + 1 < N:
            fs[ns] = order[newcut + 1]
            ns += 1
    return nc


@jit
def _chain(E, S, order, pos, n, sp, si, sw, pp, pi, pw, p0, K, wl, cuts, fe, fs, dp, par, C, buf, level):
    """One chain from position p0. Returns (letters saved, 1 if the order changed)."""
    N = len(order)
    nc = _grow(order, pos, sp, si, sw, pp, pi, pw, p0, K, wl, cuts, fe, fs)
    if nc < 3:
        return 0, 0
    c = np.sort(cuts[:nc])
    m = nc - 1
    tail = c[nc - 1] < N - 1
    old = 0
    for t in range(nc):
        if c[t] + 1 < N:
            old += W(E, S, order[c[t]], order[c[t] + 1], n)
    SC = 64
    for i in range(m + 1):
        ei = order[c[i]]
        for j in range(1, m + 2):
            if j <= m:
                sj = order[c[j - 1] + 1]
                natural = j == i + 1
                C[i, j] = (W(E, S, ei, sj, n) * SC + (0 if natural else np.random.randint(4))) if i != j else 1 << 40
            else:
                C[i, j] = (W(E, S, ei, order[c[m] + 1], n) * SC) if tail else 0
    full = (1 << m) - 1
    for mask in range(1, full + 1):
        for last in range(m):
            dp[mask, last] = 1 << 50
    for j in range(m):
        dp[1 << j, j] = C[0, j + 1]
        par[1 << j, j] = -1
    for mask in range(1, full + 1):
        for last in range(m):
            if not (mask >> last) & 1:
                continue
            cur = dp[mask, last]
            if cur >= (1 << 50):
                continue
            for j in range(m):
                if (mask >> j) & 1:
                    continue
                v = cur + C[last + 1, j + 1]
                nm = mask | (1 << j)
                if v < dp[nm, j]:
                    dp[nm, j] = v
                    par[nm, j] = last
    best = 1 << 50
    bl = -1
    for last in range(m):
        v = dp[full, last] + C[last + 1, m + 1]
        if v < best:
            best = v
            bl = last
    new = best // SC
    if new > old:
        return 0, 0
    seq = np.empty(m, dtype=np.int64)
    mask = full
    last = bl
    for t in range(m - 1, -1, -1):
        seq[t] = last
        pl = par[mask, last]
        mask ^= 1 << last
        last = pl
    same = True
    for t in range(m):
        if seq[t] != t:
            same = False
    if same:
        return 0, 0
    if new == old and not level:
        return 0, 0
    k = 0
    for t in range(m):
        s = seq[t]
        for u in range(c[s] + 1, c[s + 1] + 1):
            buf[k] = order[u]
            k += 1
    base = c[0] + 1
    for u in range(k):
        order[base + u] = buf[u]
        pos[buf[u]] = base + u
    return old - new, 1


@jit
def campaign(E, S, order, pos, n, sp, si, sw, pp, pi, pw, K, wmin, level, seed):
    """One pass: a chain from every join of weight wmin or more, in random order. Returns (saved, kept, tried)."""
    np.random.seed(seed)
    N = len(order)
    cuts = np.zeros(KMAX + 1, dtype=np.int64)
    fe = np.zeros(KMAX + 2, dtype=np.int64)
    fs = np.zeros(KMAX + 2, dtype=np.int64)
    dp = np.zeros((1 << KMAX, KMAX), dtype=np.int64)
    par = np.zeros((1 << KMAX, KMAX), dtype=np.int64)
    C = np.zeros((KMAX + 2, KMAX + 2), dtype=np.int64)
    buf = np.empty(N, dtype=np.int64)
    saved = 0
    kept = 0
    tried = 0
    heavy = np.empty(N, dtype=np.int64)
    nh = 0
    for p in range(N - 1):
        if W(E, S, order[p], order[p + 1], n) >= wmin:
            heavy[nh] = order[p]
            nh += 1
    for t in range(nh):
        x = heavy[np.random.randint(nh)]
        p0 = pos[x]
        if p0 >= N - 1:
            continue
        w0 = W(E, S, x, order[p0 + 1], n)
        if w0 < wmin:
            continue
        kk = 4 + np.random.randint(K - 3)
        wl = 4 + np.random.randint(3) if w0 > 5 else 4 + np.random.randint(2)
        g, kpt = _chain(E, S, order, pos, n, sp, si, sw, pp, pi, pw, p0, kk, wl, cuts, fe, fs, dp, par, C, buf, level)
        saved += g
        kept += kpt
        tried += 1
    return saved, kept, tried


class SegmentSearch:
    """The order of the segments of a Detour word, and the chains that improve it."""

    def __init__(self, segments, wmax: int = 8, K: int = 10):
        self.segments = segments
        sg = segments
        self.n, self.N = sg.n, sg.nseg
        self.E, self.S = sg.stop, sg.start
        x, y, w = arcs(self.E, self.S, self.n, wmax)
        self.sp, self.si, self.sw = csr(x, y, w, self.N)
        self.pp, self.pi, self.pw = csr(y, x, w, self.N)
        self.K = min(K, KMAX)
        self.order = np.arange(self.N, dtype=np.int64)
        self.pos = np.arange(self.N, dtype=np.int64)
        self.start_cost = self.cost = int(path_cost(self.E, self.S, self.order, self.n))

    def run(self, seconds: float, seed: int = 0, wmin: int = 5, on_round=None) -> int:
        """Passes of chains for `seconds`. Returns the letters saved against the starting order."""
        started = time.perf_counter()
        r = 0
        while time.perf_counter() - started < seconds:
            campaign(self.E, self.S, self.order, self.pos, self.n, self.sp, self.si, self.sw, self.pp, self.pi,
                     self.pw, self.K, wmin, True, seed * 1_000_003 + r)
            r += 1
            self.cost = int(path_cost(self.E, self.S, self.order, self.n))
            if on_round:
                on_round(r, self.start_cost - self.cost)
        return self.start_cost - self.cost
