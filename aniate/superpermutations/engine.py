"""The search over a route of orderings."""

import numpy as np

from ._jit import jit

MMAX = 15            # most pieces re-ordered by one chain
QMAX = 128           # most (piece, entry) choices in one chain
SCALE = 1 << 12      # letters are multiplied by this, so random tie-breaks stay below one letter
INF = 1 << 30
SKIP = -(1 << 20)    # returned by `solve` when a chain is too large to solve
BAN = 64 * SCALE     # charged for the step a chain is told to remove


@jit
def step(perms, inv, p, q):
    """Letters added when a word ending in ordering p is extended to end in q."""
    n = perms.shape[1]
    k = int(inv[p, perms[q, 0]])
    if k == 0:
        return n
    for i in range(1, n - k):
        if perms[p, k + i] != perms[q, i]:
            return n
    return k


@jit
def load_route(perms, inv, route, pos, wout):
    """Fill `pos` (where each ordering stands) and `wout` (letters of the step leaving it).
    Returns the letters of all steps."""
    N = route.shape[0]
    total = 0
    for i in range(N):
        pos[route[i]] = i
    for i in range(N - 1):
        w = step(perms, inv, route[i], route[i + 1])
        wout[route[i]] = w
        total += w
    wout[route[N - 1]] = 0
    return total


@jit
def seed_random(seed):
    np.random.seed(seed)


@jit
def _has(cuts, n, c):
    for i in range(n):
        if cuts[i] == c:
            return True
    return False


@jit
def _arc(route, wout, x, N):
    """First and last position of the stretch of one-letter steps through position x."""
    s = x
    while s > 0 and wout[route[s - 1]] == 1:
        s -= 1
    e = x
    while e < N - 1 and wout[route[e]] == 1:
        e += 1
    return s, e


@jit
def grow(anchor, kmax, route, pos, wout, out_nbr, in_nbr, colcum, n_sym, p_ring, p_chain, cuts):
    """Grow a set of cuts from `anchor`. Cut c separates positions c and c+1; -1 and N-1 stand for the
    two ends of the route.

    Each cut leaves a loose end (the stop before it) and a loose start (the stop after it). A new cut is
    placed where a loose end finds a successor it reaches in few letters, or a loose start a cheap
    predecessor. That frees another stop, and with probability `p_chain` the next cut continues from the
    stop just freed, so the cuts form a chain. Returns the number of cuts, sorted in `cuts`."""
    N = route.shape[0]
    ncol = out_nbr.shape[1]
    n = 1
    cuts[0] = anchor
    tail_cut = anchor
    head_cut = anchor
    tries = 0
    while n < kmax and tries < 6 * kmax:
        tries += 1
        r = np.random.randint(colcum[ncol - 1])
        j = 0
        while colcum[j] <= r:
            j += 1
        chain = np.random.random() < p_chain
        forward = np.random.randint(2) == 0
        if forward:
            ci = tail_cut if chain else cuts[np.random.randint(n)]
            if ci < 0:
                continue
            x = pos[out_nbr[route[ci], j]]
            if x == ci + 1:
                continue
            c = x - 1
        else:
            ci = head_cut if chain else cuts[np.random.randint(n)]
            if ci >= N - 1:
                continue
            x = pos[in_nbr[route[ci + 1], j]]
            if x == ci:
                continue
            c = x
        if _has(cuts, n, c):
            continue
        if 0 <= c < N - 1 and wout[route[c]] == 1 and np.random.random() < p_ring:
            s, e = _arc(route, wout, x, N)
            if e - s + 1 == n_sym:
                if not _has(cuts, n, s - 1) and n < kmax:
                    cuts[n] = s - 1
                    n += 1
                if not _has(cuts, n, e) and n < kmax:
                    cuts[n] = e
                    n += 1
                if forward:
                    tail_cut = s - 1
                else:
                    head_cut = e
                continue
        cuts[n] = c
        n += 1
        if forward:
            tail_cut = c
        else:
            head_cut = c
    for i in range(1, n):
        v = cuts[i]
        k = i - 1
        while k >= 0 and cuts[k] > v:
            cuts[k + 1] = cuts[k]
            k -= 1
        cuts[k + 1] = v
    return n


@jit
def solve(cuts, n, anchor, route, wout, perms, inv, f, par, C, head, tail, item, base, order, turn, opt):
    """Best order of the pieces between the first and last cut, with the entry of every ring.

    Returns the letters saved against the present order, or SKIP. `order[i]` is the piece placed i-th
    and `turn[i]` how far it is turned (0 unless it is a ring). When `anchor` is one of the cuts, the step
    across it is forbidden, so the answer is the best order without that step, and the saving can be
    negative. Among equally short answers one is drawn at random."""
    N = route.shape[0]
    K = perms.shape[1]
    m = n - 1
    if m < 1 or m > MMAX:
        return SKIP
    lo = cuts[0]
    hi = cuts[n - 1]
    Q = 0
    for j in range(m):
        a = cuts[j] + 1
        b = cuts[j + 1]
        ring = b - a + 1 == K
        if ring:
            for i in range(a, b):
                if wout[route[i]] != 1:
                    ring = False
                    break
        base[j] = Q
        if ring:
            if Q + K > QMAX:
                return SKIP
            for r in range(K):
                head[Q] = route[a + r]
                tail[Q] = route[a + (r - 1 + K) % K]
                item[Q] = j
                Q += 1
        else:
            if Q + 1 > QMAX:
                return SKIP
            head[Q] = route[a]
            tail[Q] = route[b]
            item[Q] = j
            Q += 1
    base[m] = Q
    gate_tail = route[lo] if lo >= 0 else -1
    gate_head = route[hi + 1] if hi < N - 1 else -1

    old = 0
    for i in range(n):
        c = cuts[i]
        if 0 <= c < N - 1:
            old += wout[route[c]]

    noise = SCALE // (m + 2)
    ban_tail = -1
    ban_head = -1
    if 0 <= anchor < N - 1:
        ban_tail = route[anchor]
        ban_head = route[anchor + 1]
    for a in range(Q):
        for b in range(Q):
            if item[a] != item[b]:
                C[a, b] = step(perms, inv, tail[a], head[b]) * SCALE + np.random.randint(noise)
                if tail[a] == ban_tail and head[b] == ban_head:
                    C[a, b] += BAN
    full = (1 << m) - 1
    for mask in range(1, full + 1):
        for a in range(Q):
            f[mask, a] = INF
    for a in range(Q):
        s = 0
        if gate_tail >= 0:
            s = step(perms, inv, gate_tail, head[a]) * SCALE + np.random.randint(noise)
            if gate_tail == ban_tail and head[a] == ban_head:
                s += BAN
        f[1 << item[a], a] = s
        par[1 << item[a], a] = -1
    for mask in range(1, full + 1):
        for a in range(Q):
            fa = f[mask, a]
            if fa >= INF:
                continue
            for b in range(Q):
                jb = item[b]
                if (mask >> jb) & 1:
                    continue
                nm = mask | (1 << jb)
                v = fa + C[a, b]
                if v < f[nm, b]:
                    f[nm, b] = v
                    par[nm, b] = a
    best = INF
    last = -1
    for a in range(Q):
        v = f[full, a]
        if v >= INF:
            continue
        if gate_head >= 0:
            v += step(perms, inv, tail[a], gate_head) * SCALE + np.random.randint(noise)
            if tail[a] == ban_tail and gate_head == ban_head:
                v += BAN
        if v < best:
            best = v
            last = a
    mask = full
    a = last
    i = m - 1
    while a >= 0:
        order[i] = item[a]
        turn[i] = a - base[item[a]]
        opt[i] = a
        parent = par[mask, a]
        mask ^= 1 << item[a]
        a = parent
        i -= 1
    new = 0
    if gate_tail >= 0:
        new += step(perms, inv, gate_tail, head[opt[0]])
    for i in range(m - 1):
        new += step(perms, inv, tail[opt[i]], head[opt[i + 1]])
    if gate_head >= 0:
        new += step(perms, inv, tail[opt[m - 1]], gate_head)
    return old - new


@jit
def apply(cuts, n, order, turn, route, pos, wout, perms, inv, buf):
    """Re-order the pieces between the first and last cut as `order` and `turn` say."""
    N = route.shape[0]
    m = n - 1
    lo = cuts[0]
    hi = cuts[n - 1]
    t = 0
    for i in range(m):
        j = order[i]
        a = cuts[j] + 1
        b = cuts[j + 1]
        L = b - a + 1
        r = turn[i]
        if r != 0:
            wout[route[b]] = 1
        for x in range(L):
            buf[t + x] = route[a + (x + r) % L]
        t += L
    for x in range(t):
        route[lo + 1 + x] = buf[x]
        pos[buf[x]] = lo + 1 + x
    t = lo
    for i in range(m):
        if t >= 0:
            wout[route[t]] = step(perms, inv, route[t], route[t + 1])
        j = order[i]
        t += cuts[j + 1] - cuts[j]
    if hi < N - 1:
        wout[route[hi]] = step(perms, inv, route[hi], route[hi + 1])
    else:
        wout[route[hi]] = 0


@jit
def _anchor(route, wout, n_sym, draws):
    """A random step of at least two letters (or, rarely, an end of the route)."""
    N = route.shape[0]
    r = np.random.randint(64)
    if r == 0:
        return -1
    if r == 1:
        return N - 1
    best = -2
    for _ in range(draws):
        x = np.random.randint(N - 1)
        for _ in range(2 * n_sym):
            if wout[route[x]] >= 2:
                break
            x += 1
            if x >= N - 1:
                x = 0
        if best < 0 or wout[route[x]] > wout[route[best]]:
            best = x
    return best


@jit
def descend(iters, kmin, kmax, temp, p_tie, p_ring, p_chain, draws, cost, route, pos, wout, perms, inv,
            out_nbr, in_nbr, colcum, f, par, C, head, tail, item, base, order, turn, opt, cuts, buf, stats):
    """`iters` chains, each grown from a random costly step and solved exactly with that step forbidden.

    A change that saves letters is applied. One that costs nothing is applied with probability `p_tie`.
    One that costs d letters is applied with probability exp(-d / temp), never at temp = 0. Returns the
    letters of all steps. stats: solved, improving, level, letters saved, worsening."""
    K = perms.shape[1]
    for _ in range(iters):
        anchor = _anchor(route, wout, K, draws)
        kk = kmin + np.random.randint(kmax - kmin + 1)
        n = grow(anchor, kk, route, pos, wout, out_nbr, in_nbr, colcum, K, p_ring, p_chain, cuts)
        if n < 2:
            continue
        gain = solve(cuts, n, anchor, route, wout, perms, inv, f, par, C, head, tail, item, base, order, turn, opt)
        if gain == SKIP:
            continue
        stats[0] += 1
        moved = False
        for i in range(n - 1):
            if order[i] != i or turn[i] != 0:
                moved = True
                break
        if not moved:
            continue
        if gain > 0:
            apply(cuts, n, order, turn, route, pos, wout, perms, inv, buf)
            cost -= gain
            stats[1] += 1
            stats[3] += gain
        elif gain == 0:
            if np.random.random() < p_tie:
                apply(cuts, n, order, turn, route, pos, wout, perms, inv, buf)
                stats[2] += 1
        elif temp > 0 and np.random.random() < np.exp(gain / temp):
            apply(cuts, n, order, turn, route, pos, wout, perms, inv, buf)
            cost -= gain
            stats[4] += 1
    return cost


@jit
def kick(count, span, cost, route, pos, wout, perms, inv, cuts, order, turn, buf):
    """Swap two neighbouring stretches of the route `count` times, whatever it costs. Returns the letters
    of all steps."""
    N = route.shape[0]
    for _ in range(count):
        a = np.random.randint(N - 3)
        b = a + 1 + np.random.randint(span)
        c = b + 1 + np.random.randint(span)
        if c > N - 2:
            continue
        cuts[0] = a
        cuts[1] = b
        cuts[2] = c
        before = wout[route[a]] + wout[route[b]] + wout[route[c]]
        order[0] = 1
        order[1] = 0
        turn[0] = 0
        turn[1] = 0
        apply(cuts, 3, order, turn, route, pos, wout, perms, inv, buf)
        after = wout[route[a]] + wout[route[a + c - b]] + wout[route[c]]
        cost += after - before
    return cost
