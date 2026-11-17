"""The Superpermutation Almanac's Detour format, in which a word of 14 symbols (94 billion letters) takes
under a megabyte.

A Detour word is a stream of tokens. Each token says how many default steps of the recursive construction
come first, then gives one step that differs from the default. `read` decodes a file to its letters and
`write` encodes a word. `Segments` walks a file step by step without writing the word out, and returns every
step of weight T or more: the stretches between them are the segments that `splice` puts in a new order.
A step token may hold several real steps (a detour that passes through orderings on its way), so a token is
always split into its real steps, and only a real step is ever a place to cut.
"""
from __future__ import annotations

import hashlib
import lzma
import math

import numpy as np

from ._jit import jit, need_compiled
from .orderings import SYMBOLS

ALPHABET = SYMBOLS.encode()
NREC = 13
_SYM = np.frombuffer(ALPHABET, dtype=np.uint8)
_XZ = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]


def _first(n):
    first = [0, 3]
    for k in range(1, n + 1):
        first.append(first[-1] + math.factorial(k))
    return first


def load(path):
    """(the file's bytes after decompression, n, offset of the first token)."""
    data = lzma.open(path, "rb").read()
    if data[:6] != b"Detour" or data[6] != 1:
        raise ValueError(f"{path} is not a Detour file")
    n = data[7]
    if bytes(data[8:8 + n]) != ALPHABET[:n]:
        raise ValueError("the alphabet of a Detour file must be 0123...")
    return np.frombuffer(data, dtype=np.uint8), n, 8 + n


# ---- decoding and encoding whole words ---------------------------------------------------------------

@jit
def _defaults(window, counters, G, out, pos):
    """Take G default steps of the recursive construction from `window`."""
    n = window.shape[0]
    tmp = np.empty(n, dtype=np.uint8)
    for _ in range(G):
        d = 0
        for k in range(1, n):
            if counters[k] < n - k:
                d = k
                break
        if d == 0:
            return -1
        for i in range(d):
            tmp[i] = window[d - 1 - i]
        for i in range(n - d):
            window[i] = window[i + d]
        for i in range(d):
            window[n - d + i] = tmp[i]
            out[pos] = tmp[i]
            pos += 1
        counters[d] += 1
        for k in range(1, d):
            counters[k] = 0
    return pos


def read(path) -> str:
    """The letters of the word in a Detour file. The file's own SHA-256 of the word is checked."""
    data, n, i = load(path)
    data = data.tobytes()
    room = sum(math.factorial(j) for j in range(1, n + 1)) + 16
    out = np.zeros(room, dtype=np.uint8)
    pos = 0
    window = np.zeros(n, dtype=np.uint8)
    counters = np.zeros(n + 1, dtype=np.int64)
    have = False
    first = _first(n)

    def varint():
        nonlocal i
        v = shift = 0
        while True:
            b = data[i]
            i += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                return v

    while True:
        G = varint()
        code = varint()
        if G:
            if not have:
                raise ValueError("default steps with no window")
            pos = _defaults(window, counters, G, out, pos)
            if pos < 0:
                raise ValueError("a default step past the end of the recursive construction")
        if code == 0:
            break
        if code == 1:
            window[:] = np.frombuffer(data[i:i + n], dtype=np.uint8)
            i += n
            out[pos:pos + n] = window
            pos += n
            counters[:] = 0
            have = True
        elif code == 2:
            m = varint()
            out[pos:pos + m] = np.frombuffer(data[i:i + m], dtype=np.uint8)
            i += m
            pos += m
            have = False
        else:
            k = 1
            while code >= first[k + 1]:
                k += 1
            r = code - first[k]
            ref = [int(window[j]) for j in range(k - 1, -1, -1)]
            added = []
            for j in range(k):
                f = math.factorial(k - 1 - j)
                added.append(ref.pop(r // f))
                r %= f
            window[:n - k] = window[k:].copy()
            window[n - k:] = added
            out[pos:pos + k] = added
            pos += k
            counters[:] = 0
    word = out[:pos].tobytes()
    if hashlib.sha256(word).digest() != data[i:i + 32]:
        raise ValueError("the decoded word does not match the file's hash")
    return word.decode()


@jit
def _put(out, o, v):
    while v >= 0x80:
        out[o] = (v & 0x7F) | 0x80
        o += 1
        v >>= 7
    out[o] = v
    return o + 1


@jit
def _encode(a, n, out):
    """Tokens of the word `a` (letters 0 .. n-1, beginning with an ordering) after the START token.
    Returns the bytes written, or -1 if the word is not made of steps between orderings."""
    L = a.shape[0]
    fact = np.ones(n + 2, dtype=np.int64)
    for k in range(1, n + 2):
        fact[k] = fact[k - 1] * k
    first = np.zeros(n + 3, dtype=np.int64)
    first[1] = 3
    for k in range(1, n + 1):
        first[k + 1] = first[k] + fact[k]
    c = np.zeros(n + 2, dtype=np.int64)
    window = a[:n].copy()
    used = np.zeros(n, dtype=np.uint8)
    o = 0
    pending = 0
    p = n
    while p < L:
        k = 0
        md = 0
        ma = 0
        for t in range(1, n + 1):
            if p + t > L:
                break
            md |= 1 << window[t - 1]
            ma |= 1 << a[p + t - 1]
            if md == ma:
                k = t
                break
        if k == 0:
            return -1
        d = 0
        for kk in range(1, n):
            if c[kk] < n - kk:
                d = kk
                break
        default = k == d
        if default:
            for t in range(k):
                if a[p + t] != window[k - 1 - t]:
                    default = False
                    break
        if default:
            pending += 1
            c[d] += 1
            for kk in range(1, d):
                c[kk] = 0
        else:
            r = 0
            for t in range(k):
                used[t] = 0
            for t in range(k):
                q = 0
                for s in range(k):
                    if used[s]:
                        continue
                    if window[k - 1 - s] == a[p + t]:
                        used[s] = 1
                        break
                    q += 1
                r += q * fact[k - 1 - t]
            o = _put(out, o, pending)
            o = _put(out, o, first[k] + r)
            pending = 0
            for kk in range(n + 2):
                c[kk] = 0
        for t in range(n - k):
            window[t] = window[t + k]
        for t in range(k):
            window[n - k + t] = a[p + t]
        p += k
    o = _put(out, o, pending)
    o = _put(out, o, 0)
    return o


def write(word, path) -> int:
    """Write a word as a Detour file, the form the Almanac takes from 10 symbols up. Returns its letters.
    The word is relabelled to begin 0, 1, 2 ... first."""
    from .words import _letters, normalise
    text = normalise(word)
    a, n = _letters(text)
    if a[:n].tolist() != list(range(n)):
        raise ValueError("the word does not begin with an ordering")
    out = np.zeros(4 * len(a) + 64, dtype=np.uint8)
    size = _encode(a, n, out)
    if size < 0:
        raise ValueError("the word has letters that are not part of a step between orderings")
    blob = (b"Detour" + bytes([1, n]) + ALPHABET[:n] + bytes([0, 1]) + ALPHABET[:n] + out[:size].tobytes()
            + hashlib.sha256(text.encode()).digest())
    with open(path, "wb") as f:
        f.write(lzma.compress(blob, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=_XZ))
    return len(a)


# ---- walking a large word step by step ---------------------------------------------------------------

@jit
def _key(window, n):
    k = np.int64(0)
    for t in range(n):
        k = (k << 4) | np.int64(window[t])
    return k


@jit
def _walk(data, i, n, T, cap, rw, rbefore, rafter, rpos, rtok, rj, rnext, rG, rca, rpar, rsub, rnsub, rsoff,
          subcodes, hist, info):
    """Returns (status, steps recorded, letters, windows, offset after END, last window key).
    status 0 = fine, 1 = out of room, 2 = a stream this walker does not take."""
    window = np.zeros(n, dtype=np.uint8)
    tmp = np.zeros(n, dtype=np.uint8)
    used = np.zeros(n, dtype=np.uint8)
    dropped = np.zeros(n, dtype=np.uint8)
    added = np.zeros(n, dtype=np.uint8)
    bounds = np.zeros(n + 1, dtype=np.int64)
    subtmp = np.zeros(n + 1, dtype=np.int64)
    c = np.zeros(n + 2, dtype=np.int64)
    fact = np.ones(n + 2, dtype=np.int64)
    for k in range(1, n + 2):
        fact[k] = fact[k - 1] * k
    first = np.zeros(n + 3, dtype=np.int64)
    first[1] = 3
    for k in range(1, n + 1):
        first[k + 1] = first[k] + fact[k]
    run = n * (n - 1) - 1
    have = False
    letters = 0
    windows = 0
    cnt = 0
    cnt4 = 0
    nsc = 0
    while True:
        tok = i
        G = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            G |= np.int64(b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        code = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            code |= np.int64(b & 0x7F) << shift
            shift += 7
            if b < 0x80:
                break
        if G > 0 and not have:
            return 2, cnt, letters, windows, i, 0
        Gtot = G
        done = 0
        while G > 0:
            if c[1] == 0 and c[2] == 0 and G >= run:
                a = window[n - 1]
                b2 = window[n - 2]
                for t in range(n - 1, 1, -1):
                    window[t] = window[t - 2]
                window[0] = a
                window[1] = b2
                c[1] = n - 1
                c[2] = n - 2
                G -= run
                done += run
                windows += run
                letters += (n - 1) * (n - 1) + 2 * (n - 2)
                hist[1] += (n - 1) * (n - 1)
                hist[2] += n - 2
            elif c[1] == 0 and G >= n - 1:
                a = window[n - 1]
                for t in range(n - 1, 0, -1):
                    window[t] = window[t - 1]
                window[0] = a
                c[1] = n - 1
                G -= n - 1
                done += n - 1
                windows += n - 1
                letters += n - 1
                hist[1] += n - 1
            else:
                d = 0
                for k in range(1, n):
                    if c[k] < n - k:
                        d = k
                        break
                if d == 0:
                    return 2, cnt, letters, windows, i, 0
                rec = d >= T
                if rec:
                    if cnt >= cap:
                        return 1, cnt, letters, windows, i, 0
                    rw[cnt] = d
                    rbefore[cnt] = _key(window, n)
                    rtok[cnt] = tok
                    rj[cnt] = done
                    rnext[cnt] = i
                    rG[cnt] = Gtot
                    rsub[cnt] = -1
                    rnsub[cnt] = 0
                    rsoff[cnt] = 0
                for t in range(d):
                    tmp[t] = window[d - 1 - t]
                for t in range(n - d):
                    window[t] = window[t + d]
                for t in range(d):
                    window[n - d + t] = tmp[t]
                c[d] += 1
                for k in range(1, d):
                    c[k] = 0
                letters += d
                windows += 1
                hist[d] += 1
                if d >= 4:
                    cnt4 += 1
                if rec:
                    rafter[cnt] = _key(window, n)
                    rpos[cnt] = letters - n
                    ca = np.int64(0)
                    for k in range(1, n):
                        ca |= c[k] << (4 * k)
                    rca[cnt] = ca
                    rpar[cnt] = cnt4
                    cnt += 1
                G -= 1
                done += 1
        if code == 0:
            info[0] = tok
            info[1] = Gtot
            info[3] = nsc
            return 0, cnt, letters, windows, i, _key(window, n)
        if code == 1:
            if have:
                return 2, cnt, letters, windows, i, 0
            for t in range(n):
                v = data[i + t]
                window[t] = v - 48 if v < 58 else v - 55
            i += n
            letters += n
            windows += 1
            for k in range(n + 2):
                c[k] = 0
            have = True
        elif code == 2:
            return 2, cnt, letters, windows, i, 0
        else:
            k = 1
            while code >= first[k + 1]:
                k += 1
            r = code - first[k]
            for t in range(k):
                tmp[t] = window[k - 1 - t]
                used[t] = 0
                dropped[t] = window[t]
            for jj in range(k):
                f = fact[k - 1 - jj]
                q = r // f
                r = r % f
                for t in range(k):
                    if used[t] == 0:
                        if q == 0:
                            used[t] = 1
                            added[jj] = tmp[t]
                            break
                        q -= 1
            md = 0
            ma = 0
            nsub = 0
            for p in range(1, k + 1):
                md |= 1 << dropped[p - 1]
                ma |= 1 << added[p - 1]
                if md == ma:
                    bounds[nsub] = p
                    nsub += 1
            p0 = 0
            anyrec = False
            for q in range(nsub):
                p = bounds[q]
                wq = p - p0
                rr = 0
                for t in range(wq):
                    used[t] = 0
                for ai in range(wq):
                    sym = added[p0 + ai]
                    pp = 0
                    for t in range(wq):
                        if used[t]:
                            continue
                        if dropped[p - 1 - t] == sym:
                            used[t] = 1
                            break
                        pp += 1
                    rr += pp * fact[wq - 1 - ai]
                subtmp[q] = first[wq] + rr
                rec = wq >= T
                if rec:
                    if cnt >= cap:
                        return 1, cnt, letters, windows, i, 0
                    rw[cnt] = wq
                    rbefore[cnt] = _key(window, n)
                    rtok[cnt] = tok
                    rj[cnt] = -1
                    rnext[cnt] = i
                    rG[cnt] = Gtot
                    rsub[cnt] = q
                    rnsub[cnt] = nsub
                    rsoff[cnt] = nsc
                for t in range(n - wq):
                    window[t] = window[t + wq]
                for t in range(wq):
                    window[n - wq + t] = added[p0 + t]
                letters += wq
                windows += 1
                hist[wq] += 1
                if wq >= 4:
                    cnt4 += 1
                if rec:
                    rafter[cnt] = _key(window, n)
                    rpos[cnt] = letters - n
                    rca[cnt] = 0
                    rpar[cnt] = cnt4
                    cnt += 1
                    anyrec = True
                p0 = p
            if anyrec and nsub > 1:
                if nsc + nsub > subcodes.shape[0]:
                    return 1, cnt, letters, windows, i, 0
                for q in range(nsub):
                    subcodes[nsc + q] = subtmp[q]
                nsc += nsub
            for kk in range(n + 2):
                c[kk] = 0


class Segments:
    """Every real step of weight T or more in a Detour file, and the segments between them.

    Segment s runs from the ordering start[s] to the ordering stop[s] (as 4-bit keys); w[s] is the weight
    of the step that ends it in the file's own order, and pos[s] the letter where the step's ordering begins.
    """

    def __init__(self, path, T: int = 4, cap: int = 1 << 20):
        need_compiled("Walking a Detour file")
        self.path = str(path)
        self.data, self.n, self.i0 = load(path)
        n = self.n
        self.T = T
        while True:
            arr = [np.zeros(cap, dtype=np.int64) for _ in range(NREC)]
            subcodes = np.zeros(cap, dtype=np.int64)
            hist = np.zeros(n + 2, dtype=np.int64)
            info = np.zeros(4, dtype=np.int64)
            st, cnt, letters, windows, end, lastkey = _walk(self.data, self.i0, n, T, cap, *arr, subcodes, hist, info)
            if st == 1:
                cap *= 4
                continue
            if st == 2:
                raise ValueError("the token stream is not a plain word of steps")
            break
        (self.w, self.before, self.after, self.pos, self.tok, self.j, self.next, self.G, rca, rpar,
         self.sub, self.nsub, self.soff) = [a[:cnt].copy() for a in arr]
        self.subcodes = subcodes[:int(info[3])].copy()
        self.ca = np.concatenate(([0], rca))
        self.letters = int(letters)
        self.windows = int(windows)
        self.hist = hist
        self.end = int(end)
        self.end_tok, self.end_G = int(info[0]), int(info[1])
        self.sha = bytes(self.data[end:end + 32])
        first_key = int(_key(np.arange(n, dtype=np.uint8), n))
        self.start = np.concatenate(([first_key], self.after))
        self.stop = np.concatenate((self.before, [lastkey]))
        self.nseg = cnt + 1


# ---- writing a word whose segments come in a new order -----------------------------------------------

@jit
def _xstep(out, o, pending, dc, code, first, n):
    """Write one real step given as a step code: as a default step if it is the default here, else as a token."""
    k = 1
    while code >= first[k + 1]:
        k += 1
    dd = 0
    for kk in range(1, n):
        if dc[kk] < n - kk:
            dd = kk
            break
    if code == first[k] and k == dd:
        pending += 1
        dc[k] += 1
        for kk in range(1, k):
            dc[kk] = 0
    else:
        o = _put(out, o, pending)
        o = _put(out, o, code)
        pending = 0
        for kk in range(n + 2):
            dc[kk] = 0
    return o, pending


@jit
def _splice(data, n, i0, order, tok, j, nxt, G, ca, start, stop, end_tok, end_G, sub, nsub, soff, subcodes, out):
    """Tokens of the re-ordered word. The steps of each segment are read from its tokens with the counters
    the old word had there, and written against the counters the new word has, so nothing is copied blindly.
    A token of several real steps that holds a cut is written step by step."""
    N = len(order)
    fact = np.ones(n + 2, dtype=np.int64)
    for k in range(1, n + 2):
        fact[k] = fact[k - 1] * k
    first = np.zeros(n + 3, dtype=np.int64)
    first[1] = 3
    for k in range(1, n + 1):
        first[k + 1] = first[k] + fact[k]
    dc = np.zeros(n + 2, dtype=np.int64)
    sc = np.zeros(n + 2, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)
    run = n * (n - 1) - 1
    o = 0
    head = i0 + 2 + n
    if data[i0] != 0 or data[i0 + 1] != 1:
        return -1
    for t in range(head):
        out[o] = data[t]
        o += 1
    pending = 0
    for t in range(N):
        s = order[t]
        same_token = s >= 1 and s < N - 1 and j[s - 1] < 0 and j[s] < 0 and tok[s - 1] == tok[s]
        if same_token:
            for q in range(sub[s - 1] + 1, sub[s]):
                o, pending = _xstep(out, o, pending, dc, subcodes[soff[s] + q], first, n)
        else:
            if s == 0:
                off = head
                gskip = 0
            elif j[s - 1] < 0:
                off = nxt[s - 1]
                gskip = 0
                for q in range(sub[s - 1] + 1, nsub[s - 1]):
                    o, pending = _xstep(out, o, pending, dc, subcodes[soff[s - 1] + q], first, n)
            else:
                off = tok[s - 1]
                gskip = j[s - 1] + 1
            if s < N - 1:
                toff = tok[s]
                gend = G[s] if j[s] < 0 else j[s]
            else:
                toff = end_tok
                gend = end_G
            for k in range(1, n):
                sc[k] = (ca[s] >> (4 * k)) & 15
            i = off
            firsttok = True
            while True:
                last = i == toff
                if last:
                    g = gend
                else:
                    g = 0
                    shift = 0
                    while True:
                        b = data[i]
                        i += 1
                        g |= np.int64(b & 0x7F) << shift
                        shift += 7
                        if b < 0x80:
                            break
                if firsttok:
                    g -= gskip
                firsttok = False
                while g > 0:
                    if sc[1] == 0 and sc[2] == 0 and dc[1] == 0 and dc[2] == 0 and g >= run:
                        pending += run
                        g -= run
                        sc[1] = n - 1
                        sc[2] = n - 2
                        dc[1] = n - 1
                        dc[2] = n - 2
                    elif sc[1] == 0 and dc[1] == 0 and g >= n - 1:
                        pending += n - 1
                        g -= n - 1
                        sc[1] = n - 1
                        dc[1] = n - 1
                    else:
                        ds = 0
                        for k in range(1, n):
                            if sc[k] < n - k:
                                ds = k
                                break
                        if ds == 0:
                            return -4
                        o, pending = _xstep(out, o, pending, dc, first[ds], first, n)
                        sc[ds] += 1
                        for k in range(1, ds):
                            sc[k] = 0
                        g -= 1
                if last:
                    break
                code = 0
                shift = 0
                while True:
                    b = data[i]
                    i += 1
                    code |= np.int64(b & 0x7F) << shift
                    shift += 7
                    if b < 0x80:
                        break
                if code < 3:
                    return -5
                o, pending = _xstep(out, o, pending, dc, code, first, n)
                for kk in range(n + 2):
                    sc[kk] = 0
            if s < N - 1 and j[s] < 0:
                for q in range(0, sub[s]):
                    o, pending = _xstep(out, o, pending, dc, subcodes[soff[s] + q], first, n)
        if t == N - 1:
            break
        s2 = order[t + 1]
        e = stop[s]
        b2 = start[s2]
        k = n
        for ov in range(n - 1, 0, -1):
            if (e & ((np.int64(1) << (4 * ov)) - 1)) == (b2 >> (4 * (n - ov))):
                k = n - ov
                break
        r = 0
        for a in range(k):
            used[a] = 0
        for a in range(k):
            sym = (b2 >> (4 * (k - 1 - a))) & 15
            p = 0
            found = False
            for q in range(k):
                ref = (e >> (4 * (n - 1 - (k - 1 - q)))) & 15
                if used[q]:
                    continue
                if ref == sym:
                    used[q] = 1
                    found = True
                    break
                p += 1
            if not found:
                return -3
            r += p * fact[k - 1 - a]
        o, pending = _xstep(out, o, pending, dc, first[k] + r, first, n)
    o = _put(out, o, pending)
    o = _put(out, o, 0)
    return o


@jit
def _stream(data, n, st, window, c, out):
    """Fill `out` with the next letters of the word. st = [offset, G left, code, phase, done, error].
    Returns how many letters were written."""
    fact = np.ones(n + 2, dtype=np.int64)
    for k in range(1, n + 2):
        fact[k] = fact[k - 1] * k
    first = np.zeros(n + 3, dtype=np.int64)
    first[1] = 3
    for k in range(1, n + 1):
        first[k + 1] = first[k] + fact[k]
    tmp = np.zeros(n, dtype=np.uint8)
    used = np.zeros(n, dtype=np.uint8)
    room = out.shape[0]
    pos = 0
    i = st[0]
    G = st[1]
    code = st[2]
    phase = st[3]
    while True:
        if phase == 0:
            G = 0
            shift = 0
            while True:
                b = data[i]
                i += 1
                G |= np.int64(b & 0x7F) << shift
                shift += 7
                if b < 0x80:
                    break
            code = 0
            shift = 0
            while True:
                b = data[i]
                i += 1
                code |= np.int64(b & 0x7F) << shift
                shift += 7
                if b < 0x80:
                    break
            phase = 1
        if phase == 1:
            while G > 0:
                if room - pos < 2 * n:
                    st[0] = i
                    st[1] = G
                    st[2] = code
                    st[3] = phase
                    return pos
                if c[1] == 0 and G >= n - 1:
                    for t in range(n - 1):
                        out[pos + t] = window[t]
                    pos += n - 1
                    a = window[n - 1]
                    for t in range(n - 1, 0, -1):
                        window[t] = window[t - 1]
                    window[0] = a
                    c[1] = n - 1
                    G -= n - 1
                else:
                    d = 0
                    for k in range(1, n):
                        if c[k] < n - k:
                            d = k
                            break
                    if d == 0:
                        st[5] = 1
                        return pos
                    for t in range(d):
                        tmp[t] = window[d - 1 - t]
                    for t in range(n - d):
                        window[t] = window[t + d]
                    for t in range(d):
                        window[n - d + t] = tmp[t]
                        out[pos] = tmp[t]
                        pos += 1
                    c[d] += 1
                    for k in range(1, d):
                        c[k] = 0
                    G -= 1
            phase = 2
        if room - pos < 2 * n:
            st[0] = i
            st[1] = G
            st[2] = code
            st[3] = phase
            return pos
        if code == 0:
            st[0] = i
            st[4] = 1
            return pos
        if code == 1:
            for t in range(n):
                v = data[i + t]
                window[t] = v - 48 if v < 58 else v - 55
                out[pos] = window[t]
                pos += 1
            i += n
            for k in range(n + 2):
                c[k] = 0
        elif code == 2:
            st[5] = 2
            return pos
        else:
            k = 1
            while code >= first[k + 1]:
                k += 1
            r = code - first[k]
            d = 0
            for kk in range(1, n):
                if c[kk] < n - kk:
                    d = kk
                    break
            if r == 0 and k == d:
                st[5] = 3
                return pos
            for t in range(k):
                tmp[t] = window[k - 1 - t]
                used[t] = 0
            for t in range(n - k):
                window[t] = window[t + k]
            for jj in range(k):
                f = fact[k - 1 - jj]
                q = r // f
                r = r % f
                for t in range(k):
                    if used[t] == 0:
                        if q == 0:
                            used[t] = 1
                            window[n - k + jj] = tmp[t]
                            out[pos] = tmp[t]
                            pos += 1
                            break
                        q -= 1
            for kk in range(n + 2):
                c[kk] = 0
        phase = 0


def sha_of_tokens(data, n, i0, chunk=1 << 27, log=None):
    """(SHA-256 of the word, its length, offset after END), streamed from the tokens without holding the word.
    Also refuses a step token equal to the default step, as the Almanac does."""
    st = np.zeros(6, dtype=np.int64)
    st[0] = i0
    window = np.zeros(n, dtype=np.uint8)
    c = np.zeros(n + 2, dtype=np.int64)
    out = np.zeros(chunk, dtype=np.uint8)
    h = hashlib.sha256()
    total = 0
    nextlog = 1 << 33
    while True:
        k = _stream(data, n, st, window, c, out)
        if st[5]:
            raise ValueError(f"bad token stream (error {int(st[5])}) at byte {int(st[0])}")
        h.update(_SYM[out[:k]].tobytes())
        total += k
        if log and total >= nextlog:
            log(total)
            nextlog += 1 << 33
        if st[4]:
            break
    return h.digest(), total, int(st[0])


def splice(S: Segments, order, path, log=None) -> int:
    """Write the word with the segments of S in `order` as a Detour file. Returns its length in letters."""
    need_compiled("Writing a Detour file")
    n, data = S.n, S.data
    out = np.zeros(2 * len(data) + 24 * S.nseg + 4096, dtype=np.uint8)
    size = _splice(data, n, S.i0, np.asarray(order, dtype=np.int64), S.tok, S.j, S.next, S.G, S.ca, S.start, S.stop,
                   S.end_tok, S.end_G, S.sub, S.nsub, S.soff, S.subcodes, out)
    if size < 0:
        raise ValueError(f"cannot write this order (error {size})")
    tokens = out[:size]
    sha, total, end = sha_of_tokens(tokens, n, S.i0, log=log)
    if end != size:
        raise ValueError("the written token stream does not end where it should")
    with open(path, "wb") as f:
        f.write(lzma.compress(tokens.tobytes() + sha, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=_XZ))
    return total
