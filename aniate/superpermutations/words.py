"""Words: checking them, building one, turning them into routes and back, and where their letters go.

`check` scans a word window by window and shares no code with the search, so it can judge the search.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .orderings import SYMBOLS, Orderings, ranks

CHUNK = 1 << 20
MAX_CHECK = 12
_SYM = np.frombuffer(SYMBOLS.encode(), dtype=np.uint8)


@dataclass(frozen=True)
class Check:
    """What a scan of every window found."""
    n: int             # symbols
    length: int        # letters
    orderings: int     # distinct orderings that occur
    total: int         # n!
    repeats: int       # windows that repeat an ordering already seen

    @property
    def missing(self) -> int:
        return self.total - self.orderings

    @property
    def complete(self) -> bool:
        return self.orderings == self.total

    def __bool__(self):
        return self.complete

    def __str__(self):
        if self.complete:
            return f"superpermutation on {self.n} symbols: {self.length:,} letters, all {self.total:,} orderings"
        return f"not a superpermutation: {self.orderings:,} of {self.total:,} orderings, {self.missing:,} missing"


def _letters(word) -> tuple[np.ndarray, int]:
    text = word.text if hasattr(word, "text") else str(word)
    text = text.strip()
    symbols = sorted(set(text))
    n = len(symbols)
    if n > len(SYMBOLS):
        raise ValueError(f"a word may use at most {len(SYMBOLS)} symbols; this one uses {n}")
    code = np.zeros(1 << 16, dtype=np.uint8)
    for i, c in enumerate(symbols):
        code[ord(c)] = i
    raw = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16)
    return code[raw], n


def check(word) -> Check:
    """Scan every window of n letters. The word may use any n distinct characters."""
    a, n = _letters(word)
    if n > MAX_CHECK:
        raise ValueError(f"a word on {n} symbols is too long to hold as text; read it from its Detour file")
    total = math.factorial(n)
    if n < 1 or len(a) < n:
        return Check(n, len(a), 0, total, 0)
    seen = np.zeros(total, dtype=np.bool_)
    full = (1 << n) - 1
    found = 0
    for lo in range(0, len(a) - n + 1, CHUNK):
        hi = min(len(a) - n + 1, lo + CHUNK)
        windows = np.lib.stride_tricks.sliding_window_view(a[lo:hi + n - 1], n)
        masks = np.bitwise_or.reduce(np.left_shift(1, windows.astype(np.int64)), axis=1)
        valid = windows[masks == full]
        found += len(valid)
        seen[ranks(valid)] = True
    distinct = int(seen.sum())
    return Check(n, len(a), distinct, total, found - distinct)


def normalise(word) -> str:
    """Relabel the symbols so that the word begins 0, 1, 2 ..., as the Superpermutation Almanac asks."""
    text = (word.text if hasattr(word, "text") else str(word)).strip()
    n = len(set(text))
    head = text[:n]
    if len(set(head)) != n:
        raise ValueError("the word does not begin with an ordering")
    table = str.maketrans({c: SYMBOLS[i] for i, c in enumerate(head)})
    return text.translate(table)


def recursive(n: int) -> str:
    """The classic construction: each ordering p of n-1 symbols becomes p, n, p, and neighbours overlap.
    It has 1! + 2! + ... + n! letters, which is the shortest possible up to n = 5."""
    if not 1 <= n <= MAX_RECURSIVE:
        raise ValueError(f"n must be between 1 and {MAX_RECURSIVE}")
    word = "0"
    for k in range(2, n + 1):
        new = SYMBOLS[k - 1]
        seen, perms = set(), []
        for i in range(len(word) - k + 2):
            window = word[i:i + k - 1]
            if window not in seen and len(set(window)) == k - 1:
                seen.add(window)
                perms.append(window)
        parts = [perms[0] + new + perms[0]]
        last = parts[0]
        for p in perms[1:]:
            block = p + new + p
            o = next(j for j in range(k - 1, -1, -1) if last.endswith(block[:j]))
            parts.append(block[o:])
            last = block
        word = "".join(parts)
    return word


MAX_RECURSIVE = 11


def route_from_word(word, tables: Orderings | None = None) -> tuple[np.ndarray, Orderings]:
    """The orderings of a word in the order they first appear, as ordering numbers."""
    a, n = _letters(word)
    tables = tables or Orderings(n)
    windows = np.lib.stride_tricks.sliding_window_view(a, n)
    full = (1 << n) - 1
    masks = np.bitwise_or.reduce(np.left_shift(1, windows.astype(np.int64)), axis=1)
    numbers = ranks(windows[masks == full])
    _, first = np.unique(numbers, return_index=True)
    route = numbers[np.sort(first)].astype(np.int32)
    if len(route) != tables.N:
        raise ValueError(f"the word holds {len(route):,} of {tables.N:,} orderings")
    return route, tables


def word_from_route(tables: Orderings, route: np.ndarray) -> str:
    """The shortest word that meets the orderings in this order, beginning 0, 1, 2 ..."""
    from .engine import step
    perms, inv, n = tables.perms, tables.inv, tables.n
    out = [perms[route[0]]]
    for a, b in zip(route[:-1].tolist(), route[1:].tolist()):
        out.append(perms[b][n - step(perms, inv, a, b):])
    return normalise(_SYM[np.concatenate(out)].tobytes().decode())


def ledger(word) -> dict:
    """Where a word spends its letters.

    A class is the n rotations of one ordering. An arc is a stretch of new orderings one letter apart.
    A join is the step between two arcs, and a run is a stretch of arcs joined by two letters. With every
    class read in one arc, the length is n + (n+1)(n-1)! - 3 + runs + (letters of joins beyond three).
    """
    a, n = _letters(word)
    L = len(a)
    T = math.factorial(n - 1)
    windows = np.lib.stride_tricks.sliding_window_view(a, n)
    full = (1 << n) - 1
    masks = np.bitwise_or.reduce(np.left_shift(1, windows.astype(np.int64)), axis=1)
    where = np.flatnonzero(masks == full)
    numbers = ranks(windows[where])
    _, first = np.unique(numbers, return_index=True)
    if len(first) != math.factorial(n):
        raise ValueError("not a superpermutation")
    first = np.sort(first)
    at = where[first]
    rows = windows[at]
    zero = np.argmax(rows == 0, axis=1)
    canon = np.take_along_axis(rows, (np.arange(n)[None, :] + zero[:, None]) % n, axis=1)
    cls = ranks(canon)
    arcs = []
    for i, c in zip(at.tolist(), cls.tolist()):
        if arcs and arcs[-1][2] == i - 1 and arcs[-1][0] == c:
            arcs[-1][2] = i
        else:
            arcs.append([c, i, i])
    joins = [arcs[j + 1][1] - arcs[j][2] for j in range(len(arcs) - 1)]
    runs, current = [], 1
    for j in joins:
        if j == 2:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    per_class = Counter(arc[0] for arc in arcs)
    return {
        "n": n, "length": L, "bound": math.factorial(n) + T + math.factorial(n - 2) + n - 3,
        "repeats": len(where) - len(first), "classes": T, "arcs": len(arcs),
        "classes_cut": sum(1 for v in per_class.values() if v > 1),
        "joins": dict(sorted(Counter(joins).items())), "runs": len(runs),
        "heavy_letters": sum(j - 3 for j in joins if j > 3),
    }
