"""The orderings of n symbols, numbered in lexicographic order, with their rotation classes and cheap steps."""
from __future__ import annotations

import itertools
import math

import numpy as np

SYMBOLS = "0123456789ABCDEF"
MAX_SYMBOLS = 10


def ranks(perms: np.ndarray) -> np.ndarray:
    """Lexicographic rank of every row of `perms`."""
    count, n = perms.shape
    rank = np.zeros(count, dtype=np.int64)
    for i in range(n - 1):
        smaller = (perms[:, i + 1:] < perms[:, i:i + 1]).sum(axis=1)
        rank += smaller * math.factorial(n - 1 - i)
    return rank


def all_orderings(n: int) -> np.ndarray:
    """Every ordering of 0 .. n-1 in lexicographic order."""
    out = np.zeros((1, 1), dtype=np.uint8)
    for size in range(2, n + 1):
        block = len(out)
        new = np.empty((block * size, size), dtype=np.uint8)
        for first in range(size):
            new[first * block:(first + 1) * block, 0] = first
            new[first * block:(first + 1) * block, 1:] = out + (out >= first)
        out = new
    return out


class Orderings:
    """Every ordering of n symbols.

    perms[p]      the ordering with number p
    inv[p, s]     where symbol s stands in ordering p
    class_of[p]   its rotation class, numbered 0 .. (n-1)! - 1
    out_nbr[p]    orderings reached from p by adding 1, 2, 2, 3 ... letters (the letters are in out_w)
    in_nbr[p]     orderings that reach p by adding that many letters
    """

    def __init__(self, n: int, max_letters: int = 3):
        if not 3 <= n <= MAX_SYMBOLS:
            raise ValueError(f"orderings are tabled for 3 to {MAX_SYMBOLS} symbols; got {n}")
        self.n = n
        self.N = math.factorial(n)
        self.perms = all_orderings(n)
        self.inv = np.argsort(self.perms, axis=1).astype(np.uint8)

        index = (np.arange(n)[None, :] + self.inv[:, :1].astype(np.int64)) % n
        canonical = np.take_along_axis(self.perms, index, axis=1)
        _, self.class_of = np.unique(ranks(canonical), return_inverse=True)
        self.class_of = self.class_of.astype(np.int32)

        out_cols, in_cols, weights = [], [], []
        for letters in range(1, min(max_letters, n - 1) + 1):
            arrangements = sorted(itertools.permutations(range(letters)), key=lambda a: a != tuple(reversed(range(letters))))
            for a in arrangements:
                a = list(a)
                target = np.concatenate([self.perms[:, letters:], self.perms[:, :letters][:, a]], axis=1)
                source = np.concatenate([self.perms[:, n - letters:][:, a], self.perms[:, :n - letters]], axis=1)
                out_cols.append(ranks(target))
                in_cols.append(ranks(source))
                weights.append(letters)
        self.out_nbr = np.stack(out_cols, axis=1).astype(np.int32)
        self.in_nbr = np.stack(in_cols, axis=1).astype(np.int32)
        self.out_w = np.array(weights, dtype=np.int32)

    def text(self, p: int) -> str:
        return "".join(SYMBOLS[s] for s in self.perms[p])
