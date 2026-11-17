"""Superpermutations: short words that hold every ordering of n symbols as n consecutive letters.

    from aniate import superpermutations

    word = superpermutations(5)                           # 153 letters, the shortest possible
    word = superpermutations(5, start="random")           # the solver from the orderings in a random order
    superpermutations.check(word)                         # every window scanned

    superpermutations.improve(word, seconds=60)           # a shorter word, when one is found
    superpermutations.improve("W-0111.detour", seconds=600)   # the same, for a Detour file

Aniate holds the world record for superpermutations on 14 symbols: W-0111, 93,904,779,622 letters, set by
Kabir Murjani on 17 November 2026, the state of the art on the Superpermutation Almanac.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .. import log
from ._jit import COMPILED
from .detour import read, write
from .orderings import MAX_SYMBOLS
from .words import Check, check, ledger, normalise, recursive

__all__ = ["find", "Word", "Detour", "Check", "check", "improve", "ledger", "read", "write", "recursive", "normalise",
           "SHORTEST", "bound"]

SHORTEST = {1: 1, 2: 3, 3: 9, 4: 33, 5: 153, 6: 872}
"""The shortest possible length, where it is proved."""


def bound(n: int) -> int:
    """n! + (n-1)! + (n-2)! + n - 3, below which no superpermutation on n symbols exists (n >= 3)."""
    return math.factorial(n) + math.factorial(n - 1) + math.factorial(n - 2) + n - 3


@dataclass(frozen=True)
class Word:
    """A superpermutation found by the solver, written with the symbols 0, 1, 2 ..."""
    text: str
    n: int
    start: int          # letters of the word the search began from
    seconds: float      # how long the search ran

    @property
    def length(self) -> int:
        return len(self.text)

    @property
    def saved(self) -> int:
        return self.start - self.length

    @property
    def shortest(self) -> int | None:
        return SHORTEST.get(self.n)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        note = ""
        if self.shortest == self.length:
            note = ", the shortest possible"
        elif self.shortest:
            note = f", {self.length - self.shortest} above the shortest possible"
        return f"Superpermutation on {self.n} symbols: {self.length:,} letters{note}"


@dataclass(frozen=True)
class Detour:
    """The result of improving a Detour file."""
    path: str | None    # the file written, or None when no saving was found
    n: int
    length: int         # letters of the improved word
    start: int          # letters of the given word
    segments: int
    seconds: float

    @property
    def saved(self) -> int:
        return self.start - self.length

    def __str__(self):
        head = f"Detour word on {self.n} symbols: {self.length:,} letters"
        if not self.saved:
            return f"{head}, no saving found among {self.segments:,} segments"
        return f"{head}, {self.saved:,} fewer than the given word ({self.segments:,} segments), written to {self.path}"


def _budget(n, seconds):
    if seconds is None:
        return min(600.0, math.factorial(n) / 20)
    if seconds < 0:
        raise ValueError("seconds must not be negative")
    return float(seconds)


def _slow_note():
    if not COMPILED:
        log.warning("superperm", "numba is not installed, so the search runs as plain Python",
                    "pip install 'aniate[fast]' for a search 50 to 100 times faster")


def _search(n, seed, start, seconds, on_better=None):
    from .search import Search
    began = time.perf_counter()
    s = Search(n, seed=seed)
    if start == "random":
        s.start_random()
    else:
        s.start_word(start)
    first = s.length
    if seconds > 0 and s.length > SHORTEST.get(n, 0):
        _slow_note()
        s.solve(seconds, target=SHORTEST.get(n), on_better=on_better)
    word = Word(s.verify(), n, first, time.perf_counter() - began)
    log.info("superperm", f"{n} symbols", f"{first:,} -> {word.length:,} letters", f"{word.seconds:.2f} s")
    return word


def superpermutations(n: int, *, start=None, seconds: float | None = None, seed: int = 0) -> Word:
    """A short superpermutation on n symbols.

    start    None for the recursive construction (the shortest possible up to n = 5), "random" for every
             ordering in a random order, or a word to begin from.
    seconds  how long the exact chains run. None means none from the construction, and n!/20 seconds
             (at most ten minutes) from a random order or a given word. The search stops early once it
             reaches the shortest possible length, where that is known.
    seed     random seed for the search.
    """
    if not isinstance(n, int) or not 1 <= n <= MAX_SYMBOLS:
        raise ValueError(f"n must be a whole number from 1 to {MAX_SYMBOLS}; larger words are improved from their Detour files")
    if start is None:
        text = recursive(n)
        if n < 3 or not seconds:
            return Word(text, n, len(text), 0.0)
        return _search(n, seed, text, float(seconds))
    if n < 3:
        raise ValueError("the search needs at least 3 symbols")
    if isinstance(start, str) and start != "random":
        found = check(start)
        if found.n != n or not found.complete:
            raise ValueError(f"start is not a superpermutation on {n} symbols: {found}")
    return _search(n, seed, start, _budget(n, seconds))


def improve(word, *, seconds: float | None = None, seed: int = 0, out=None):
    """Shorten a word with exact chains.

    word     a superpermutation as text (or a Word), searched over its orderings; or the path of a Detour
             file, searched over its segments without writing the word out (numba is required).
    seconds  how long to search. By default n!/20 seconds for text, and ten minutes for a Detour file.
    out      for a Detour file, where to write a shorter word. By default beside the given file, named by
             its length. Nothing is written when no saving is found.
    """
    if isinstance(word, (str, os.PathLike)) and str(word).endswith(".detour"):
        return _improve_detour(Path(word), 600.0 if seconds is None else float(seconds), seed, out)
    found = check(word)
    if not found.complete:
        raise ValueError(str(found))
    return _search(found.n, seed, word.text if isinstance(word, Word) else str(word), _budget(found.n, seconds))


def _improve_detour(path, seconds, seed, out):
    from .detour import Segments, splice
    from .segments import SegmentSearch
    began = time.perf_counter()
    segments = Segments(path)
    search = SegmentSearch(segments)
    log.info("superperm", f"{segments.n} symbols", f"{segments.letters:,} letters", f"{segments.nseg:,} segments")
    saved = search.run(seconds, seed=seed)
    length = segments.letters - saved
    written = None
    if saved > 0:
        written = str(out or path.with_name(f"{path.stem}-{length}.detour"))
        total = splice(segments, search.order, written)
        again = Segments(written)
        if total != length or again.letters != length or again.windows != segments.windows or (again.hist[:4] < segments.hist[:4]).any():
            raise AssertionError(f"the written word does not read back as expected: {total:,} letters, expected {length:,}")
    result = Detour(written, segments.n, length, segments.letters, segments.nseg, time.perf_counter() - began)
    log.info("superperm", str(result))
    return result


class _Callable(ModuleType):
    def __call__(self, *args, **kwargs):
        """A short superpermutation: the same as `superpermutations.find`."""
        return superpermutations(*args, **kwargs)


find = superpermutations
sys.modules[__name__].__class__ = _Callable
