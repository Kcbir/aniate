"""Superpermutations: the construction, the checker, the search over orderings, and the Detour format."""
import math

import pytest

from aniate import superpermutations as sp
from aniate.superpermutations import _jit


@pytest.mark.parametrize("n", range(1, 8))
def test_the_construction_holds_every_ordering(n):
    word = sp(n)
    found = sp.check(word.text)
    assert found.complete and found.orderings == math.factorial(n)
    assert word.length == sum(math.factorial(k) for k in range(1, n + 1))


def test_the_construction_is_shortest_up_to_five():
    for n in range(1, 6):
        assert sp(n).length == sp.SHORTEST[n]


def test_the_check_finds_what_is_missing():
    assert sp.check("123121321").complete
    found = sp.check("1231213")
    assert not found and found.missing == 2
    assert sp.check("bacbabcab").complete


def test_the_search_reaches_the_shortest_from_a_random_order():
    word = sp(4, start="random", seconds=10, seed=1)
    assert word.length == 33 and sp.check(word.text).complete
    assert word.start > word.length


def test_improve_keeps_a_valid_word_and_never_lengthens_it():
    start = sp(5, start="random", seconds=0.2, seed=2)
    better = sp.improve(start.text, seconds=0.5)
    assert sp.check(better.text).complete and better.length <= start.length


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        sp(0)
    with pytest.raises(ValueError):
        sp(4, start="0123")
    with pytest.raises(ValueError):
        sp.improve("0120")


def test_detour_round_trip(tmp_path):
    word = sp(6)
    path = tmp_path / "w.detour"
    assert sp.write(word, path) == word.length
    assert sp.read(path) == word.text


@pytest.mark.skipif(not _jit.COMPILED, reason="segments need numba")
def test_segments_of_a_detour_word_rewrite_to_a_valid_word(tmp_path):
    import numpy as np
    from aniate.superpermutations.detour import Segments, splice

    word = sp(7)
    path = tmp_path / "w.detour"
    sp.write(word, path)
    seg = Segments(path, T=3)
    assert seg.letters == word.length and seg.nseg > 2
    assert splice(seg, np.arange(seg.nseg), tmp_path / "same.detour") == word.length
    assert sp.read(tmp_path / "same.detour") == word.text
