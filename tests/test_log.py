"""Logging: levels, capture, and silence."""

from __future__ import annotations

import pytest

import aniate as an
from aniate import log, problems


def test_verbosity_get_set_and_validate():
    old = an.verbosity()
    try:
        assert an.verbosity("debug") == "debug" and an.verbosity() == "debug"
        with pytest.raises(ValueError, match="verbosity"):
            an.verbosity("loud")
    finally:
        an.verbosity(old)


def test_quiet_prints_nothing_and_info_prints_to_stderr(capsys):
    old = an.verbosity()
    try:
        an.verbosity("quiet")
        problems.chain().solve()
        assert capsys.readouterr().err == ""
        an.verbosity("info")
        problems.chain().solve()
        err = capsys.readouterr().err
        assert "aniate | model" in err and "aniate | solve" in err
        assert err.isascii()
    finally:
        an.verbosity(old)


def test_capture_restores_the_previous_level():
    before = an.verbosity()
    with log.capture("debug") as lines:
        problems.chain()
    assert lines and an.verbosity() == before
