"""ant(): the installation report and self-test."""

from __future__ import annotations

import io

import aniate
from aniate import ant, selfcheck


def report(monkeypatch=None, missing=()):
    if monkeypatch is not None:
        real = selfcheck._installed
        monkeypatch.setattr(selfcheck, "_installed", lambda name: None if name in missing else real(name))
    buf = io.StringIO()
    return ant(stream=buf), buf.getvalue()


def test_reports_versions_runs_the_self_test_and_confirms():
    working, text = report()
    assert working
    assert "1 1 2 3 5 8 13" in text
    assert f"aniate {aniate.__version__}" in text
    for name in ("python", "numpy", "scipy", "matplotlib", "pytest"):
        assert name in text
    assert "check passed" in text and "PDF plot written" in text
    assert text.rstrip().endswith("aniate is working.")


def test_missing_components_are_reported(monkeypatch):
    working, text = report(monkeypatch, missing={"matplotlib"})
    assert working and "plots unavailable" in text and "PDF plot" not in text

    working, text = report(monkeypatch, missing={"matplotlib", "scipy"})
    assert not working
    assert "missing" in text and text.rstrip().endswith("see the lines above.")
