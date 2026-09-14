"""The Fibonacci art: printed once on import, silenced by ANIATE_BANNER=0, shown by python -m aniate."""

from __future__ import annotations

import os
import subprocess
import sys

import aniate
from aniate.art import ART


def run(args, banner):
    env = {**os.environ, "ANIATE_BANNER": banner}
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, check=True)


def test_art_is_plain_ascii_fibonacci_with_no_names():
    assert ART.isascii()
    assert "1 1 2 3 5 8 13" in ART
    assert "aniate" not in ART.lower()


def test_import_prints_art_to_stderr_once_and_can_be_silenced():
    shown = run(["-c", "import aniate, aniate.vis, aniate"], "1")
    assert shown.stdout == ""
    assert shown.stderr.count("F(n+1) / F(n)") == 1
    assert "F(n+1)" not in run(["-c", "import aniate"], "0").stderr


def test_module_entry_point_prints_art_and_version():
    out = run(["-m", "aniate"], "0").stdout
    assert out.startswith(ART.lstrip("\n"))
    assert aniate.__version__ in out
    both = run(["-m", "aniate"], "1")
    assert (both.stdout + both.stderr).count("F(n+1) / F(n)") == 1
